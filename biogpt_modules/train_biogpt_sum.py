#from lib2to3.pgen2 import token
from inspect import indentsize
import os
import pickle as pkl
import pandas as pd
import json
from itertools import chain
from typing import Optional, List, Dict, Tuple, Union, Callable

from copy import deepcopy
from random import shuffle

import torch
from torch.utils.data import DataLoader as batch_gen
from helpers import find_best_checkpoint, load_config, set_seed, robust_decode, robust_encode

from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from modeling_BioGPT_pointer import GPT_Chat2Note
from tokenization_biogpt import BioGptTokenizer

import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

"""
def read_data(src_file, trg_file=None):
    data =  {'source_data': [robust_decode(l).strip() for l in open(src_file, 'rb')]}
    if trg_file is not None:
        data['target_data'] = [robust_decode(l).strip() for l in open(trg_file, 'rb')]
    return data

def shuffle_data(current_src: dict, current_trg:dict):
    indices = list(current_src.keys())
    shuffle(indices)
    new_current_data = {'source_data': [current_src[i] for i in indices], 'target_data':[current_trg[i] for i in indices]}
    return new_current_data
"""
PROMPTS = {'TaskA': {'section_header': "The header of this section is <{}>.", 'section_text': "The header of this section is <{}> and summarize this section: "}, 'TaskB': "The header of this section is <{}> and summarize this section "}
TaskA_LABELS = {'GENHX': "HISTORY of PRESENT ILLNESS",
  'MEDICATIONS':'MEDICATIONS' ,
  'CC': "CHIEF COMPLAINT",
  'PASTMEDICALHX': "PAST MEDICAL HISTORY",
  'ALLERGY':'ALLERGY' ,
  'FAM/SOCHX':"FAMILY HISTORY/SOCIAL HISTORY" ,
  'PASTSURGICAL': "PAST SURGICAL HISTORY",
  'OTHER_HISTORY':  'OTHER_HISTORY',
  'ASSESSMENT':'ASSESSMENT' ,
  'ROS': "REVIEW OF SYSTEMS" ,
  'DISPOSITION': 'DISPOSITION',
  'EXAM': "EXAM",
  'PLAN': "PLAN",
  'DIAGNOSIS':'DIAGNOSIS' ,
  'EDCOURSE': "EMERGENCY DEPARTMENT COURSE" ,
  'IMMUNIZATIONS':'IMMUNIZATIONS' ,
  'LABS': "LABS",
  'IMAGING':'IMAGING' ,
  'PROCEDURES': 'PROCEDURES' ,
  'GYNHX': "GYNECOLOGIC HISTORY"}

def load_data(src_pkl_file, trg_pkl_file=None):
    src_data = pkl.load(open(src_pkl_file,'rb'))
    trg_data = pkl.load(open(trg_pkl_file, 'rb')) if trg_pkl_file else None
    return src_data, trg_data

def clear_task_a_dialogue(s, chunking=6, splitor = '\n'):
    """

    """
    if chunking > 0:
        s_splited = s.split(splitor)
        #print(len(s_splited))
        s = []
        # Starts new chunk with the previous three sentences
        for i in range(0, len(s_splited), 3):
            end = min(i+chunking, len(s_splited))
            s.append(''.join(s_splited[i:end]))
            if end == len(s_splited):
                break
        #print(len(s), 'after chunking')
    else:
        s = s.replace(splitor, '')

    return s

def clear_task_b_dialogue(s, chunking = 6, splitor = '\n'):

    if chunking  > 0:
        s_splited = s.split(splitor)
        s = []
        # Starts new chunk with the previous three sentences
        for i in range(0, len(s_splited), 3):
            end = min(i+chunking, len(s_splited))
            s.append(''.join(s_splited[i:end]))
            if end == len(s_splited):
                break

    else:
        s = s.replace(splitor, '').replace('[','').replace(']', ':')
    return s

def split_note_sections(text):
    pattern = r'[A-Z :*]+\n\n'
    sections = re.split(pattern, text)
    headers = re.findall(pattern, text)
    
    if 'HIV\n\n' in text:
        headers = [h for h in headers if 'HIV' not in h]
    sections = [section.strip() for section in sections if section.strip()]
    pairs = []
    for h, s in zip(headers, sections):
        h = h.strip('\n\n')
        h = h +':' if h[-1] != ':' else h
        s = s.replace('\n','')
        pairs.append((h,s))
    #print(len(sections), len(headers))
    return pairs


class MedicalChatDataset(torch.utils.data.Dataset):
    """Construct data set for dataloader"""

    def __init__(self, source_list, target_list, header_list):
        
        """
        Source data sample: 

        """
        self.source = source_list
        self.target = target_list
        self.header = header_list
      
        
    def __getitem__(self, index:int)->Dict:
        """
        :param index : int
        """
        
        result = {'source':  self.source[index], 'target':self.target[index], 'header':self.header[index]}
        
        return result
    
    def __len__(self):
        return len(self.source)

def make_task_a_datalist(task_a_df, target_input, chunking = 0, splitor='\n'):

    # Chunking is 0 for summarization task
    source_list = [clear_task_a_dialogue(s, chunking=0, splitor=splitor) for s in task_a_df['dialogue'] ]
    target_list = [PROMPTS['TaskA'][target_input].format(TaskA_LABELS[s].lower()) for s in task_a_df['section_header']]
    header_list = [TaskA_LABELS[s].lower() for s in task_a_df['section_header']]
    if target_input == 'section_text':
        target_list = task_a_df['section_text']
    #prompt = 
    if chunking > 0:
        new_source_list = []
        new_target_list = []
        new_header_list = []
        for s_list, t ,h in zip(source_list, target_list, header_list):
            if target_input == 'section_header':
                for s_ in s_list:
                    new_source_list.append(s_)
                    new_target_list.append(t)
                    new_header_list.append(h)
            else:
                new_source_list.append('\t'.join(s_list))
                new_target_list.append(t)
                new_header_list.append(h)


        return new_source_list, new_target_list, new_header_list
    else: 
        return source_list, target_list, header_list

def make_task_b_datalist(task_b_df, chunking=6, splitor='\n'):
    source_list = [clear_task_b_dialogue(s, chunking=chunking, splitor='\n') for s in task_a_df['dialogue'] ]
    # split_sections 
    sections = [split_note_sections(note) for note in task_b_df['note']]
    #for section in sections: 
    #_LABELS[s]) if target_input == 'section_header' else '<{}>'.format(TaskA_LABELS[s])+t for s,t in zip(task_a_df['section_header'], task_a_df['section_text']) ]
    prompt = PROMPTS['TaskB']
    return source_list, target_list

def main(config_file):
    # Prepare configurations
    configs = load_config(config_file)
    #seeds= configs['train'].get('random_seeds', [1,42, 99])

    # language of pre-trained BERT: english, german or multilingual
    #batch_size = configs['train']['batch_size']

    gpus = 1
    
    task_name = configs['data']['task']['name']

    task = 'TaskA'
  
    source_input = configs['data']['task'][task]['source']
    target_input = configs['data']['task'][task]['target']

    train_csv = configs['data']['train'].format(task, task)
    valid_csv = configs['data']['valid'].format(task, task)



    chunking = configs['data']['chunking']
    chunking_valid = configs['data']['chunking_valid']

    task_df_train = pd.read_csv(train_csv)
    task_df_valid = pd.read_csv(valid_csv)

    source_list, target_list, header_list = make_task_a_datalist(task_df_train, target_input, chunking=chunking, splitor='\r\n')
    train_dataset = MedicalChatDataset(source_list=source_list, target_list = target_list, header_list=header_list)
    
    source_list, target_list, header_list = make_task_a_datalist(task_df_valid, target_input, chunking = chunking_valid, splitor='\n')
    valid_dataset = MedicalChatDataset(source_list=source_list, target_list = target_list, header_list = header_list)

    bert_languages = configs['bert_languages']
 
    seeds = [42, 99, 1]
    #pointer = 
    
    for seed in seeds:
        torch.cuda.empty_cache()
        set_seed(seed=seed)
        
        for bert_language, train_params in bert_languages.items():
            init_model_path = configs['model']['gpt'][bert_language]
            add_pointer = configs['model']['add_pointer']
            add_context_hidden = configs['model']['add_context_hidden']
            batch_size = train_params['batch_size']
            workers = train_params['workers']
            
            tokenizer = BioGptTokenizer.from_pretrained(init_model_path)
            #new_tokens = list(language_codes.values()) 
            #tokenizer.add_tokens(new_tokens)
            
            biogpt_trainer = GPT_Chat2Note(configs, add_pointer=add_pointer, add_context_hidden=add_context_hidden, init_model_path=init_model_path, tokenizer=tokenizer, task=task_name)
            biogpt_trainer.freeze_parameters()
            #biogpt_trainer.model.cpu()
            biogpt_trainer.init_optimizer_grouped_parameters()
            
            if biogpt_trainer.add_pointer:
                biogpt_trainer.update_pointer_parameters()
            #else:
            biogpt_trainer.update_layers_parameters(configs['train']['update_last_layers'])

                #biogpt_trainer.update_output_projection_parameters()

            if "pre_checkpoints" in configs['train']:
                    ckpt_paths = configs['train']['pre_checkpoints']

            target_length = configs['model'].get('target_seq_length', -1)
            context_length = configs['model'].get('context_seq_length', -1)
            save_path = configs['train']['save_path'].format(add_context_hidden, bert_language, task_name+'_chunking_{}_target_length_{}_context_length_{}'.format(chunking, target_length, context_length), target_input, add_pointer, configs['train']['update_last_layers'])
                    #save_path = model_path.format(mlm_pretrained, mlm_training, name + '_update_encoder_{}_we_{}'.format(update_encoder, update_we))
            if not os.path.exists(save_path):
                os.mkdir(save_path)
          
            train_batch = batch_gen(train_dataset, batch_size = 1, num_workers = 1)
            valid_batch = batch_gen(valid_dataset, batch_size = 1, num_workers = 1)

            # debug mode
            #for i in range(len(train_dataset)):
                #current_batch = {k:[v] for k,v in train_dataset[i].items()}
                #print(' current batch ', i)
                #train_batch, target_input_list = biogpt_trainer._tokenize(current_batch)
        
                #outputs = biogpt_trainer(train_batch, target_input_list)
                #outputs = 
            cur_model_dir = os.path.join(save_path, 'seed_{}_chunking_valid_{}'.format(seed, chunking_valid))
            
            checkpoint_callback = callbacks.ModelCheckpoint(monitor='val_loss', dirpath=cur_model_dir, mode = 'min',
                                                    filename='seq2seq-{epoch:02d}-{val_loss:.5f}', save_top_k=2, save_weights_only=True)
            gpu_trainer = Trainer(gpus=gpus, gradient_clip_val = 1.0, stochastic_weight_avg=True, max_epochs=10,callbacks=checkpoint_callback, precision=16)      
            
            gpu_trainer.fit(biogpt_trainer, train_batch,valid_batch)
            #cpu_trainer = Trainer(accelerator='cpu', devices=2, gradient_clip_val = 1.0,max_epochs = 3, callbacks=checkpoint_callback)
                    #current_bert2bert = deepcopy(bert2bert)
            #cpu_trainer.fit(biogpt_trainer, train_batch, valid_batch)

            del gpu_trainer, biogpt_trainer
                    #print('Best model path of task {}'.format(task + round), checkpoint_callback.best_model_path)
                    #del current_bert2bert, trainer
            gc.collect()
            torch.cuda.empty_cache()

                    # reset tokenizer for anther dataset
                   

if __name__ == '__main__':
    import sys
    #config_path = 'configs/transformer_config.yaml'
    #main(config_path)
    assert len(sys.argv) >= 2, "The path to the config file must be given as argument!"
    main(sys.argv[1])
