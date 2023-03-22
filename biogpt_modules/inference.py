from inspect import indentsize
import os
import pandas as pd
import json
from itertools import chain
from typing import Optional, List, Dict, Tuple, Union, Callable

from copy import deepcopy
from random import shuffle

import torch
from torch.utils.data import DataLoader as batch_gen
from helpers import find_best_checkpoint, load_config, set_seed, average_checkpoints


from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from modeling_BioGPT_pointer import GPT_Chat2Note, BioGptForCausalLMAddPointer
from tokenization_biogpt import BioGptTokenizer

from train_biogpt_sum import PROMPTS, TaskA_LABELS, clear_task_a_dialogue, make_task_a_datalist


TaskA_LABELS_reverted = {''.join(l.lower().split()):k for k,l in TaskA_LABELS.items()}

name_base = "microsoft/biogpt"
save_result = "taskA_DFKI-MedIML_run{}.csv" 


def inference_tokenize_for_task_a(dialogue, context_seq_length, tokenizer:BioGptTokenizer, prompt: str = None, prompt_ids: torch.Tensor = None):
    if type(dialogue) != list:
        dialogue = [dialogue]
        
    if prompt_ids is None:
        #prompt = 'The header of this section is <' if prompt is None else prompt
        assert prompt is not None, "prompt is requried, for header prediction:'The header of this section is < ' or for summarization 'Summarize the section of < '."
        prompt_ids = tokenizer([prompt], return_tensors = 'pt').input_ids[:,1:]
    
    d_ids = tokenizer(dialogue, return_tensors = 'pt').input_ids
    context_input_ids = []
    current_length = d_ids.size(1)
    while current_length > context_seq_length:
        d_ids = d_ids[:,:context_seq_length]
        
        context_input_ids.append((d_ids, None))
        current_length = d_ids.size(1)
        
    #if current_length < context_seq_length:
    context_input_ids.append((d_ids, None))
  
    return context_input_ids, prompt_ids

def post_process_header(prediction, label_dict):
    prediction = ''.join(prediction[0].split())
    prediction = prediction.split('>.')[0]
    print(prediction)
    try:
        result = label_dict[prediction]
    except:
        result = 'GENHX'

    return result

def predict(input_ids, context_input_ids_list, model, tokenizer, target_length=20):

    context_input_ids_list = [(inputs[0].to(model.device), None) for inputs in context_input_ids_list]
    model_args = {'input_ids': input_ids.to(model.device), 'context_input_ids': context_input_ids_list, 'use_cache' : True}
    #print(input_ids.shape)
    context_seq_length = context_input_ids_list[0][0].shape[1]
    #print(context_seq_length)
    generates = model.generate(max_length = context_seq_length + input_ids.shape[1] + target_length, **model_args).detach().cpu().numpy()
    current_prediction = tokenizer.batch_decode(generates)
    prediction = tokenizer.batch_decode(generates[:,input_ids.shape[1]:])
    #print(i, current_prediction)
    #print(prediction)
    return prediction

def post_process_sum(prediction, finish = False):  
    if type(prediction) == list:
        prediction = prediction[0]   

    if 'Finish.' in prediction:
        prediction = prediction.split('Finish.')[0]
        finish = True        
    #if :    
    prediction = prediction.split('summary:')[-1]
    prediction = prediction.strip('</s>')
    return prediction, finish

HEADER_RUNS = {1: 42, 1: 99, 2: 1}
CONTEXT_LENGTHS = {1:100, 2:300, 3: 500}

def main(inference_config_file='biogpt_inference.yaml', run=0, test_path='taskA_testset4participants_inputConversations.csv'):
    
    seed = 42
    #if task == 'predict_header':
    #   seed = HEADER_RUNS[]

    set_seed(seed)
    configs = load_config(inference_config_file)

    init_model_path = configs['model']['gpt']['biogpt_base']

    biogpt_tokenizer = BioGptTokenizer.from_pretrained(init_model_path)
    chat2note = GPT_Chat2Note(configs, init_model_path=init_model_path, tokenizer=biogpt_tokenizer, add_pointer=False, add_context_hidden=False)

    #"/netscratch/iml_liang/nlp/new_gpt_biogpt_base_predict_header_chunking_0_target_length_50_context_length_300_section_header_add_pointer_False_update_last_layers_-2/seed_42_chunking_valid_0/"
    #test_path = 'TaskA/taskA_testset4participants_inputConversations.csv'
    task_a_test = pd.read_csv(test_path)
    dialogues_taska_test = [clear_task_a_dialogue(d, chunking=0) for d in task_a_test['dialogue']]

    # task is either 'predict_header' or 'summarization'
    
    checkpoint_predict_header = configs['checkpoints']['predict_header'].format(seed)
    checkpoint_summarization = configs['checkpoints']['summarization'].format(CONTEXT_LENGTHS[run])

    # Predict headers
    model_state = torch.load(checkpoint_predict_header,map_location=torch.device('cpu') )
    chat2note.load_state_dict(model_state['state_dict'])
    chat2note.model.eval()
    if torch.cuda.is_available():
        chat2note.model.cuda()
    
    headers = []
    prompt = 'The header of this section is <'
    prompt_ids = None
    for i in range(len(dialogues_taska_test)):
        dialogue = dialogues_taska_test[i]
        context_input_ids_list, prompt_ids = inference_tokenize_for_task_a(dialogue, context_seq_length=300, tokenizer=biogpt_tokenizer, prompt=prompt, prompt_ids = prompt_ids)
        if len(context_input_ids_list) > 1:
            context_input_ids_list = context_input_ids_list[:1]
        prediction = predict(prompt_ids, context_input_ids_list, chat2note.model, biogpt_tokenizer)
        header = post_process_header(prediction, TaskA_LABELS_reverted)
        headers.append(header)
    
    results = [{"TestID" : i, "SystemOutput1": headers[i]} for i in range(len(dialogues_taska_test))]
    
    save_result = "taskA_header_DFKI-MedIML_run{}.csv" 
    df = pd.DataFrame(results)
    df.to_csv(save_result.format(run))

    del model_state
    # Summarize section

    model_state = torch.load(checkpoint_summarization,map_location=torch.device('cpu') )
    chat2note.load_state_dict(model_state['state_dict'])
    chat2note.model.eval()
    if torch.cuda.is_available():
        chat2note.cuda()

    sums = []
   
    prompt_ids = None
    for i in range(len(dialogues_taska_test)):
        dialogue = dialogues_taska_test[i]
        prompt = 'Summarize the section of <{}> . Here is the <first> part of the summary: '.format(TaskA_LABELS[headers[i]].lower())
        context_input_ids_list, prompt_ids = inference_tokenize_for_task_a(dialogue, context_seq_length=300, tokenizer=biogpt_tokenizer, prompt=prompt, prompt_ids = None)
        prediction = predict(prompt_ids, context_input_ids_list, chat2note.model, biogpt_tokenizer, target_length=100)
        p, finish = post_process_sum(prediction)
        sums.append(p)

    results = [{"TestID" : i, "SystemOutput1": headers[i], "SystemOutput2": sums[i]} for i in range(len(dialogues_taska_test))]
    
    save_result = "taskA_DFKI-MedIML_run{}.csv" 
    df = pd.DataFrame(results)
    df.to_csv(save_result.format(run))

if __name__== '__main__':

    import sys
    #config_path = 'configs/transformer_config.yaml'
    #main(config_path)
    assert len(sys.argv) >= 3, "The path to the config file must be given as argument!"
    main(sys.argv[1], 2, sys.argv[2])
