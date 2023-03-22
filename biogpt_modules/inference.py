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


def inference_tokenize_for_task_a(dialogue, context_seq_length, tokenizer:BioGptTokenizer, prompt: str = None, prompt_ids: torch.Tensor = None):
    if type(dialogue) != list:
        dialogue = [dialogue]
        
    if prompt_ids is None:
        prompt = 'The header of this section is <' if prompt is None else prompt
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
    try:
        result = label_dict[prediction]
    except:
        result = 'GENHX'
        #print(prediction, 'has problem ')
        #results_header[i] = 'GENHX({})'.format(prediction)
    return result


def post_process_sum(prediction, finish = False):  
    if type(prediction) == list:
        prediction = prediction[0]       
    if 'Finish.' in prediction:
        prediction = prediction.split('Finish.')[0]
        finish = True        
    if finish:    
        prediction = prediction.split('summary:')[-1]
    prediction = prediction.strip('</s>')
    return prediction, finish


SEED_OF_RUNS = {0: 42, 1: 99, 3: 1}

def main(inference_config_file, test_path='', task = 'predict_header', run=0):

    set_seed(SEED_OF_RUNS[run])
    configs = load_config(inference_config_file)

    init_model_path = configs['model']['biogpt_base']
    add_pointer = configs['model']['add_pointer']

    #"/netscratch/iml_liang/nlp/new_gpt_biogpt_base_predict_header_chunking_0_target_length_50_context_length_300_section_header_add_pointer_False_update_last_layers_-2/seed_42_chunking_valid_0/"
    model_averaged_state = average_checkpoints(checkpoint_dir)

    biogpt_chat2note = GPT_Chat2Note(configs, )
    biogpt_chat2note.load_state_dict(model_averaged_state['state_dict'])

    #test_path = 'TaskA/taskA_testset4participants_inputConversations.csv'
    test task_a_test = pd.read_csv(test_path)
    dialogues_taska_test = [clear_task_a_dialogue(d, chunking=0) for d in task_a_test['dialogue']]

    # task is either 'predict_header' or 'summarization'
    checkpoint_dir_predict_header = configs['inference']['checkpoint_dir']['predict_header'].format(SEED_OF_RUNS[run])
    checkpoint_dir_summarization = configs['inference']['checkpoint_idr']['summarization'].format(SEED_OF_RUNS)

    
    results = []
    







    






if __name__== '__main__':

    import sys
    #config_path = 'configs/transformer_config.yaml'
    #main(config_path)
    assert len(sys.argv) >= 2, "The path to the config file must be given as argument!"
    main(sys.argv[1])
