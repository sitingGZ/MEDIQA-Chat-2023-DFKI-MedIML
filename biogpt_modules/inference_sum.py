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


TaskA_LABELS_reverted = {v:k for k,v in TaskA_LABELS.items()}

name_base = "microsoft/biogpt"


SEED_OF_RUNS = {0: 42, 1: 99, 3: 1}

def main(inference_config_file, task = 'task_a_header', run=0):
    set_seed(SEED_OF_RUNS[run])
    configs = load_config(inference_config_file)

    init_model_path = configs['model']['biogpt_base']
    add_pointer = configs['model']['add_pointer']

    checkpoint_dir = configs['inference']['checkpoint_dir'].format(SEED_OF_RUNS[run])
    #"/netscratch/iml_liang/nlp/new_gpt_biogpt_base_predict_header_chunking_0_target_length_50_context_length_300_section_header_add_pointer_False_update_last_layers_-2/seed_42_chunking_valid_0/"
    model_averaged_state = average_checkpoints(checkpoint_dir)

    biogpt_chat2note = GPT_Chat2Note(configs, )
    biogpt_chat2note.load_state_dict(model_averaged_state['state_dict'])

    path_test = configs['test']['test_dir']
    task_a_test = pd.read_csv(os.path.join(path_test, 'TaskA', 'taskA_testset4participants_inputConversations.csv'))
    task_b_test = 
    for i in range(len()): 
    prompt = 'Summarize the section of <'
    dialogue = dialogues_taska_test[i]
    context_input_ids_list, input_ids = inference_tokenize_for_header(dialogue, context_seq_length=300, tokenizer=biogpt_tokenizer, prompt=prompt)
    
    model_args = {'input_ids': input_ids, 'context_input_ids': context_input_ids_list, 'use_cache' : True}
    print(input_ids.shape)
    context_seq_length = context_input_ids_list[0][0].shape[1]
    print(context_seq_length)
    generates = biogpt_trainer.model.generate(max_length = context_seq_length + input_ids.shape[1] + 100, **model_args).detach().numpy()
    current_prediction = biogpt_tokenizer.batch_decode(generates)
    prediction = biogpt_tokenizer.batch_decode(generates[:,input_ids.shape[1]:])
    print(i, current_prediction)
    print(prediction)
    if 'Finish.' in prediction:
        last_pred = prediction.split('Finish.')[0].split('summary:')[-1]
        print(last_pred)
    