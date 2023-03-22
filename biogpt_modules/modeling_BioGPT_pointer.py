from helpers import tile

import torch
from torch import nn
import torch.nn.functional as F

from typing import Optional, List, Dict, Tuple, Union, Callable

from modeling_BioGPT import BioGptModel, BioGptForCausalLM, BioGptPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from transformers import AutoTokenizer
import pytorch_lightning as pl
from transformers import (AdamW, WEIGHTS_NAME, get_linear_schedule_with_warmup)

import gc


_CHECKPOINT_FOR_DOC = "microsoft/biogpt"
_CONFIG_FOR_DOC = "BioGptConfig"

CONVERT_IDX = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',  'last' ]
SUMMARIZATION_PROMPTS={'top': 'Summarize the section of <{}>.', 'part': 'Here is the <{}> part of the summary: ', 'end': 'Finish.', 'chunk':'...'}


class BioGptForCausalLMAddPointer(BioGptPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["output_projection.weight"]

    def __init__(self, config, loss_func= None, add_pointer = False, add_context_hidden=False):
        super().__init__(config)

        self.biogpt = BioGptModel(config)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        self.add_pointer = add_pointer
        self.add_context_hidden = add_context_hidden

        if self.add_pointer:
            self._init_pointer()
        
        #self._reset_context_length()
        self._set_loss_func(loss_func)
    

    def _init_pointer(self, ):
        input_size = self.config.hidden_size * 2
        self.pointer = nn.Linear(input_size, 1, bias=True)
        self.add_pointer = True
        
    def _set_loss_func(self, loss_func):
        """
        Use NLLoss that takes log_softmax output as input instead of logits
        :param loss_func:
        :param pad_token_id:
        :return:
        """
        if loss_func is None:
                loss_func = nn.NLLLoss(ignore_index = self.config.pad_token_id)
                
            #loss_func = nn.CrossEntropyLoss(ignore_index = pad_token_id)
        self.loss_func = loss_func

    def _reset_context_length(self, context_sequence_length = None):
        self.context_sequence_length = context_sequence_length

    def get_output_embeddings(self):
        return self.output_projection

    def set_output_embeddings(self, new_embeddings):
        self.output_projection = new_embeddings

    #@add_start_docstrings_to_model_forward(BIOGPT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@add_code_sample_docstrings(
    #    checkpoint=_CHECKPOINT_FOR_DOC,
    #    output_type=CausalLMOutputWithCrossAttentions,
    #    config_class=_CONFIG_FOR_DOC,
    #)

    def _pointer(self, previous_prediction_scores=None, context_input_ids=None, target_ids_length = None, previous_hidden_state = None, current_hidden_state = None, last_key_states = None, attentions = None, is_training = True):
        """
        Chunks of context_input_ids, inputs_embeds, attentions (last layer), and outputs
        Length of inputs_embeds == Length of outputs
        Length of inputs_embeds > Length of context_input_ids
        """
        #if self.context_sequence_length is None:
        context_sequence_length = context_input_ids.size(1)
        #print(last_hidden_state.shape, target_ids_length, 'last_hidden_state.shape and target_ids_length ')
        #print('current outputs attentions shape ', attentions.shape)
        src_trg_attention = attentions[:,:,-target_ids_length:,:context_sequence_length]

        #print('src_trg_attention shape', src_trg_attention.shape, )
        
        src_trg_attention_sum = src_trg_attention.sum(1)
        src_trg_attention_mean = src_trg_attention.mean(1)
        
        if len(src_trg_attention_sum.size()) > 3:
                src_trg_attention_sum = src_trg_attention_sum.squeeze(1)
                #print('src_trg_atteion shape', src_trg_attention.shape)
        
        #first_key_states = past_key_values[0][0]
        #first_key_states = first_key_states.transpose(1,2)
        
        #first_key_states = first_key_states.reshape(batch_size, seq_length, self.config.hidden_size)
        
        #last_key_states = past_key_values[-1][0]
        if previous_hidden_state is None:
            last_key_states = last_key_states.transpose(1,2)
            batch_size, seq_length = last_key_states.shape[0], last_key_states.shape[1]
            last_key_states = last_key_states.reshape(batch_size, seq_length, self.config.hidden_size)
        
            context_hiddens = last_key_states[:,:context_sequence_length,:]
            context_hiddens_attended = torch.matmul(src_trg_attention_sum, context_hiddens)
        

        
            pointer_inputs = torch.cat((context_hiddens_attended, current_hidden_state), dim=-1)

        else:
            pointer_inputs = torch.cat((previous_hidden_state, current_hidden_state), dim=-1)
        
        p_gen = torch.sigmoid(self.pointer(pointer_inputs))
        #print(p_gen.shape, 'p_gen shape')
        #distribution = F.softmax(outputs.logits, dim=-1)
        prediction_scores_ = self.output_projection(current_hidden_state)
        distribution = F.softmax(prediction_scores_,dim = -1)

        #distribution[:,-target_ids_length:,:] 
        #past_scores = p_gen * previous_predition_scores
        
        #self.src_trg_attention_mean = torch.mean(past_cross_attentions, dim = 1)
            #self.cross_attention = self.src_trg_attention_mean.detach().cpu()

        #distribution += previous_prediction_scores 
        distribution *= p_gen
        
        if previous_prediction_scores is None:
            

            src_trg_attention_mean *= (1-p_gen)**2

            if len(context_input_ids.size()) < 3:
                context_input_ids = context_input_ids.unsqueeze(1)

            src_ids = tile(context_input_ids, src_trg_attention_mean.size(1), dim=1)
            #print('src_ids shape', src_ids.size())
            current_prediction_scores = distribution.scatter_add_(dim=2, index=src_ids, src=src_trg_attention_mean)

        else:
            previous_prediction_scores *= (1-p_gen)**2
            current_prediction_scores = previous_prediction_scores + distribution

        return current_prediction_scores, p_gen

    def _add_context_hidden(self,context_seq_length,  target_ids_length, current_hidden_state = None, attentions = None ):
        
        batch_size = current_hidden_state.shape[0]
        target_hidden = current_hidden_state[:,-target_ids_length:,:]
        context_hidden = current_hidden_state[:,:context_seq_length, :]
       
        src_trg_attention = attentions[:,:,-target_ids_length:,:context_seq_length]

        #src_trg_attention = src_trg_attention.transpose(1,2)
        
        #src_trg_attention_reshaped = src_trg_attention.reshape(batch_size, target_ids_length, int(context_sequence_length * self.config.num_attention_heads))
        src_trg_attention_sum = src_trg_attention.sum(1)
        #src_trg_attention_mean = src_trg_attention.mean(1)
        
        #if len(src_trg_attention_sum.size()) > 3:
        #        src_trg_attention_sum = src_trg_attention_sum.squeeze(1)

        #last_key_states = last_key_states.transpose(1,2)
        #batch_size, seq_length = last_key_states.shape[0], last_key_states.shape[1]
        #last_key_states = last_key_states.reshape(batch_size, seq_length, self.config.hidden_size)
        
        #context_states = last_key_states[:,:context_sequence_length,:]
        context_hidden_attended = torch.matmul(src_trg_attention_sum, context_hidden)
        
        #context_hiddens_attended = torch.matmul(src_trg_attention_reshaped, context_states_reshaped)
        averaged_hiddens = torch.cat((context_hidden_attended.unsqueeze(-1), target_hidden.unsqueeze(-1)), dim=-1).mean(-1)
        
        return averaged_hiddens

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        context_input_ids= None,
        generated_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attentions: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        past_attentions (context_input_ids)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if labels is not None:
            labels = labels.to(self.device)
            
        if generated_ids is None:
            generated_ids = input_ids.to(self.device)
        
        target_ids_length = input_ids.shape[1]
            
        #current_attention_mask = None
            
        #if self.add_pointer:
        if type(context_input_ids) != list:
                context_input_ids = [context_input_ids]
                
        if past_key_values is None:
            past_key_values = [None] * len(context_input_ids)

        if attentions is None:
            attentions = [None] * len(context_input_ids)

        #context_input_hiddens = [None] * len(context_input_ids)
        context_input_hiddens = []

        # The first context_input_ids
        start_past_key_values = past_key_values[0]
        if start_past_key_values is None:
            start_input_ids = torch.cat((context_input_ids[0][0].to(self.device), input_ids), dim=1)
        else:
            start_input_ids = input_ids

        outputs = self.biogpt(
            start_input_ids,
            attention_mask=None,   # attention mask is always None
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=start_past_key_values,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,)

        past_key_values[0] = outputs.past_key_values

        if attentions[0] is None:
                attentions[0] = outputs.attentions[-1].detach().cpu()
                    
        else:
                attentions[0] = torch.cat((attentions[0], outputs.attentions[-1].detach().cpu()),dim=-2)
        
        context_seq_length = start_input_ids.shape[1]
        if self.add_context_hidden:
            current_hidden = self._add_context_hidden(context_seq_length=context_seq_length,  target_ids_length = target_ids_length, current_hidden_state = outputs[0], attentions=attentions[0].to(self.device) )
            context_input_hiddens.append(current_hidden.unsqueeze(-1))
        else:
            context_input_hiddens.append(outputs[0][:,-target_ids_length:,:].unsqueeze(-1))
            
                    
        for i in range(1, len(context_input_ids)):
            #print('idx of context input ids ', i)

            current_past_key_values = past_key_values[i]
            if current_past_key_values is None:
                current_input_ids = torch.cat((context_input_ids[i][0].to(self.device), input_ids), dim=1)
            else:
                current_input_ids = input_ids

            
                
            #current_attention_mask = torch.cat((context_input_ids[i][1].to(self.device), attention_mask), dim=1)
            #print('context input ids and attention shape ', context_input_ids[i][0].shape, context_input_ids[i][1].shape)
            #print('input ids and attention shape {} {}, current_input_ids_shape {}, current_attention_mask shape{} '.format(input_ids.shape, attention_mask.shape, current_input_ids.shape, current_attention_mask.shape) )
            #if any([t <0 or t>= self.config.vocab_size for t in current_input_ids[0]]):
            #    print(i, current_input_ids, 'has out of index token')
            
            outputs = self.biogpt(
            current_input_ids,
            attention_mask=None,   # attention mask is always None
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=current_past_key_values,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,)
                
            if attentions[i] is None:
                attentions[i] = outputs.attentions[-1].detach().cpu()
                    
            else:
                attentions[i] = torch.cat((attentions[i], outputs.attentions[-1].detach().cpu()),dim=-2)
                    
            past_key_values[i] = outputs.past_key_values
                #prediction_scores, attentions = self._pointer(context_input_ids, outputs, attentions=attentions[i], is_training = labels is not None)

            if self.add_context_hidden:
                current_hidden = self._add_context_hidden(context_seq_length=context_seq_length,  target_ids_length = target_ids_length, current_hidden_state = outputs[0], attentions=attentions[i].to(self.device) )
                context_input_hiddens.append(current_hidden.unsqueeze(-1))
            else:
                context_input_hiddens.append(outputs[0][:,-target_ids_length:,:].unsqueeze(-1))

        if self.add_pointer:
            #prediction_scores = []
            start_hidden = context_input_hiddens[0]
            #pre_prediction_scores = F.softmax(self.output_projection(start_hidden), dim=-1)
            dis, p_gen = self._pointer(None, context_input_ids[0][0].to(self.device), target_ids_length, previous_hidden_state=None, current_hidden_state=start_hidden, last_key_states=past_key_values[0][-1][0], attentions=attentions[0].to(self.device), is_training = labels is not None)
            pre_prediction_scores = None
            if len(context_input_hiddens) > 1:
                for t in range(1, len(context_input_hiddens)):
                    #if t > 0:
                    pre_prediction_scores = dis
                    start_hidden = context_input_hiddens[t-1]
                    #current_hidden = context_input_hiddens[t]
                    pre_prediction_scores, p_gen = self._pointer(pre_prediction_scores, context_input_ids[t][0].to(self.device), target_ids_length, previous_hidden_state=start_hidden, current_hidden_state=context_input_hiddens[t], last_key_states=past_key_values[t][-1][0], attentions=attentions[t].to(self.device), is_training = labels is not None)
               
            if pre_prediction_scores is not None:
                dis = pre_prediction_scores
            
                #if len(context_input_ids) > 1:
                        #prediction_scores_ *= p_gen
                #prediction_scores.append(prediction_scores_.unsqueeze(-1))

                #else:
                    #prediction_scores += (prediction_scores_ * p_gen)
        else:

            #prediction_scores = [F.softmax(self.output_projection(sequence_output), dim=-1).unsqueeze(-1) for sequence_output in context_input_hiddens]
        
            #if len(prediction_scores) > 1:
            #    prediction_scores = torch.cat(prediction_scores, dim=-1).mean(-1)

            #else:
            #    prediction_scores = prediction_scores[0].squeeze(-1)

            if len(context_input_hiddens) > 1:
                hiddens = torch.cat(context_input_hiddens, dim=-1).mean(-1)
        
            else:
                hiddens = context_input_hiddens[0].squeeze(-1)

            dis = F.softmax(self.output_projection(hiddens), dim=-1)

        prediction_scores = torch.log(dis + 0.0000003)
        
        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            #if labels.shape[-1] != prediction_scores.shape[1], "Labels have a length of {} that is different to the length of the output length {}".format(labels.shape[-1], prediction_scores.shape[1])
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            
            lm_loss = self.loss_func(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output
        
        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=attentions,
            cross_attentions=outputs.cross_attentions,
        )
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask, past_key_values=None, attentions=None, **kwargs):

        # only last token for inputs_ids if past is defined in kwargs
        #kwargs.[context_input_ids = input_ids[:,:-1]
        generated_ids = kwargs.get('generated_ids')
        if generated_ids is None:
            generated_ids = input_ids
        else:
            generated_ids = torch.cat((generated_ids, input_ids[:,-1]), dim=1)
                
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "context_input_ids": kwargs.get("context_input_ids"),
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "attentions":attentions,
            "use_cache": kwargs.get("use_cache"),
            "generated_ids": generated_ids,

        }


    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class GPT_Chat2Note(pl.LightningModule):
    """
    Pytorch lightning train and inference module of integrating BioGpt(AddPointer)
    """
    def __init__(self, config_dict, add_pointer, add_context_hidden, init_model_path:str, tokenizer:AutoTokenizer,  loss_func=None, pointer_ratio = 0.,vocab_weight= None, task='summarization'):
        """
        Initialize model and configurations defined in the config file.
        :param config_file:
        :param loss_func: 
        """
        super().__init__()
        #config_dict = self._load_configs(config_file)
        config_dict = self._load_configs(config_dict)
        self.save_hyperparameters()

        self.add_pointer = add_pointer
        self.add_context_hidden = add_context_hidden
        
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token':'<pad>'})
        
        self.model = BioGptForCausalLMAddPointer.from_pretrained(init_model_path)

        if self.add_pointer:
            self.model._init_pointer()

        self.model.add_context_hidden = add_context_hidden

        # Summarization or header_prediction
        self.task = task
        
        self.max_model_length = min(self.model_config.get('max_model_length', 500), self.model.config.max_position_embeddings/2)
        
        self.target_seq_length = self.model_config.get('target_seq_length', 50)
        self.context_seq_length = min(self.model_config.get('context_seq_length', 100), self.max_model_length - self.target_seq_length)
        
        self.prompts_tokens = {'parts':[]}
        
        chunk = self.tokenizer(['...'], return_token_type_ids=False, return_tensors='pt')
        self.prompts_tokens['chunk'] = (chunk.input_ids[:,1:], chunk.attention_mask[:,1:])
        
        #print(' vocabulary length after added new tokens ', len(self.tokenizer.vocab)) 
        self.output_size = self.tokenizer.vocab_size
        
        self.convert_idx = CONVERT_IDX
        self.summarization_prompts= SUMMARIZATION_PROMPTS
    
    def _model_special_tokens(self, bos_id=None, eos_id=None, pad_id = None):
        """
        """
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id
        self.pad_token_id = pad_id 

    def _load_configs(self, config):
        """
        Load the data, model, training, testing configs from one yaml file
        :param config_file: 
        :return: 
        """
        #config = load_config(config_file)
        self.data_config = config['data']
        self.model_config = config['model']
        self.train_config = config['train']
        #self.test_config = config['test']
        self.model_dir = self.train_config['save_path']
        return config

    def forward_one_batch(self, context_input_ids = None, input_ids = None, inputs_embeds = None, attention_mask = None, 
                    past_key_values=None,
                    labels = None, use_cache=True, inference = True):
        """

        :param input_ids: tokenized ids of source text
        :param inputs_embeds: output from the embeddings of encoder
        :param attention_mask: ignore the pad token ids in encoder attention
        :param past_key_values: 
        :param labels: target ids
        :param use_cache:
        :param output_attentions: true if add pointer
        :param output_hidden_states: true if add pointer
        :return:
        """
        #print('self.model devices in forward', self.model.device, self.model.encoder.device, self.model.decoder.device)

        
        #if context_input_ids is not None:
            #context_input_ids = context_input_ids.to(self.model.device)
        #if torch.cuda.is_available():
            #self.model.cuda()

        if input_ids is not None:
            input_ids = input_ids[:,:self.model.config.max_position_embeddings]
        if inputs_embeds is not None:
             inputs_embeds=inputs_embeds[:,:self.model.config.max_position_embeddings]
        if attention_mask is not None:
            attention_mask = attention_mask[:,:self.model.config.max_position_embeddings]
        if labels is not None:
            labels = labels[:, :self.model.config.max_position_embeddings]

        # Set attention mask to be None                   
        outputs = self.model(context_input_ids = context_input_ids,  input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,  labels = labels, use_cache=use_cache, 
                                            past_key_values = past_key_values, return_dict=True)
            
      
        return outputs
    
    
    def _prepare_summarization_prompts(self, header, parts=2):
        """
        Tokenize the target prompts for summarization task
        """
        top = self.tokenizer([self.summarization_prompts['top'].format(header)], return_token_type_ids=False, return_tensors='pt')
        self.prompts_tokens['top'] = (top.input_ids, top.attention_mask)

        #print(top.input_ids.shape, top.attention_mask.shape, 'prompt tops ')
        
        if 'chunk' not in self.prompts_tokens:
            chunk = self.tokenizer([self.summarization_prompts['chunk']], return_token_type_ids=False, return_tensors='pt')
            self.prompts_tokens['chunk'] = (chunk.input_ids[:,1:], chunk.attention_mask[:,1:])
            
        if 'end' not in self.prompts_tokens:
            end = self.tokenizer([self.summarization_prompts['end']+self.tokenizer.eos_token], return_token_type_ids=False, return_tensors='pt')
            self.prompts_tokens['end'] = (end.input_ids[:, 1:], end.attention_mask[:,1:])
            
        
        for i in range(parts):
            # end with the parts limit
            if i == 0:
                 part = self.summarization_prompts['top'].format(header) + self.summarization_prompts['part'].format(self.convert_idx[0]) 
                    
            else:
                if i < parts-1:
                    part = self.summarization_prompts['part'].format(self.convert_idx[i]) 
                else:
                    part = self.summarization_prompts['part'].format(self.convert_idx[-1])
            
            part = self.tokenizer([part], return_token_type_ids=False, return_tensors='pt')
            self.prompts_tokens['parts'].append((part.input_ids[:,1:], part.attention_mask[:, 1:]))

    def _tokenize(self, batch):
        """
        :param batch: dict
        batch ['source']
        batch ['target']
        batch ['header']
        Currently only takes one sequence in one batch: batch_size = 1
        maximum context length = 100
        maximum target length = 50
        """

        target_inputs =  self.tokenizer(batch['target'], return_token_type_ids=False, return_tensors='pt')
        total_length = target_inputs.input_ids.shape[1]
        #print('target inputs shape ', target_inputs.input_ids.shape)

        source = batch['source'][0].split('\t')
        #print('source sents', len(source), source[0])

        source_inputs = [self.tokenizer(source[i:i+1],return_token_type_ids=False, return_tensors = 'pt') for i in range(len(source))]
        #context_input_ids = [(inputs.input_ids[:, :self.max_model_length], inputs.attention_mask[:,:self.max_model_length]) for inputs in source_inputs]
        #print(len(source_inputs), len(context_input_ids))
        context_input_ids = self._chunk_context_input_ids(source_inputs)

        parts = min(int(total_length / self.target_seq_length) +1, len(self.convert_idx))
        
        header = batch['header'][0]
        #if type(header) == list:
            #header = header[0]

        target_inputs_list = [(target_inputs.input_ids[:, 1:], target_inputs.attention_mask[:,1:])]

        if "sum" in self.task:
            target_inputs_list = []

            self._prepare_summarization_prompts(header,parts = parts)
            
            #print('current context input ids ', len(context_input_ids))
            for i in range(parts):
                #print(parts, i, 'current parts')    
                # Start after the bos token
                start = i*self.target_seq_length+1
                end = start + self.target_seq_length

                prompt = self.prompts_tokens['parts'][i]
            
                input_ids = torch.cat((prompt[0], target_inputs.input_ids[:, start:end]),dim=1) 
                attention_mask = torch.cat((prompt[1], target_inputs.attention_mask[:, start:end]), dim=1)
                #print('current input and attention shape ', input_ids.shape, attention_mask.shape)
        
                if end >= total_length:
                    input_ids = torch.cat((input_ids, self.prompts_tokens['end'][0]), dim=1)
                    attention_mask = torch.cat((attention_mask, self.prompts_tokens['end'][1]), dim= 1)
                
                else:
                    #if i == parts-1:
                    # The last part is still shorter than total length
                    input_ids = torch.cat((input_ids, self.prompts_tokens['chunk'][0]), dim=1)
                    attention_mask = torch.cat((attention_mask, self.prompts_tokens['chunk'][1]), dim= 1)
            
                target_inputs_list.append((input_ids, attention_mask))

            #print('lengt target inputs list ', len(target_inputs_list))
            
                    #print(i, current_length, 'the final context input ids ')

        #print('context input_ids length', len(context_input_ids))
        train_batch = {'context_input_ids': context_input_ids, 'input_ids': target_inputs_list[0][0],  'attention_mask': target_inputs_list[0][1], 'labels': target_inputs_list[0][0]}
        
        return train_batch, target_inputs_list

    def _chunk_context_input_ids(self, source_inputs):
        
        context_input_ids = []

        current_context_input_ids = source_inputs[0].input_ids
        current_context_attention_mask = source_inputs[0].attention_mask
        current_length = current_context_input_ids.shape[1]

        current_length = self.context_seq_length
               
        #if current_length < self.context_seq_length:
        if len(source_inputs) > 1:
                for i in range(len(source_inputs)-1):
                    current_context_input_ids = torch.cat((current_context_input_ids, source_inputs[i+1].input_ids), dim = 1)
                    current_context_attention_mask = torch.cat((current_context_attention_mask, source_inputs[i+1].input_ids), dim =1)
            
        current_length = current_context_input_ids.shape[1]
      
        while current_length > self.context_seq_length:
                    part_ids = torch.cat((current_context_input_ids[:,:self.context_seq_length], self.prompts_tokens['chunk'][0]), dim=1)
                    part_mask = torch.cat((current_context_attention_mask[:,:self.context_seq_length], self.prompts_tokens['chunk'][1]), dim=1)
                    context_input_ids.append((part_ids, part_mask))
                                          
                    current_context_input_ids = current_context_input_ids[:,self.context_seq_length:]
                    current_context_attention_mask = current_context_attention_mask[:, self.context_seq_length:]
                    current_length = current_context_input_ids.shape[1]
                            #print(i, current_length, 'length is now cutting .. ')                                           
            #if i == len(source_inputs) - 2:
        context_input_ids.append((current_context_input_ids, current_context_attention_mask))
        current_length = current_context_input_ids.shape[1]
        return context_input_ids

    

    def _predict_header(self, dialogues, prompt_ids=None):

        self.model.eval()
        prompt = 'The header of this section is <'
        #list_of_input_ids, prompt_ids = self._inference_tokenize(dialogues, prompt=prompt, prompt_ids=prompt_ids)

        results = {}
        for i, input_ids in enumerate(list_of_input_ids):
            model_args = {'input_ids': input_ids, 'context_input_ids': input_ids[:,:-10], 'use_cache' : True}
            generates = self.model.generate(max_length = input_ids.shape[1] + 100, **model_args).detach().numpy()[0]
            #current_prediction = self.tokenizer.batch_decode(generates)
            prediction = self.tokenizer.batch_decode(generates[input_ids.shape[1] +1 :-2])
        
            results[i] = prediction

        return results

    def _dialoge_to_note(self, context_input_ids, prompt_ids):

        return 


    def forward(self, train_batch, target_input_list=None, is_training = True):
        #if self.model.add_pointer:
        #self.model.cpu()
        outputs = self.forward_one_batch(**train_batch)
        if target_input_list is not None and len(target_input_list) > 1:
            #print('length of target input list', len(target_input_list))
            for i in range(1,len(target_input_list)):
                #print('part of target {}'.format(i+1))
                context_input_ids = [target_input_list[i-1]] + train_batch['context_input_ids']
                #print('i, context input ids ', i, len(context_input_ids))
                train_batch = {'context_input_ids': context_input_ids, 'input_ids': target_input_list[i][0],  'attention_mask': target_input_list[i][1], 'labels': target_input_list[i][0]}
                further_outputs = self.forward_one_batch(**train_batch)
                outputs.loss += further_outputs.loss

        return outputs


    def training_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        #print(len(batch))
        
        train_batch, target_input_list = self._tokenize(batch)
        outputs = self(train_batch, target_input_list)
        #else:
        #    self.model.base_model.embed_tokens.cpu
        #    self.model.base_model.embed_positions.cpu()

        return outputs.loss

    def validation_step(self, batch, batch_idx):

        
        train_batch, target_input_list = self._tokenize(batch)
        
        outputs = self(train_batch, target_input_list)

        self.log('val_loss', outputs.loss)
        
        
    def test_step(self, batch, batch_idx):
        
        src_batch, trg_batch = self._tokenize(batch)

        #outputs = self(input_ids = src_batch['input_ids'], attention_mask = src_batch['attention_mask'], decoder_input_ids = trg_batch['input_ids'], decoder_attention_mask = trg_batch['attention_mask'],
                                #labels = trg_batch['input_ids'])

        #self.log('test_loss', outputs.loss)

    def freeze_parameters(self,):
        """
        Freeze all the parameters of the self.model
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def update_layers_parameters(self, last_layers = -10):

        params_for_decay = []
        params_for_no_decay = []
        for layer in self.model.base_model.layers[last_layers:]:
            for n, param in layer.named_parameters():
                param.requires_grad = True
                if not any(nd in n for nd in self.no_decay):
                    params_for_decay.append(param)
                else:
                    params_for_no_decay.append(param)

        self.optimizer_grouped_parameters.extend([{"params": params_for_decay,  "weight_decay": self.weight_decay},
                {"params": params_for_no_decay,"weight_decay": 0.0,}])
        
    def update_pointer_parameters(self):
        self.model.pointer.requires_grad = True
        params = []
        for p in self.model.pointer.parameters():
            p.requires_grad = True
            params.append(p)

        self.optimizer_grouped_parameters.append({"params": params,
                "weight_decay": self.weight_decay})


    def update_output_projection_parameters(self):
        self.model.output_projection.requires_grad = True
        params = []
        for  p in self.model.output_projection.parameters():
            p.requires_grad = True
            params.append(p)

        self.optimizer_grouped_parameters.append({"params": params,
                "weight_decay": self.weight_decay})

    def init_optimizer_grouped_parameters(self):

        self.weight_decay = self.train_config.get("weight_decay", 0.3)
        self.no_decay = ["bias", "LayerNorm.weight"]
        self.optimizer_grouped_parameters = []


    def configure_optimizers(self):
        """
        :param config: training configs
        :return:
        """
        #optimizer = build_optimizer(self.train_config, optimizer_grouped_parameters)
        lr = float(self.train_config['learning_rate'])
        optimizer = AdamW(self.optimizer_grouped_parameters, lr=lr)
        #print('optimizer lr ', optimizer.param_groups[0]['lr'])
        #scheduler_mode = 'min'
        #scheduler, scheduler_step_at = build_scheduler(self.train_config, optimizer, scheduler_mode, self.model.decoder.config.hidden_size)
        #scheduler_config = {"scheduler": scheduler, "interval": scheduler_step_at, "frequency": 1}
        return optimizer

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings
        :return: string representation
        """
        return "%s"%(self.__class__.__name__, self.model)
