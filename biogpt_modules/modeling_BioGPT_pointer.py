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


_CHECKPOINT_FOR_DOC = "microsoft/biogpt"
_CONFIG_FOR_DOC = "BioGptConfig"
class BioGptForCausalLMAddPointer(BioGptPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["output_projection.weight"]

    def __init__(self, config, loss_func= None, add_pointer = False):
        super().__init__(config)

        self.biogpt = BioGptModel(config)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        
        self.add_pointer = add_pointer
        if self.add_pointer:
            self._init_pointer()
        
        #self._reset_context_length()
        self._set_loss_func(loss_func)
    

    def _init_pointer(self, ):
        input_size = self.config.hidden_size * 3
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

    def _pointer(self, context_input_ids, target_ids_length, last_hidden_state, past_key_values, attentions, is_training = True):
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
        
        first_key_states = past_key_values[0][0]
        first_key_states = first_key_states.transpose(1,2)
        batch_size, seq_length = first_key_states.shape[0], first_key_states.shape[1]
        first_key_states = first_key_states.reshape(batch_size, seq_length, self.config.hidden_size)
        
        last_key_states = past_key_values[-1][0]
        last_key_states = first_key_states.transpose(1,2)
        last_key_states = first_key_states.reshape(batch_size, seq_length, self.config.hidden_size)
        
        last_value_states = past_key_values[-1][1]
        last_value_states = last_value_states.transpose(1,2)
        last_value_states = last_value_states.reshape(batch_size, seq_length, self.config.hidden_size)

        #inputs_embeds = outputs
        context_hiddens = last_key_states[:,:context_sequence_length,:]
        context_hiddens_attended = torch.matmul(src_trg_attention_sum, context_hiddens)
        
        pointer_inputs = torch.cat((context_hiddens_attended, first_key_states[:,-target_ids_length:,:], last_value_states[:,-target_ids_length:,:]), dim=-1)
        
        p_gen = torch.sigmoid(self.pointer(pointer_inputs))
        #print(p_gen.shape, 'p_gen shape')

        #distribution = F.softmax(outputs.logits, dim=-1)
        prediction_scores = self.output_projection(last_hidden_state)
       
        distribution = F.softmax(prediction_scores,dim = -1)
        past_scores = p_gen * distribution[:,-target_ids_length:,:]
        
        #self.src_trg_attention_mean = torch.mean(past_cross_attentions, dim = 1)
            #self.cross_attention = self.src_trg_attention_mean.detach().cpu()

        if len(src_trg_attention_mean.size()) > 3:
                src_trg_attention_mean = src_trg_attention_mean.squeeze()

        src_trg_attention_mean *= (1-p_gen)

        # Fix ids dimension 
       
        #print('input ids shape before ', context_input_ids.shape)
        if len(context_input_ids.size()) < 3:
                context_input_ids = context_input_ids.unsqueeze(1)
        
        #print('input ids shape after ', context_input_ids.shape)

        src_ids = tile(context_input_ids, src_trg_attention_mean.size(1), dim=1)
        #print('src_ids shape', src_ids.size())
        
        final_prediction_scores = past_scores.scatter_add_(dim=2, index=src_ids, src=src_trg_attention_mean)
        #prediction_scores *= p
        #print('final_prediction scores ', final_prediction_scores.shape)
        #if is_training:
        return final_prediction_scores, p_gen
        #else:
            #prediction_scores[:,context_sequence_length:,:] = final_prediction_scores
            #print('prediction scores ', prediction_scores.shape)
            #return prediction_scores, p_gen

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        context_input_ids: Optional[torch.LongTensor] = None,
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
        if type(context_input_ids ) != list:
                context_input_ids = [context_input_ids]
                
        prediction_scores = []
        if past_key_values is None:
            past_key_values = [None] * len(context_input_ids)
        if attentions is None:
            attentions = [None] * len(context_input_ids)
            
        for i in range(len(context_input_ids)):
                
            current_past_key_values = past_key_values[i]
            if current_past_key_values is None:
                concat_input_ids = torch.cat((context_input_ids[i][0].to(self.device), input_ids), dim=1)
                

            concat_attention_mask = torch.cat((context_input_ids[i][1].to(self.device), attention_mask), dim=1)
            if concat_input_ids.shape[1] < self.config.max_position_embeddings:
                input_ids = concat_input_ids
                attention_mask = concat_attention_mask
            
                #else:
                    #attention_mask = None
                    
                #print('len of past key values list', len(past_key_values_list))
                #print(current_past_key_values[0][0].shape)
                
            outputs = self.biogpt(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=current_past_key_values,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=return_dict,)
                
            if attentions[i] is None:
                attentions[i] = outputs.attentions[-1]
                    
            else:
                attentions[i] = torch.cat((attentions[i], outputs.attentions[-1]),dim=-2)
                    
                
            past_key_values[i] = outputs.past_key_values
                #prediction_scores, attentions = self._pointer(context_input_ids, outputs, attentions=attentions[i], is_training = labels is not None)
            
            if self.add_pointer:
                prediction_scores_, p_gen = self._pointer(context_input_ids[i][0].to(self.device), target_ids_length, last_hidden_state=outputs[0], past_key_values=outputs.past_key_values, attentions=attentions[i], is_training = labels is not None)
                #if len(context_input_ids) > 1:
                        #prediction_scores_ *= p_gen
                #prediction_scores.append(prediction_scores_.unsqueeze(-1))
                #else:
                    #prediction_scores += (prediction_scores_ * p_gen)
            else:
                
                sequence_output = outputs[0][:,-target_ids_length:,:]
                prediction_scores_ = F.softmax(self.output_projection(sequence_output), dim=-1)/len(context_input_ids)
            
            prediction_scores.append(prediction_scores_.unsqueeze(-1))
        
        if len(prediction_scores) > 1:
            prediction_scores = torch.cat(prediction_scores, dim=-1).mean(-1)
        else:
            prediction_scores = prediction_scores[0].squeeze(-1)

        prediction_scores = torch.log(prediction_scores + 0.0000003)
        
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


