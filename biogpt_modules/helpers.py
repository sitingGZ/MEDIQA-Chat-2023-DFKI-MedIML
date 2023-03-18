import os
import yaml
from torch import nn, Tensor
import torch
import random
import numpy as np

def robust_decode(bs):
    '''Takes a byte string as param and convert it into a unicode one.
First tries UTF8, and fallback to Latin1 if it fails'''
    cr = None
    try:
        cr = bs.decode('utf-8')
    except UnicodeDecodeError:
        cr = bs.decode('latin-1')
    return cr

def robust_encode(bs):
    '''Encode a string as param and convert it into a unicode one.
First tries UTF8, and fallback to Latin1 if it fails'''
    cr = None
    try:
        cr = bs.encode('utf-8')
    except UnicodeEncodeError:
        cr = bs.encode('latin-1')
    return cr

def remove_umlaut(string):
    string = string.replace('ü', 'ue')
    string = string.replace('Ü', 'Ue')
    string = string.replace('ä' ,'ae')
    string = string.replace('Ä', 'Ae')
    string = string.replace('ö', 'oe')
    string = string.replace('Ö', 'Oe')
    string = string.replace('ß', 'ss')
    return string

def revert_umlaut(string):
    string = string.replace('ue', 'ü')
    string = string.replace('Ue','Ü')
    string = string.replace('ae','ä')
    string = string.replace('Ae', 'Ä')
    string = string.replace('oe', 'ö')
    string = string.replace('Oe', 'Ö')
    #string = string.replace('ss', 'ß')
    return string


def _nonlin_fct(nonlin):
    if nonlin == "tanh":
        return torch.tanh
    elif nonlin == "relu":
        return torch.relu
    elif nonlin == "gelu":
        return nn.functional.gelu
    elif nonlin == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError("Unsupported nonlinearity!")

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-9):
        super().__init__()
        self.mean_square_error = nn.MSELoss(reduction = 'mean')
        self.eps = eps
        
    def forward(self,x,y):
        loss = torch.sqrt(self.mean_square_error(x,y) + self.eps)
        return loss

def find_best_checkpoint(checkpoint_files):
    files = [f.split('=') for f in checkpoint_files]
    scores_and_files = {p[-1].strip('.ckpt'): '='.join(p) for p in files}
    scores_sorted=sorted(scores_and_files.keys())
    return scores_and_files[scores_sorted[0]]

def find_best_checkpoint_path(checkpoint_paths):
    list_to_dict = {}
    for path in checkpoint_paths:
        ckpts = os.listdir(path)
        for ckpt in ckpts:
            ckpt_path = os.path.join(path, ckpt)
            ckpt_loss = ckpt.split('=')[-1].strip('.ckpt')
            list_to_dict[ckpt_loss] = ckpt_path
    best_ckpt_key =  sorted(list(list_to_dict.keys()))[0]
    return list_to_dict[best_ckpt_key] 
    

def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Originally used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

def build_combine_vocab(eng_vocab, ger_vocab, eng_tokenizer, ger_tokenizer, check = False):  
    
    # German BERT has bigger vocab than English BERT, add the English vocab to the German tokenizer
    # eng_vocab = eng_tokenizer.vocab
    # ger_vocab = ger_tokenizer.vocab
    
    # If need to check the intersection between two vocab again for new inital BERT models
    added_vocab = set(eng_vocab.keys()) - set(ger_vocab.keys())
    if check:
        eng_tokenizer.add_tokens(list(ger_vocab.keys()))
        ger_tokenizer.add_tokens(list(eng_vocab.keys()))
        
        if len(eng_tokenizer.vocab) > len(ger_tokenizer.vocab):
            return eng_tokenizer, added_vocab, 'English'
        else:
            return ger_tokenizer, added_vocab, 'German'
    
    
    else:
        ger_tokenizer.add_tokens(list(eng_vocab.keys()))
        return ger_tokenizer, added_vocab, 'German'
    
    
def copy_word_embeddings(eng_embeddings, ger_embeddings, added_vocab, origin_vocab, extended_vocab, retain='German'):
    
    # Initial the resized German embeddings weight with weight from English embeddings
    if retain == 'German':
        embeddings_to_use = ger_embeddings
        embeddings_to_drop = eng_embeddings
    else:
        embeddings_to_use = eng_embeddings
        embeddings_to_drop = ger_embeddings
    
    
    assert embeddings_to_use.word_embeddings.weight.size(0) == len(extended_vocab), "The embeddings for further usage {} must be resized to be same as the length {} of the extended vocabulary.".format(embeddings_to_use.word_embeddings.weight.size(0), len(extended_vocab))
        
    #with torch.no_grad():
    for w in added_vocab:
            
            try:
                #print(w, extended_vocab[w], origin_vocab[w])
                new_idx = extended_vocab[w]
                old_idx = origin_vocab[w]
                embeddings_to_use.word_embeddings.weight[new_idx].data = embeddings_to_drop.word_embeddings.weight[old_idx].detach().clone()
            except:
                print("Token {} is not found in the vocab.".format(w))
                
