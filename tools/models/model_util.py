# ==================== Import ==================== #
import time
import sys
import os 

import numpy as np 
import json 
import math 
from collections import OrderedDict

import torch 
import torch.nn as nn 
import torchtext.vocab as vocab
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from nltk.tokenize import word_tokenize

from einops import rearrange, repeat
from torch import einsum 
from einops_exts import rearrange_many, repeat_many

from models.transformer_encoder_modified import TransformerEncoderLayer, TransformerEncoder

import warnings
warnings.filterwarnings('ignore')

# ==================== Functions ==================== # 
def l2norm(x):
    """L2-normalize columns of x"""
    return F.normalize(x, p=2, dim=-1)

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class I2DFormer_logits_enhanced_text(nn.Module):
    def __init__(self, feature_dim, pooling_method, temperature=np.log(1/0.07)):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones([1]))
        nn.init.constant_(self.logit_scale, temperature)

        if pooling_method == "max":
            self.pool_func = max_pool 
        elif pooling_method == "mean":
            self.pool_func = mean_pool 

        self.local_linear = nn.Linear(feature_dim, 1)

    def forward(self, feature_image_global, feature_text_global, attention_values_local):

        feature_image_global = feature_image_global / (feature_image_global.norm(dim=-1, keepdim=True))
        feature_text_global = feature_text_global / (feature_text_global.norm(dim=-1, keepdim=True) + 1e-10)

        logits_per_image = self.logit_scale * torch.matmul(feature_image_global.unsqueeze(1), feature_text_global.transpose(-1, -2))
        logits_per_image = logits_per_image.squeeze(1)
        
        values_local_pool = self.pool_func(attention_values_local, dim=2)
        logits_per_image_local = self.local_linear(values_local_pool).squeeze(2)

        return logits_per_image, logits_per_image_local
    
class GatedCrossAttention_modified(nn.Module):
    def __init__(
        self,
        *,
        dim,
        ff_mult=4,
    ):
        super().__init__()
        self.attn = Attention_layer(d_model=dim)
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        if ff_mult == 0:
            self.ff_flag = False
        else: 
            self.ff_flag = True

            self.ff = FeedForward(dim, mult=ff_mult)
            self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(self, query, key, key_padding_mask=None, return_score_flag=False):
        
        if return_score_flag:
            x_attn, scores = self.attn(query, key, key_padding_mask=key_padding_mask, return_score_flag=return_score_flag)
        else:
            x_attn = self.attn(query, key, key_padding_mask=key_padding_mask)

        x = ((x_attn * self.attn_gate.tanh()).transpose(1,0) + query)

        if self.ff_flag:
            x = self.ff(x) * self.ff_gate.tanh() + x

        if return_score_flag:
            return x, scores
        else:    
            return x
    
class Attention_layer_attention_logits(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()

        self.d_model = d_model

        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)

        attn_std = self.d_model ** -0.5
        nn.init.normal_(self.W_Q.weight, std=attn_std)
        nn.init.normal_(self.W_K.weight, std=attn_std)
        nn.init.normal_(self.W_K.weight, std=attn_std)

        self.ln = nn.LayerNorm(self.d_model)

    def get_attn_pad_mask(self, len_q, seq_k): # 

        batch_size, category_num, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(2) # [batch_size, category_num, 1, len_k]

        return pad_attn_mask.expand(batch_size, category_num, len_q, len_k)

    def forward(self, query, key, key_padding_mask=None, return_score_flag=False):

        query_new = self.W_Q(query)
        key_new = self.W_K(key)
        value_new = self.W_V(key)

        if key_padding_mask is not None:
            len_q = query_new.shape[1]
            key_padding_mask = key_padding_mask.expand(query.shape[0], key_padding_mask.shape[0], key_padding_mask.shape[1]) # [batch_size, category_num, len_k]
            src_key_padding_mask = self.get_attn_pad_mask(len_q, key_padding_mask)
        else:
            src_key_padding_mask = None

        scores = torch.matmul(query_new.unsqueeze(1), key_new.transpose(-2, -1)) / math.sqrt(self.d_model)

        if src_key_padding_mask is not None: 
            scores.masked_fill_(src_key_padding_mask==False, -1e9)

        scores_value = max_pool(scores, dim=2) 

        scores_value_repeat = repeat(scores_value, "a b c -> a b c d", d=value_new.shape[-2])

        enhanced_value = torch.matmul(scores_value_repeat, value_new)

        enhanced_value = self.ln(enhanced_value)

        if return_score_flag:
            return enhanced_value, scores
        else:
            return enhanced_value
        
class Attention_layer_enhanced_text(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
 
        self.d_model = d_model

        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)

        attn_std = self.d_model ** -0.5
        nn.init.normal_(self.W_Q.weight, std=attn_std)
        nn.init.normal_(self.W_K.weight, std=attn_std)
        nn.init.normal_(self.W_K.weight, std=attn_std)

    def get_attn_pad_mask(self, len_q, seq_k): # 

        batch_size, category_num, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(2) # [batch_size, category_num, 1, len_k]

        return pad_attn_mask.expand(batch_size, category_num, len_q, len_k)

    def forward(self, query, key, key_padding_mask=None, return_score_flag=False):

        query_new = self.W_Q(query)
        key_new = self.W_K(key)
        value_new = self.W_V(key)

        if key_padding_mask is not None:
            len_q = query_new.shape[1]
            key_padding_mask = key_padding_mask.expand(query.shape[0], key_padding_mask.shape[0], key_padding_mask.shape[1]) # [batch_size, category_num, len_k]
            src_key_padding_mask = self.get_attn_pad_mask(len_q, key_padding_mask)
        else:
            src_key_padding_mask = None

        scores = torch.matmul(query_new.unsqueeze(1), key_new.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        if src_key_padding_mask is not None: 
            scores.masked_fill_(src_key_padding_mask==False, -1e9)

        p_attn = F.softmax(scores, dim = -1)
        value_attention = torch.matmul(p_attn, value_new)

        if return_score_flag:
            return value_attention, scores
        else:
            return value_attention
    
class Text_Encoder(nn.Module):
    def __init__(self, context_length, transformer_width, transformer_layers, transformer_heads, positional_embedd_flag=False):
        super().__init__()
        
        encoder_layer = TransformerEncoderLayer(d_model=transformer_width, nhead=transformer_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.bos_embedding = nn.Embedding(1, transformer_width)
        self.transformer_width = transformer_width

        self.context_length = context_length

        self.positional_embedd_flag = positional_embedd_flag
        if positional_embedd_flag:
            print("\n\tusing positional embedding !")
            self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        else:
            print("\n\tDon't using positional embedding !")

        self.ln_final = nn.LayerNorm(transformer_width)

        self.init_weights()

    def init_weights(self):
        
        if self.positional_embedd_flag:
            nn.init.normal_(self.positional_embedding, std=0.01)
        
        nn.init.normal_(self.bos_embedding.weight, std=0.02)
    
    def get_attn_pad_mask(self, seq_q, seq_k):                       # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
        batch_size, len_q = seq_q.size()
        batch_size, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)          
        return pad_attn_mask.expand(batch_size, len_q, len_k)

    def forward(self, x, key_padding_mask=None):
        
        bos_embed = self.bos_embedding(torch.tensor(0).cuda()).expand(x.shape[0], 1, self.transformer_width)
        
        x = torch.cat((bos_embed, x), dim=1)
        
        if self.positional_embedd_flag:
            x = x + self.positional_embedding

        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        x = x.permute(1, 0, 2) 

        x = self.ln_final(x)

        return x 

class MLP_text(nn.Module):
    def __init__(self, d_model_in=300, d_model_out=64, d_model_mid=512, dropout_rate=0.3, text_mlp_res_flag=False):
        super().__init__()

        self.text_mlp_res_flag = text_mlp_res_flag

        if text_mlp_res_flag:
            scale = d_model_in ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(d_model_in, d_model_out))

        self.dropout = nn.Dropout(dropout_rate)

        self.ln_0 = nn.LayerNorm(d_model_in)

        self.linear_1 = nn.Linear(d_model_in, d_model_mid)
        self.ln_1 = nn.LayerNorm(d_model_mid)
        self.relu_1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(d_model_mid, d_model_out)
        self.ln_2 = nn.LayerNorm(d_model_out)
        self.relu_2 = nn.ReLU(inplace=True)

        
    def forward(self, x):
        
        if self.text_mlp_res_flag:
            x_res = x @ self.proj

        x = self.relu_1(self.linear_1(x))
        x = self.ln_1(x)
        x = self.dropout(x)

        x = self.relu_2(self.linear2(x))

        if self.text_mlp_res_flag:
            x = x + x_res

        x = self.ln_2(x)
        x = self.dropout(x)

        
        return x 

class MLP_image_layer_2(nn.Module):
    def __init__(self, d_model_in=768, d_model_out=64, d_model_mid=1024, dropout_rate=0.3, image_mlp_res_flag=False):
        super().__init__()

        self.image_mlp_res_flag = image_mlp_res_flag

        if image_mlp_res_flag:
            scale = d_model_in ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(d_model_in, d_model_out))

        self.dropout = nn.Dropout(dropout_rate)

        self.ln_0 = nn.LayerNorm(d_model_in)

        self.linear_1 = nn.Linear(d_model_in, d_model_mid)
        self.ln_1 = nn.LayerNorm(d_model_mid)
        self.relu_1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(d_model_mid, d_model_out)
        self.ln_2 = nn.LayerNorm(d_model_out)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):

        if self.image_mlp_res_flag:
            x_res = x @ self.proj
        
        x = self.relu_1(self.linear_1(x))
        x = self.ln_1(x)
        x = self.dropout(x)

        x = self.relu_2(self.linear2(x))

        if self.image_mlp_res_flag:
            x = x + x_res

        x = self.ln_2(x)
        x = self.dropout(x)

            
        return x 
        

class MLP_image_layer_3(nn.Module):
    def __init__(self, d_model_in=768, d_model_out=64, d_model_mid=1024, dropout_rate=0.3, image_mlp_res_flag=False):
        super().__init__()

        self.image_mlp_res_flag = image_mlp_res_flag
        if image_mlp_res_flag:
            scale = d_model_in ** -0.5
            self.proj = nn.Parameter(scale * torch.randn(d_model_in, d_model_out))

        self.dropout = nn.Dropout(dropout_rate)

        self.ln_0 = nn.LayerNorm(d_model_in)

        self.linear_1 = nn.Linear(d_model_in, d_model_mid)
        self.ln_1 = nn.LayerNorm(d_model_mid)
        self.relu_1 = nn.ReLU(inplace=True)

        self.linear2 = nn.Linear(d_model_mid, d_model_mid)
        self.ln_2 = nn.LayerNorm(d_model_mid)
        self.relu_2 = nn.ReLU(inplace=True)

        self.linear3 = nn.Linear(d_model_mid, d_model_out)
        self.ln_3 = nn.LayerNorm(d_model_out)
        self.relu_3 = nn.ReLU(inplace=True)

        
    def forward(self, x):

        if self.image_mlp_res_flag:
            x_res = x @ self.proj

        x = self.relu_1(self.linear_1(x))
        x = self.ln_1(x)
        x = self.dropout(x)

        x = self.relu_2(self.linear2(x))
        x = self.ln_2(x)
        x = self.dropout(x)

        x = self.relu_3(self.linear3(x))

        if self.image_mlp_res_flag:
            x = x + x_res

        x = self.ln_3(x)
        x = self.dropout(x)

        return x 


class Attention_layer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()

        self.d_model = d_model

        self.W_Q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_model, bias=False)

        attn_std = self.d_model ** -0.5
        nn.init.normal_(self.W_Q.weight, std=attn_std)
        nn.init.normal_(self.W_K.weight, std=attn_std)
        nn.init.normal_(self.W_K.weight, std=attn_std)

    def get_attn_pad_mask(self, len_q, seq_k): # 

        batch_size, category_num, len_k = seq_k.size()
        pad_attn_mask = seq_k.data.eq(0).unsqueeze(2) # [batch_size, category_num, 1, len_k]

        return pad_attn_mask.expand(batch_size, category_num, len_q, len_k)

    def forward(self, query, key, key_padding_mask=None, return_score_flag=False):

        query_new = self.W_Q(query)
        key_new = self.W_K(key)
        value_new = self.W_V(key)

    
        if key_padding_mask is not None:
            len_q = query_new.shape[1]
            key_padding_mask = key_padding_mask.expand(query.shape[0], key_padding_mask.shape[0], key_padding_mask.shape[1]) # [batch_size, category_num, len_k]
            src_key_padding_mask = self.get_attn_pad_mask(len_q, key_padding_mask)
        else:
            src_key_padding_mask = None

        scores = ((query_new @ key_new.view(-1, key_new.shape[-1]).transpose(-2, -1)) / math.sqrt(self.d_model)).view(query_new.shape[0], query_new.shape[1], key_new.shape[0], key_new.shape[1]).transpose(1, 2)

        if src_key_padding_mask is not None: 
            scores.masked_fill_(src_key_padding_mask==False, -1e9)

        p_attn = F.softmax(scores, dim = -1)
        value_attention = torch.matmul(p_attn, value_new)

        if return_score_flag:
            return value_attention, scores
        else:
            return value_attention

def max_pool(x, dim):
    """max pooling"""

    x_max = torch.max(x, dim=dim)[0].contiguous()
    
    return x_max 

def mean_pool(x, dim):
    """average pooling"""

    x_mean = torch.mean(x, dim=dim) 

    return x_mean 

class I2DFormer_logits(nn.Module):
    def __init__(self, feature_dim, pooling_method, temperature=np.log(1/0.07), fixed_temperature_flag=False):
        super().__init__()

        if fixed_temperature_flag:
            self.logit_scale = nn.Parameter(torch.ones([1]), requires_grad=False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([1]))
        nn.init.constant_(self.logit_scale, temperature)

        if pooling_method == "max":
            self.pool_func = max_pool 
        elif pooling_method == "mean":
            self.pool_func = mean_pool 

        self.local_linear = nn.Linear(feature_dim, 1)

    def forward(self, feature_image_global, feature_text_global, attention_values_local):

        feature_image_global = feature_image_global / (feature_image_global.norm(dim=-1, keepdim=True))
        feature_text_global = feature_text_global / (feature_text_global.norm(dim=-1, keepdim=True) + 1e-10)

        logits_per_image = self.logit_scale * feature_image_global @ feature_text_global.t()
        
        values_local_pool = self.pool_func(attention_values_local, dim=2)
        logits_per_image_local = self.local_linear(values_local_pool).squeeze(2)
        
        return logits_per_image, logits_per_image_local


class I2DFormer_logits_only_global(nn.Module):
    def __init__(self, feature_dim, pooling_method, temperature=np.log(1/0.07), fixed_temperature_flag=False):
        super().__init__()

        if fixed_temperature_flag:
            self.logit_scale = nn.Parameter(torch.ones([1]), requires_grad=False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([1]))
        nn.init.constant_(self.logit_scale, temperature)

        if pooling_method == "max":
            self.pool_func = max_pool 
        elif pooling_method == "mean":
            print("使用 mean !")
            self.pool_func = mean_pool 

        self.local_linear = nn.Linear(feature_dim, 1)

    def forward(self, feature_image_global, feature_text_global, attention_values_local):

        feature_image_global = feature_image_global / (feature_image_global.norm(dim=-1, keepdim=True))
        feature_text_global = feature_text_global / (feature_text_global.norm(dim=-1, keepdim=True) + 1e-10)

        logits_per_image = self.logit_scale * feature_image_global @ feature_text_global.t()
        logits_per_image_local = 0
        
        return logits_per_image, logits_per_image_local



class I2DFormer_logits_local(nn.Module):
    def __init__(self, feature_dim, pooling_method):
        super().__init__()

        if pooling_method == "max":
            self.pool_func = max_pool 
        elif pooling_method == "mean":
            print("使用 mean !")
            self.pool_func = mean_pool 

        self.local_linear = nn.Linear(feature_dim, 1)

    def forward(self, attention_values_local):

        values_local_pool = self.pool_func(attention_values_local, dim=2)
        logits_per_image_local = self.local_linear(values_local_pool).squeeze(2)
        
        return logits_per_image_local

def load_text_embedding(path_document_json, max_len, path_cache, d_text_glove_embedding=300):
    
    glove = vocab.GloVe(name='6B', dim=d_text_glove_embedding, cache=path_cache) 

    with open(path_document_json, 'r') as f:
        data = json.load(f)

    category_glove_embedding_dict = {}

    eos_embed = torch.zeros(d_text_glove_embedding)

    for category in data:
        document = data[category]

        doc = word_tokenize(document)

        len_text = 0 
        mask_matrix = [False]
        sentence_embed = 0 

        for temp_token in doc:
            if temp_token in glove.stoi:
                temp_vector =  glove.vectors[glove.stoi[temp_token]]
                mask_matrix.append(False)
            elif temp_token.lower() in glove.stoi:
                temp_vector =  glove.vectors[glove.stoi[temp_token.lower()]]
                mask_matrix.append(False)
            else:
                temp_vector = eos_embed
                mask_matrix.append(True)
            
            if len_text == 0:
                sentence_embed = temp_vector.unsqueeze(0)
            else:
                sentence_embed = torch.cat((sentence_embed, temp_vector.unsqueeze(0)), dim=0)

            len_text += 1
        
        
        mask_matrix = torch.tensor(mask_matrix)
        
        if len_text < max_len - 1:
            sentence_embed = torch.cat((sentence_embed, eos_embed.expand(max_len - 1 - len_text, eos_embed.shape[0])), dim=0)
            mask_matrix = torch.cat((mask_matrix, torch.tensor(True).expand(max_len - 1 - len_text)))
        elif len_text > max_len - 1:
            print(category)
            sys.exit()

        category_glove_embedding_dict[category] = {}
        category_glove_embedding_dict[category]["glove_embed"] = sentence_embed 
        category_glove_embedding_dict[category]["mask_matrix"] = mask_matrix 

    return category_glove_embedding_dict

class Attention_layer_true_i2mv(nn.Module):
    def __init__(self, d_model=64, heads=4):
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads
        self.scale = d_model ** -0.5

        self.W_Q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_K = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_V = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.fc_out = nn.Linear(self.d_model, self.d_model, bias=False)

        attn_std = self.d_model ** -0.5
        nn.init.normal_(self.W_Q.weight, std=attn_std)
        nn.init.normal_(self.W_K.weight, std=attn_std)
        nn.init.normal_(self.W_K.weight, std=attn_std)

    def forward(self, query, key, key_padding_mask=None, return_score_flag=False):

        query = rearrange(query, 'i q (h d) -> i h q d', h=self.heads, d=self.head_dim)
        key = rearrange(key, 't k (h d) -> t h k d', h=self.heads, d=self.head_dim)

        query_new = self.W_Q(query)
        key_new = self.W_K(key)
        value_new = self.W_V(key)

        scores =  einsum("i h q d, t h k d -> i t h q k", query_new, key_new) / math.sqrt(self.d_model)
    
        if key_padding_mask is not None:
            src_key_padding_mask = repeat(key_padding_mask, 't k -> i h q t k', i=scores.shape[0], h=scores.shape[2], q=scores.shape[3])
            src_key_padding_mask = rearrange(src_key_padding_mask, 'i h q t k -> i t h q k')
            scores.masked_fill_(src_key_padding_mask, torch.finfo(torch.float32).min)
            
        p_attn = F.softmax(scores, dim = -1)
        value_attention = einsum('i t h q k, t h k d -> i t h q d', p_attn, value_new)
        value_attention = rearrange(value_attention, 'i t h q d -> i t q (h d)', h=self.heads)

        value_attention = self.fc_out(value_attention)

        if return_score_flag:
            return value_attention, scores
        else:
            return value_attention


class CosineAnnealingLRWarmup(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=1.0e-6, last_epoch=-1, verbose=False,
                 warmup_steps=2, warmup_start_lr=0, warmup_flag="linear"):
        
        self.lr_scale_list = []
        for i, param_group in enumerate(optimizer.param_groups):
            self.lr_scale_list.append(param_group["lr_scale"])
        
        super(CosineAnnealingLRWarmup, self).__init__(optimizer,T_max=T_max,
                                                      eta_min=eta_min,
                                                      last_epoch=last_epoch)

        self.warmup_steps=warmup_steps
        self.warmup_start_lr = warmup_start_lr

        if warmup_steps>0:
            self.warmup_flag = warmup_flag

            if self.warmup_flag == "square":
                self.base_warup_factors = [
                    (base_lr/warmup_start_lr)**(1.0/self.warmup_steps)
                    for base_lr in self.base_lrs
                ]
            elif self.warmup_flag == "linear":
                self.base_warup_factors = [
                    np.linspace(warmup_start_lr, base_lr, warmup_steps)
                    for base_lr in self.base_lrs
                ]
 
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        return self._get_closed_form_lr()
 
    def _get_closed_form_lr(self):

        if hasattr(self,'warmup_steps'):
            if self.last_epoch<self.warmup_steps:
                if self.warmup_flag == "square":
                    return [self.warmup_start_lr*(warmup_factor**self.last_epoch)
                            for warmup_factor in self.base_warup_factors]
                elif self.warmup_flag == "linear":
                    return [self.base_warup_factors[i][self.last_epoch] * self.lr_scale_list[i]
                            for i in range(len(self.base_warup_factors))]
                    
            else:
                return [(self.eta_min + (self.base_lrs[i] - self.eta_min) *
                        (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps)))*0.5) * self.lr_scale_list[i]
                        for i in range(len(self.base_lrs))]


        else:
            return [(self.eta_min + (self.base_lrs[i] - self.eta_min) *
                    (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2) * self.lr_scale_list[i]
                    for i in range(len(self.base_lrs))]
 