# ==================== Import ==================== #
import time
import sys
import os 

import numpy as np 

import torch
from torch import dropout, nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat 
from einops_exts import rearrange_many, repeat_many

from models.SDM.pos_encoding import build_position_encoding
from models.SDM.attention import * 
from models.norm_loss_util import variance_loss

# ==================== Functions ==================== #

def FeedForward_flamingo(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

def FeedForward_flamingo_dropout(dim, mult = 4, dropout=0.1):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

def FeedForward_flamingo_dropout_more(dim, mult = 4, dropout=0.1):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False),
        nn.Dropout(dropout)
    )

class PerceiverAttention_modified(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        xavier_init = False,
        variance_constant = 1,
        variance_after_softmax_flag=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.variance_constant = variance_constant
        self.variance_after_softmax_flag = variance_after_softmax_flag

        if xavier_init:
            self._reset_parameter()

    def _reset_parameter(self):
        nn.init.xavier_uniform_(self.to_q.weight)
        nn.init.xavier_uniform_(self.to_kv.weight)

    def forward(self, x, latents, mask=None, softmax_mode='default', key_mode='default'):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """

        b = x.shape[0]
        h = self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        if key_mode == "concat":
            kv_input = torch.cat((x, latents), dim = -2)
        elif key_mode == "default":
            kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        # attention
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        
        if exists(mask):
            mask_std = mask.detach().clone()
            mask_attn = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask_attn = repeat(mask_attn, 'b j -> b h () j', h = h)
            sim.masked_fill_(mask_attn, max_neg_value)

            mask_std = repeat(mask_std, 'b j -> b h j', h = h)

        sim_new = sim - sim.amax(dim = -1, keepdim = True).detach()

        if not self.variance_after_softmax_flag:
            std_parameter = torch.sqrt(sim_new.var(dim=-2) + 0.0001)

            if exists(mask):
                std_parameter = std_parameter.masked_fill(mask_std, self.variance_constant)
            
            loss_variance = torch.mean(F.relu(self.variance_constant - std_parameter))
            
        if softmax_mode == "default":
            attn = sim_new.softmax(dim = -1)
        elif softmax_mode == "slot":
            attn = sim_new.softmax(dim = -2)
            attn = attn / (attn.sum(dim = -1, keepdim = True) + 1e-7)
        
        if torch.isnan(attn).any():
            import pdb; pdb.set_trace()
        
        if self.variance_after_softmax_flag:
            std_parameter = torch.sqrt(attn.var(dim=-2) + 0.0001)

            if exists(mask):
                std_parameter = std_parameter.masked_fill(mask_std, self.variance_constant)
            
            loss_variance = torch.mean(F.relu(self.variance_constant - std_parameter))
            
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)

        out = self.to_out(out)

        return out, loss_variance
    
class SDM(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        ff_mult = 4,
        variance_constant=1, 
        dropout=0, 
        more_drop_flag=False,
        variance_after_softmax_flag=False
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        self.layers = nn.ModuleList([])
        if dropout:
            
            if more_drop_flag:
                for _ in range(depth):
                    self.layers.append(nn.ModuleList([
                        PerceiverAttention_modified(dim = dim, dim_head = dim_head, heads = heads, variance_constant=variance_constant, variance_after_softmax_flag=variance_after_softmax_flag),
                        nn.LayerNorm(dim), 
                        FeedForward_flamingo_dropout_more(dim = dim, mult = ff_mult, dropout=dropout),
                        nn.LayerNorm(dim)
                    ]))
            else:
                for _ in range(depth):
                    self.layers.append(nn.ModuleList([
                        PerceiverAttention_modified(dim = dim, dim_head = dim_head, heads = heads, variance_constant=variance_constant, variance_after_softmax_flag=variance_after_softmax_flag),
                        nn.LayerNorm(dim), 
                        FeedForward_flamingo_dropout(dim = dim, mult = ff_mult, dropout=dropout),
                        nn.LayerNorm(dim)
                    ]))

        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PerceiverAttention_modified(dim = dim, dim_head = dim_head, heads = heads, variance_constant=variance_constant, variance_after_softmax_flag=variance_after_softmax_flag),
                    nn.LayerNorm(dim), 
                    FeedForward_flamingo(dim = dim, mult = ff_mult),
                    nn.LayerNorm(dim)
                ]))

        self.depth = depth

    def forward(self, x, mask=None, softmax_mode='default', key_mode='default', variance_flag=False):

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        loss_variance = 0 
        
        for attn, ln1, ff, ln2 in self.layers:
            
            # Attention # 
            latents_attn, loss_variance_layer = attn(x, latents, mask=mask, softmax_mode=softmax_mode, key_mode=key_mode) 
            latents = latents_attn + latents
            latents = ln1(latents)
            
            # Feed Forward # 
            latents = ff(latents) + latents
            latents = ln2(latents)

            loss_variance += loss_variance_layer 

        loss_variance /= self.depth
            
        if variance_flag:
            return latents, loss_variance
        else:
            return latents