# ==================== Import ==================== #
import time
import sys
import os 

import numpy as np 

import torch  

from einops import rearrange, repeat
import torch.nn.functional as F

torch.set_printoptions(threshold=50)

# ==================== Functions ==================== #
def mmd_rbf_loss(x, y, gamma=None, reduction='mean'):
    if gamma is None:
        gamma = 1./x.size(-1)
    if reduction=='mean':
        loss = rbf_memory_efficient(x, x, gamma).mean() - 2 * rbf_memory_efficient(x, y, gamma).mean() + rbf_memory_efficient(y, y, gamma).mean()
    else:
        loss = rbf_memory_efficient(x, x, gamma).sum() - 2 * rbf_memory_efficient(x, y, gamma).sum() + rbf_memory_efficient(y, y, gamma).sum()
    return loss

def rbf_memory_efficient(x, y, gamma=0.5):
    """RBF kernel that does not cause memory shortage"""
    cdist = torch.cdist(x, y)
    return torch.exp(-gamma * cdist)

def variance_loss(x, dim, variance_constant=1):
    
    std_parameter = torch.sqrt(x.var(dim=dim) + 0.0001) 
    loss_variance = torch.mean(F.relu(variance_constant - std_parameter))

    print("\nstd_parameter: \n", std_parameter.shape)
    
    return loss_variance

def diversity_loss(x, num_embeds, reduction="mean"):
    
    if num_embeds == 1:
        return 0.0
    
    x = x / x.norm(dim=-1, keepdim=True)

    gram_x = x.bmm(x.transpose(1,2))

    I = torch.autograd.Variable((torch.eye(x.size(1)) > 0.5).repeat(gram_x.size(0), 1, 1))
    if torch.cuda.is_available():
        I = I.cuda()

    gram_x.masked_fill_(I, 0.0)

    loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (num_embeds**2)
    
    if reduction == "mean":
        return loss.mean()
    else:
        return loss.sum()
 