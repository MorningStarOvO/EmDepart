# ==================== Import ==================== #
import time
import sys
import os 

import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


# ==================== Functions ==================== #
def cosine_sim(x, y):
    """Cosine similarity between all the image and sentence pairs. Assumes that x and y are l2 normalized"""
    return x.mm(y.t())

class MPdistance(nn.Module):
    def __init__(self, avg_pool):
        super(MPdistance, self).__init__()
        self.avg_pool = avg_pool
        self.alpha, self.beta = nn.Parameter(torch.ones(1), requires_grad=False), nn.Parameter(torch.zeros(1), requires_grad=False)
        
        
    def forward(self, img_embs, txt_embs):
        dist = cosine_sim(img_embs, txt_embs)
        avg_distance = self.avg_pool(torch.sigmoid(self.alpha * dist.unsqueeze(0) + self.beta)).squeeze(0)
        return avg_distance

class SetwiseDistance_origin(nn.Module):
    def __init__(self, img_set_size, txt_set_size, denominator, temperature=1, temperature_txt_scale=1):
        super(SetwiseDistance, self).__init__()
        # poolings
        self.img_set_size = img_set_size
        self.txt_set_size = txt_set_size
        self.denominator = denominator
        self.temperature = temperature
        self.temperature_txt_scale = temperature_txt_scale # used when computing i2t distance
        
        self.xy_max_pool = torch.nn.MaxPool2d((self.img_set_size, self.txt_set_size))
        self.xy_avg_pool = torch.nn.AvgPool2d((self.img_set_size, self.txt_set_size))
        self.x_axis_max_pool = torch.nn.MaxPool2d((1, self.txt_set_size))
        self.x_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(1, self.txt_set_size))
        self.y_axis_max_pool = torch.nn.MaxPool2d((self.img_set_size, 1))
        self.y_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(self.img_set_size, 1))
        
        self.mp_dist = MPdistance(self.xy_avg_pool)
        
    def smooth_chamfer_distance_euclidean(self, img_embs, txt_embs):
        """
            Method to compute Smooth Chafer Distance(SCD). Max pool is changed to LSE.
            Use euclidean distance(L2-distance) to measure distance between elements.
        """
        dist = torch.cdist(img_embs, txt_embs)
        
        right_term = -self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(-self.temperature * self.temperature_txt_scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = -self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(-self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.temperature * self.temperature_txt_scale) + left_term / (self.txt_set_size * self.temperature)) / (self.denominator)

        return smooth_chamfer_dist
    
    def smooth_chamfer_distance_cosine(self, img_embs, txt_embs):
        """
            cosine version of smooth_chamfer_distance_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        
        right_term = self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(self.temperature * self.temperature_txt_scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(self.temperature * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.temperature * self.temperature_txt_scale) + left_term / (self.txt_set_size * self.temperature)) / (self.denominator)

        return smooth_chamfer_dist
    
    def chamfer_distance_cosine(self, img_embs, txt_embs):
        """
            cosine version of chamfer_distance_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        
        right_term = self.y_axis_sum_pool(self.x_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        left_term = self.x_axis_sum_pool(self.y_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        
        chamfer_dist = (right_term / self.img_set_size + left_term / self.txt_set_size) / self.denominator

        return chamfer_dist
    
    def max_distance_cosine(self, img_embs, txt_embs):
        dist = cosine_sim(img_embs, txt_embs)
        max_distance = self.xy_max_pool(dist.unsqueeze(0)).squeeze(0)
        return max_distance

    def smooth_chamfer_distance(self, img_embs, txt_embs):
        return self.smooth_chamfer_distance_cosine(img_embs, txt_embs)
    
    def chamfer_distance(self, img_embs, txt_embs):
        return self.chamfer_distance_cosine(img_embs, txt_embs)
    
    def max_distance(self, img_embs, txt_embs):
        return self.max_distance_cosine(img_embs, txt_embs)
    
    def avg_distance(self, img_embs, txt_embs):
        return self.mp_dist(img_embs, txt_embs)
    
class SetwiseDistance(nn.Module):
    """Contrastive Loss 的 Set wise Distance"""
    def __init__(self, img_set_size, txt_set_size, denominator=2, temperature=np.log(1/0.07), scale=1, fixed_temperature_flag=False):
        super(SetwiseDistance, self).__init__()
        # poolings
        self.img_set_size = img_set_size
        self.txt_set_size = txt_set_size
        self.denominator = denominator # it acts as a average between two sets when = 2

        # ----- 温度系数的计算 ----- #  
        if fixed_temperature_flag:
            self.temperature = nn.Parameter(torch.ones([1]), requires_grad=False)
        else:
            self.temperature = nn.Parameter(torch.ones([1]))
        
        nn.init.constant_(self.temperature, temperature)

        self.scale = scale

        self.xy_max_pool = torch.nn.MaxPool2d((self.img_set_size, self.txt_set_size))
        self.xy_avg_pool = torch.nn.AvgPool2d((self.img_set_size, self.txt_set_size))
        self.x_axis_max_pool = torch.nn.MaxPool2d((1, self.txt_set_size))
        self.x_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(1, self.txt_set_size))
        self.y_axis_max_pool = torch.nn.MaxPool2d((self.img_set_size, 1))
        self.y_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(self.img_set_size, 1))
        
        self.mp_dist = MPdistance(self.xy_avg_pool)
        
    def smooth_chamfer_distance_euclidean(self, img_embs, txt_embs):
        """
            Method to compute Smooth Chafer Distance(SCD). Max pool is changed to LSE.
            Use euclidean distance(L2-distance) to measure distance between elements.
        """
        dist = torch.cdist(img_embs, txt_embs)
        
        right_term = -self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(-self.scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = -self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(-self.scale * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.scale ) + left_term / (self.txt_set_size * self.scale)) / (self.denominator)

        return smooth_chamfer_dist
    
    def smooth_chamfer_distance_cosine(self, img_embs, txt_embs):
        """
            cosine version of smooth_chamfer_distance_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        
        right_term = self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(self.scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(self.scale * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.scale) + left_term / (self.txt_set_size * self.scale)) / (self.denominator)

        return smooth_chamfer_dist
    
    def chamfer_distance_cosine(self, img_embs, txt_embs):
        """
            cosine version of chamfer_distance_euclidean(img_embs, txt_embs)
        """
        dist = cosine_sim(img_embs, txt_embs)
        
        right_term = self.y_axis_sum_pool(self.x_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        left_term = self.x_axis_sum_pool(self.y_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        
        chamfer_dist = (right_term / self.img_set_size + left_term / self.txt_set_size) / self.denominator

        return chamfer_dist
    
    def max_distance_cosine(self, img_embs, txt_embs):
        dist = cosine_sim(img_embs, txt_embs)
        max_distance = self.xy_max_pool(dist.unsqueeze(0)).squeeze(0)
        return max_distance

    def smooth_chamfer_distance(self, img_embs, txt_embs):
        return self.temperature * self.smooth_chamfer_distance_cosine(img_embs, txt_embs)
    
    def chamfer_distance(self, img_embs, txt_embs):
        return self.temperature * self.chamfer_distance_cosine(img_embs, txt_embs)
    
    def max_distance(self, img_embs, txt_embs):
        return self.temperature * self.max_distance_cosine(img_embs, txt_embs)
    
    def avg_distance(self, img_embs, txt_embs):
        return self.temperature * self.mp_dist(img_embs, txt_embs)


class MPdistance_any_score_func(nn.Module):
    def __init__(self, avg_pool):
        super(MPdistance_any_score_func, self).__init__()
        self.avg_pool = avg_pool
        self.alpha, self.beta = nn.Parameter(torch.ones(1),requires_grad=False), nn.Parameter(torch.zeros(1), requires_grad=False)    
        
    def forward(self, dist):
        avg_distance = self.avg_pool(torch.sigmoid(self.alpha * dist.unsqueeze(0) + self.beta)).squeeze(0)
        return avg_distance

class SetwiseDistance_any_score_func(nn.Module):
    """Contrastive Loss 的 Set wise Distance"""
    def __init__(self, img_set_size, txt_set_size, denominator=2, temperature=np.log(1/0.07), scale=1, fixed_temperature_flag=False):
        super(SetwiseDistance_any_score_func, self).__init__()
        # poolings
        self.img_set_size = img_set_size
        self.txt_set_size = txt_set_size
        self.denominator = denominator # it acts as a average between two sets when = 2

        if fixed_temperature_flag:
            self.temperature = nn.Parameter(torch.ones([1]), requires_grad=False)
        else:
            self.temperature = nn.Parameter(torch.ones([1]))
        
        nn.init.constant_(self.temperature, temperature)

        self.scale = scale

        self.xy_max_pool = torch.nn.MaxPool2d((self.img_set_size, self.txt_set_size))
        self.xy_avg_pool = torch.nn.AvgPool2d((self.img_set_size, self.txt_set_size))
        self.x_axis_max_pool = torch.nn.MaxPool2d((1, self.txt_set_size))
        self.x_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(1, self.txt_set_size))
        self.y_axis_max_pool = torch.nn.MaxPool2d((self.img_set_size, 1))
        self.y_axis_sum_pool = torch.nn.LPPool2d(norm_type=1, kernel_size=(self.img_set_size, 1))

        self.mp_dist = MPdistance_any_score_func(self.xy_avg_pool)
    
    def smooth_chamfer_distance(self, dist):

        right_term = self.y_axis_sum_pool(
            torch.log(self.x_axis_sum_pool(
                torch.exp(self.scale * dist.unsqueeze(0))
            ))).squeeze(0)
        left_term = self.x_axis_sum_pool(
            torch.log(self.y_axis_sum_pool(
                torch.exp(self.scale * dist.unsqueeze(0))
            ))).squeeze(0)
        
        smooth_chamfer_dist = (right_term / (self.img_set_size * self.scale) + left_term / (self.txt_set_size * self.scale)) / (self.denominator)

        return self.temperature * smooth_chamfer_dist
    
    def chamfer_distance(self, dist):

        right_term = self.y_axis_sum_pool(self.x_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        left_term = self.x_axis_sum_pool(self.y_axis_max_pool(dist.unsqueeze(0))).squeeze(0)
        
        chamfer_dist = (right_term / self.img_set_size + left_term / self.txt_set_size) / self.denominator

        return self.temperature * chamfer_dist
    
    def max_distance(self, dist):

        max_distance = self.xy_max_pool(dist.unsqueeze(0)).squeeze(0)
        return self.temperature * max_distance
    
    def avg_distance(self, dist):
        return self.temperature * self.mp_dist(dist)
