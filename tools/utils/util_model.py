# ==================== Import ==================== #
import os 

import numpy as np 
import shutil
import random
import math 
import json 

import torch 
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from pprint import pprint

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

font = FontProperties(fname='SimHei.ttf', size=16)  

import warnings

# ==================== Function ==================== #
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True 

def calibrated_stacking(test_seen_label, output, lam=1e-3):
    """
    output: the output predicted scores
    lam: the parameter to control the output score of seen classes.
    self.test_seen_label
    self.test_unseen_label
    :return
    """
    output = output.cpu().numpy()
    seen_L = list(set(test_seen_label.numpy()))
    output[:, seen_L] = output[:, seen_L] - lam
    return torch.from_numpy(output)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res 

def save_checkpoint(state, is_best, filename='', model_name=''):

    torch.save(state, os.path.join(filename, model_name + 'latest.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filename, model_name + 'latest.pth.tar'), os.path.join(filename, model_name + 'best.pth.tar'))


def draw_acc(checkpoint_path, prec, iter, title):

    path_json = os.path.join(checkpoint_path, title + ".json")

    if not os.path.exists(path_json):
        data_dict = {}
    else:
        file = open(path_json, 'r')
        data_dict = json.load(file)

    data_dict[iter] = prec

    data_json = json.dumps(data_dict, indent=4) 
    file = open(path_json, 'w') 
    file.write(data_json)
    file.close()

    val_acc_list = []
    for key in data_dict:
        val_acc_list.append(data_dict[key])

    acc_max, acc_index = max(val_acc_list), val_acc_list.index(max(val_acc_list))
    
    plt.figure()
    plt.plot(val_acc_list, label="val acc")
    plt.annotate("iter " + str(acc_index) + ": " + str(round(acc_max, 2)), xy=(acc_index, acc_max), xytext=(acc_index, acc_max))
    plt.plot(acc_index, acc_max, 'ro')
    plt.xlabel("iter")
    plt.ylabel("acc")
    str_title = title
    plt.title(str_title)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_path, title + ".jpg"))

def draw_acc_list(checkpoint_path, prec_list, iter, title_list, save_name="acc_info", rewrite_flag=0):
    
    path_json = os.path.join(checkpoint_path, save_name + ".json")

    if rewrite_flag and os.path.exists(path_json):
        os.remove(path_json)

    if not os.path.exists(path_json):
        data_dict = {}
    else:
        file = open(path_json, 'r')
        data_dict = json.load(file)

    index = 0
    data_dict[iter] = {}
    for key in title_list:
        data_dict[iter][key] = prec_list[index]
        index += 1

    data_json = json.dumps(data_dict, indent=4) 
    file = open(path_json, 'w') 
    file.write(data_json)
    file.close()

    val_acc_list = []
    acc_max_list = []
    acc_index_list = []

    for key in title_list:
        val_acc_list.append([])

    for iter in data_dict:
        index = 0
        for key in title_list:
            val_acc_list[index].append(data_dict[iter][key])

            index += 1

    for index in range(len(val_acc_list)):
        acc_max_list.append(max(val_acc_list[index]))
        acc_index_list.append(val_acc_list[index].index(max(val_acc_list[index])))
    
    plt.figure()

    index = 0 
    for key in title_list:

        plt.plot(val_acc_list[index], label=key)
        plt.annotate("it " + str(acc_index_list[index]) + ": " + str(round(acc_max_list[index], 2)), xy=(acc_index_list[index], acc_max_list[index]), xytext=(acc_index_list[index], acc_max_list[index]))
        plt.plot(acc_index_list[index], acc_max_list[index], 'ro')

        index += 1

    plt.xlabel("iter")
    plt.ylabel("acc")
    str_title = save_name
    plt.title(str_title)
    plt.legend()
    plt.savefig(os.path.join(checkpoint_path, save_name + ".jpg"))



def get_MAE_augment(args):
    
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )

    return transform


