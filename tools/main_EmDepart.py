# ==================== Import ==================== #
import time
import sys
import os 
import shutil 

sys.path.append("tools")

import numpy as np 

from pprint import pprint 

import argparse 

import torch 
import torch.backends.cudnn as cudnn 
from torch.optim.lr_scheduler import CosineAnnealingLR 

from configs.option import parse_opt 
from utils.util_model import setup_seed
from utils.util import build_save_file
from models.EmDepart import EmDepart_model
from data_process.named_data import get_named_data 

from models.model_util import CosineAnnealingLRWarmup

from process.train import train

import warnings
warnings.filterwarnings('ignore')



# ==================== Functions ==================== #

def param_groups(model, args):
    
    # Frozen Image Encoder # 
    for name, param in model.named_parameters():
        if "feature_extractor" in name:
            param.requires_grad = False

    param_group_names = {}
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim < 2 or 'bias' in name or 'ln' in name or 'bn' in name or "positional_embedding" in name: 
            g_decay = "no_decay"
            this_decay = 0 
        else:
            g_decay = "decay"
            this_decay = args.weight_decay 
        
        lr = args.learning_rate 

        group_name = "%s" % (g_decay) 

        if group_name not in param_group_names:
            param_group_names[group_name] = {
                "weight_decay": this_decay,
                "params": [],
                'lr': lr,
                'lr_scale': 1
            }

            param_groups[group_name] = {
                "weight_decay": this_decay,
                "params": [],
                'lr': lr,
                'lr_scale': 1
            }

        param_group_names[group_name]["params"].append(name)
        param_groups[group_name]["params"].append(param)

    params = list(param_groups.values())

    return params

# ==================== main ==================== #
if __name__ == '__main__':

    # ----- Start ----- #
    T_Start = time.time()
    print("The programme is running ! \n")
    print("Python environment: ", sys.executable)
    print("")

    # ---------- step0 ---------- # 
    args = parse_opt() 

    if args.seed == "0":
        args.seed = np.random.randint(10000)
    print("seed:", args.seed)
    setup_seed(args.seed)

    build_save_file(args)

    # pprint(vars(args))
    # print('')

    # ---------- step1: Build Dataset ---------- #  
    print("\nLoad Dataset =>")
    if args.gzsl_flag:
        train_loader, test_seen_loader, test_unseen_loader, train_classes_name_to_index, test_classes_name_to_index = get_named_data(args, train_mode=args.train_mode)
    else: 
        train_loader, test_unseen_loader, train_val_loader, train_classes_name_to_index, test_classes_name_to_index = get_named_data(args, train_mode=args.train_mode)

    # ---------- step2: Load Model ---------- #  
    print("\nLoad Model =>")

    model = EmDepart_model(args, train_classes_name_to_index, test_classes_name_to_index).cuda()
    # print(model) 

    # ----- load checkpoint ----- #  
    if args.load_checkpoint_path:
        checkpoint = torch.load(args.load_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])

        if "best_unseen" in checkpoint:
            if args.gzsl_flag and "best_H" in checkpoint:
                print("\nbest H {:.4f}, Seen acc {:.4f}, Unseen acc {:.4f}".format(checkpoint["best_H"], checkpoint["best_seen"], checkpoint["best_unseen"]))
            else:
                print("\nbest Unseen acc {:.4f}".format(checkpoint["best_unseen"]))

    cudnn.benchmark = True

    # ---------- step3: Optimization ---------- # 
    params = param_groups(model, args)

    if args.optim.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, eps=args.adam_epsilon) #  betas=(0.9, 0.999)
    elif args.optim.lower() == "adam":
        optimizer = torch.optim.Adam(params, eps=args.adam_epsilon) # , betas=(0.5, 0.999)

    t_total = (len(train_loader) // args.gradient_accumulation_steps) * float(args.epochs)
    print("steps: ", t_total)

    if args.warmup_epochs > 0:
        scheduler = CosineAnnealingLRWarmup(optimizer, T_max=t_total, warmup_start_lr=args.warmup_start_lr, warmup_steps=args.warmup_epochs*len(train_loader), eta_min=args.eta_min)
    else:
        scheduler = CosineAnnealingLR(optimizer, t_total)

    # ---------- step4: Train ---------- # 
    if args.gzsl_flag:
        train(args, train_loader, test_unseen_loader, model, optimizer, scheduler, test_seen_loader=test_seen_loader, train_mode=args.train_mode)
    else: 
        train(args, train_loader, test_unseen_loader, model, optimizer, scheduler, train_val_loader=train_val_loader, train_mode=args.train_mode)

    # ----- Finish ----- #
    T_End = time.time()
    T_Sum = T_End  - T_Start
    T_Hour = int(T_Sum/3600)
    T_Minute = int((T_Sum%3600)/60)
    T_Second = round((T_Sum%3600)%60, 2)
    print('Program running time: {} hours {} minutes {} seconds'.format(T_Hour, T_Minute, T_Second))
    print('The programme has finished !')