# ==================== Import ==================== #
import time
import sys
import os 

import numpy as np 
import json 

import torch  

from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

from utils.util_model import AverageMeter, accuracy
from utils.util_path import save_json, sorted_dict

# ==================== Functions ==================== #
def compute_H(acc_seen, acc_unseen):
    if (acc_seen + acc_unseen) == 0:
        H = 0
    else:
        H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen) 

    return H 

def test_cs_list(args, test_unseen_loader, test_seen_loader, model):
    
    best_cs_H = 0
    best_cs_seen = 0 
    best_cs_unseen = 0 
    best_cs_value = 0
    best_cs_only_unseen = 0

    print("list for calibrated stacking ...")

    cs_value_original = args.calibrated_stacking

    args.calibrated_stacking = 0 

    logits_array_unseen, label_array_unseen = test_get_logits(args, test_unseen_loader, model, mode="unseen_test")
    logits_array_seen, label_array_seen = test_get_logits(args, test_seen_loader, model, mode="seen_test")

    for cs_value in args.cs_list:

        logits_array_seen_temp = model.calibrated_stacking_func(logits_array_seen.clone().detach(), cs_value)
        acc_seen_temp = test_use_logits(args, logits_array_seen_temp, label_array_seen, mode="seen_test")

        logits_array_unseen_temp = model.calibrated_stacking_func(logits_array_unseen.clone().detach(), cs_value)
        acc_unseen_temp = test_use_logits(args, logits_array_unseen_temp, label_array_unseen, mode="unseen_test")

        acc_H_temp = compute_H(acc_seen_temp, acc_unseen_temp)

        if acc_H_temp > best_cs_H:
            best_cs_H = acc_H_temp
            best_cs_seen = acc_seen_temp 
            best_cs_unseen = acc_unseen_temp
            best_cs_value = cs_value
    
        best_cs_only_unseen = acc_unseen_temp

    args.calibrated_stacking = cs_value_original

    print(f"ZSL: {round(best_cs_only_unseen, 4)}")

    return best_cs_H, best_cs_seen, best_cs_unseen, best_cs_only_unseen

    
def test_get_logits(args, test_loader, model, mode="unseen_test"):
    
    model.eval()

    label_array = 0
    logits_array = 0 

    with torch.no_grad():
        for i, (img, label) in enumerate(tqdm(test_loader)):
            
            img, label = img.cuda(), label.cuda() 

            outputs = model.compute_loss(img, label, mode=mode, cs_value=args.calibrated_stacking)

            loss = outputs["loss"]
            output = outputs["logits_per_image"]

            if i == 0:
                logits_array = output.clone().detach()
                label_array = label.clone().detach()
            else:
                logits_array = torch.cat((logits_array, output.clone().detach()))
                label_array = torch.cat((label_array, label.clone().detach()))
    
    return logits_array, label_array

def test_use_logits(args, logits_array, label_array, mode="unseen_test"):
    
    if "unseen" in mode:
        if args.gzsl_flag:
            with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_unseen_gzsl.json"), 'r') as f:
                data = json.load(f)
        else:
            with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_unseen_zsl.json"), 'r') as f:
                data = json.load(f)
    else:
        if args.gzsl_flag:
            with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_seen_gzsl.json"), 'r') as f:
                data = json.load(f)
        else:
            with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_seen_zsl.json"), 'r') as f:
                data = json.load(f)

    label_test_list = []
    for key in data:
        label_test_list.append(int(key))

    if args.gzsl_flag:
        label_all_list = []
        data_all = {}

        with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_seen_gzsl.json"), 'r') as f:
            data_seen = json.load(f)

        for key in data_seen:
            label_all_list.append(int(key))
            data_all[key] = f"s_{data_seen[key]}"
        
        with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_unseen_gzsl.json"), 'r') as f:
            data_unseen = json.load(f)

        for key in data_unseen:
            label_all_list.append(int(key))
            data_all[key] = f"u_{data_unseen[key]}"
    else:
        label_all_list = label_test_list
        data_all = data
    
    _, pred_array = logits_array.topk(1, 1, True, True)
    pred_array = pred_array.squeeze()

    acc_per_class = torch.zeros(len(label_test_list))
    for i in label_test_list:
        idx = (label_array == i)
        
        if "unseen" in mode and args.gzsl_flag:
            index_new = i - args.seen_classes_num
        else:
            index_new = i

        acc_per_class[index_new] = torch.sum(label_array[idx] == pred_array[idx]).item() / torch.sum(idx).item()
                
    average_per_class_top_1_acc = acc_per_class.mean() * 100

    return average_per_class_top_1_acc.item()
    

def test(args, test_loader, model, mode="unseen_test", return_logits=False):
    
    if "unseen" in mode:
        if args.gzsl_flag:
            with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_unseen_gzsl.json"), 'r') as f:
                data = json.load(f)
        else:
            with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_unseen_zsl.json"), 'r') as f:
                data = json.load(f)
    else:
        if args.gzsl_flag:
            with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_seen_gzsl.json"), 'r') as f:
                data = json.load(f)
        else:
            with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_seen_zsl.json"), 'r') as f:
                data = json.load(f)

    label_test_list = []
    for key in data:
        label_test_list.append(int(key))

    if args.gzsl_flag:
        label_all_list = []
        data_all = {}

        with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_seen_gzsl.json"), 'r') as f:
            data_seen = json.load(f)

        for key in data_seen:
            label_all_list.append(int(key))
            data_all[key] = f"s_{data_seen[key]}"
        
        with open(os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_unseen_gzsl.json"), 'r') as f:
            data_unseen = json.load(f)

        for key in data_unseen:
            label_all_list.append(int(key))
            data_all[key] = f"u_{data_unseen[key]}"
    else:
        label_all_list = label_test_list
        data_all = data
            
    top1 = AverageMeter()
    loss_test = AverageMeter()

    model.eval()

    pred_array = 0
    label_array = 0
    logits_array = 0

    if args.only_test_flag:
        T_Start = time.time()
    with torch.no_grad():
        for i, (img, label) in enumerate(tqdm(test_loader)):
            
            img, label = img.cuda(), label.cuda() 

            outputs = model.compute_loss(img, label, mode=mode, cs_value=args.calibrated_stacking)

            loss = outputs["loss"]
            output = outputs["logits_per_image"]

            prec1 = accuracy(output.data, label, topk=(1,))[0]
            top1.update(prec1.item(), img.size(0))
            loss_test.update(loss.item(), img.size(0))

            _, pred = output.topk(1, 1, True, True)
            if i == 0:
                pred_array = pred.clone().detach()
                label_array = label.clone().detach()
                logits_array = output.clone().detach()
            else:
                pred_array = torch.cat((pred_array, pred.clone().detach()))
                label_array = torch.cat((label_array, label.clone().detach()))
                logits_array = torch.cat((logits_array, output.clone().detach()))
    
    if args.only_test_flag:
        T_End = time.time()

        T_Sum = (T_End  - T_Start) / 1155 * 1000
        T_Second = round((T_Sum%3600)%60, 2)
        
    pred_array = pred_array.squeeze()

    acc_per_class = torch.zeros(len(label_test_list))
    acc_dict = {} 
    acc_error_dict = {} 
    for i in label_test_list:
        idx = (label_array == i)
        
        if "unseen" in mode and args.gzsl_flag:
            index_new = i - args.seen_classes_num
        else:
            index_new = i
        
        acc_per_class[index_new] = torch.sum(label_array[idx] == pred_array[idx]).item() / torch.sum(idx).item()
        acc_dict[data[str(i)]] = round(acc_per_class[index_new].item() * 100, 2)

        acc_error_dict[data[str(i)]] = {}
        acc_error_dict[data[str(i)]]["T-" + data[str(i)]] = round(acc_per_class[index_new].item() * 100, 2)
        error_num = 0
        for j in label_all_list:
            if j != i:
                error_acc = torch.sum(pred_array[idx] == j).item() / torch.sum(idx).item()
                
                if error_acc * 100 > 1:
                    acc_error_dict[data[str(i)]][data_all[str(j)]] = round(error_acc * 100, 2)
                    error_num += 1
        
               
    average_per_class_top_1_acc = acc_per_class.mean() * 100

    print('* avg acc {avg_acc:.4f}, loss {loss.avg:.4f}'.format(avg_acc=average_per_class_top_1_acc, loss=loss_test))

    if return_logits:
        return average_per_class_top_1_acc.item(), top1.avg, loss_test.avg, acc_dict, acc_error_dict, logits_array, label_array
    else:
        return average_per_class_top_1_acc.item(), top1.avg, loss_test.avg, acc_dict, acc_error_dict
