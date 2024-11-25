# ==================== Import ==================== #
import time
import sys
import os 

import numpy as np 
import json 

from torch.utils.tensorboard import SummaryWriter

import torch  
import timm 

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

font = FontProperties(fname='SimHei.ttf', size=16)   

from utils.util_model import AverageMeter, accuracy, draw_acc_list
from process.test import test, test_get_logits, test_use_logits
from utils.util_path import save_json

# ====================  ==================== #
best_best_cs_H = 0
best_best_cs_value = 0
best_best_cs_seen = 0
best_best_cs_unseen = 0
best_best_zsl_unseen = 0
best_best_epoch_gzsl = 0 
best_best_epoch_zsl = 0

# ==================== Functions ==================== #
def compute_H(acc_seen, acc_unseen):
    if (acc_seen + acc_unseen) == 0:
        H = 0
    else:
        H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen) 

    return H 


def train(args, train_loader, test_unseen_loader,  model, optimizer, scheduler, test_seen_loader=None, train_val_loader=None, train_mode="default"):
    
    step = 0 

    top1 = AverageMeter()

    best_unseen = 0 
    best_acc = 0
    best_top1_unseen = 0
    best_H = 0 
    best_seen = 0 

    if not args.debug:
        number = len(os.listdir(args.tensorboard_dir))
        writer = SummaryWriter(comment=str(number), log_dir=args.tensorboard_dir) # comment='', 
    else:
        writer = None

    if args.text_logit_loss_flag and args.image_enhanced_model_flag:
        use_text_logit_flag = True 
    else:
        use_text_logit_flag = False

    for epoch in range(args.epochs):

        args.epoch_now = epoch

        model.train()

        print('Train Epoch: {:} / {:}'.format(epoch, args.epochs))

        top1.reset()

        for i, data in enumerate(tqdm(train_loader)):

            if train_mode == "default":
                
                if use_text_logit_flag:
                    img, label, text_logit_label = data[0].cuda(), data[1].cuda(), data[2].cuda() 
                    outputs = model.compute_loss(img, label, text_logits_label=text_logit_label) 
                else:
                    img, label = data[0].cuda(), data[1].cuda() 
                    outputs = model.compute_loss(img, label) 
            elif "image_definition" in train_mode:
                img, label = data[0].cuda(), data[1].cuda() 
                image_definition_embedding, image_definition_mask = data[2].cuda(), data[3].cuda()

                outputs = model.compute_loss(img, label, image_definition_embedding=image_definition_embedding, image_definition_mask=image_definition_mask) 

                label = torch.range(0, img.shape[0]-1).long().cuda()

            loss = outputs["loss"]
            image_logits = outputs["logits_per_image"]

            output = image_logits.softmax(dim=-1) 
            prec1 = accuracy(output.data, label, topk=(1,))[0]
            top1.update(prec1.item(), img.size(0))

            if not args.constant_lr_flag and step % args.gradient_accumulation_steps == 0:
                scheduler.step()
            
            optimizer.zero_grad()
            loss.backward() 

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            step += 1 

             
        if epoch % args.print_freq == 0:
            for key in outputs:
                if "loss" in key:
                    try:
                        print("{}:{:.3f}".format(key, outputs[key].item()))
                    except:
                        continue

                    if not args.debug:
                        writer.add_scalar(key, outputs[key].item(), step) # /args.print_freq)

            if not args.constant_lr_flag:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = args.learning_rate

            if not args.debug:
                writer.add_scalar("lr", current_lr, step) # /args.print_freq) 
                if hasattr(model, "I2DFormer_logits"):
                    writer.add_scalar("temperature logits", model.I2DFormer_logits.logit_scale.item(), step)

                if hasattr(model, "setwise_distance"):
                    writer.add_scalar("temperature logits - Set", model.setwise_distance.temperature.item(), step)

        with torch.no_grad():
            best_unseen, best_acc, best_top1_unseen, best_H, best_seen = test_func(args, test_unseen_loader, train_val_loader, test_seen_loader, model, writer, step, best_unseen, best_acc, best_top1_unseen, best_H, best_seen)
            
        if epoch % args.save_freq == 0:

            state = {'state_dict': model.state_dict(), 
                        'step': step}
            if args.gzsl_flag:
                torch.save(state, os.path.join(args.path_model_save, "gzsl_last.pt"))
            else:
                torch.save(state, os.path.join(args.path_model_save, "last.pt"))

        print("Epoch {} Train Seen Acc: {:.4f}".format(epoch, top1.avg))

        if not args.debug:
            writer.add_scalar("train Seen Acc", top1.avg, epoch)

        if epoch > args.max_epochs:
            break 

    if not args.debug:
        writer.close()

def test_func(args, test_unseen_loader, train_val_loader, test_seen_loader, model, writer, step, best_unseen, best_acc, best_top1_unseen, best_H, best_seen):
    
    acc_record_list = [10, 60, 90, 100]

    model.eval()

    
    if args.gzsl_flag and args.cs_list_flag:
        acc_unseen, acc_unseen_top1, loss_test_unseen, acc_dict_unseen, acc_error_dict_unseen, logits_array_unseen, label_array_unseen = test(args, test_unseen_loader, model, mode="unseen_test", return_logits=True)
        logits_array_unseen = model.calibrated_stacking_func(logits_array_unseen.clone().detach(), -args.calibrated_stacking)
    else:
        acc_unseen, acc_unseen_top1, loss_test_unseen, acc_dict_unseen, acc_error_dict_unseen = test(args, test_unseen_loader, model, mode="unseen_test")
    
    
    if not args.debug:
        writer.add_scalar("acc_unseen_top1", acc_unseen_top1, step) # /args.print_freq) 

    if best_top1_unseen < acc_unseen_top1:
        best_top1_unseen = acc_unseen_top1
    if not args.debug:
        writer.add_scalar("acc_unseen_best_top1", best_top1_unseen, step) # /args.print_freq) 

    if args.gzsl_flag:

        if args.cs_list_flag:
            acc_seen, acc_seen_top1, loss_test_seen, acc_dict_seen, acc_error_dict_seen, logits_array_seen, label_array_seen = test(args, test_seen_loader, model, mode="seen_test", return_logits=True) 
            logits_array_seen = model.calibrated_stacking_func(logits_array_seen.clone().detach(), -args.calibrated_stacking)
        else:
            acc_seen, acc_seen_top1, loss_test_seen, acc_dict_seen, acc_error_dict_seen = test(args, test_seen_loader, model, mode="seen_test") 
        
        acc_H = compute_H(acc_seen, acc_unseen)
        print("acc_H: ", round(acc_H, 2))
        acc = acc_H 
        if not args.debug:
            writer.add_scalar("acc_seen", acc_seen, step) # /args.print_freq) 

        if args.cs_list_flag:
            best_cs_H = 0
            best_cs_seen = 0 
            best_cs_unseen = 0 
            best_cs_value = 0
            best_cs_only_unseen = 0

            global best_best_cs_H
            global best_best_cs_value
            global best_best_cs_seen
            global best_best_cs_unseen
            global best_best_zsl_unseen
            global best_best_epoch_gzsl
            global best_best_epoch_zsl

    else: 
        acc = acc_unseen 
        acc_train_val, acc_train_val_top1, loss_train_val_seen, acc_dict_train_val, acc_error_dict_train_val = test(args, train_val_loader, model, mode="train_val") 
        if not args.debug:
            writer.add_scalar("acc_train_val", acc_train_val, step) # /args.print_freq) 
        
    if args.task_name:
        key_task = args.model_name.replace(args.task_name, "")

        if not os.path.exists(args.task_dir):
            data_temp = {}
            data_temp["best model"] = {}
            data_temp["best model"]["H"] = 0
            data_temp["best model"]["S"] = 0
            data_temp["best model"]["U"] = 0
            data_temp["best model"]["ZSL"] = 0

            save_json(data_temp, args.task_dir)
        
        with open(args.task_dir, 'r') as f:
            data_task_info = json.load(f)

        if args.epoch_now == 0:
            data_task_info[key_task] = {}
    
    if acc > best_acc: 
        best_acc = acc 
                
        if args.gzsl_flag:
            best_H = acc_H
            best_seen = acc_seen
            best_unseen = acc_unseen

            print("\nstep{} Seen Acc {:.4f} Unseen Acc {:.4f} H {:.4f}=>".format(step, best_seen, best_unseen, best_H)) 

            state = {'state_dict': model.state_dict(), 
                     'best_seen': best_seen,
                     'best_unseen': best_unseen,
                     'best_H': best_H}
            
            if args.task_name:
                
                data_task_info[key_task]["best_seen"] = best_seen
                data_task_info[key_task]["best_unseen"] = best_unseen 
                data_task_info[key_task]["best_H"] = best_H
                data_task_info[key_task]["best_epoch"] = args.epoch_now

            torch.save(state, os.path.join(args.path_model_save, f"gzsl_best{args.checkpoint_name}.pt"))

        else:
            best_unseen = acc_unseen

            print("\nstep{} unseen {:.4f}=>".format(step, best_unseen)) 

            state = {'state_dict': model.state_dict(), 
                     'best_unseen': best_unseen}
            
            if args.task_name:
                data_task_info[key_task]["best_unseen"] = best_unseen
                data_task_info[key_task]["best_epoch"] = args.epoch_now

            torch.save(state, os.path.join(args.path_model_save, f"best{args.checkpoint_name}.pt"))

        if not args.debug:
            for key in acc_error_dict_unseen:
                
                figure = plt.figure()

                name_list = []
                acc_list = []

                init_flag = 1

                for temp_category in acc_error_dict_unseen[key]:
                    name_list.append(temp_category)
                    acc_list.append(acc_error_dict_unseen[key][temp_category])

                    if init_flag:
                        init_flag = 0
                        acc_category = acc_error_dict_unseen[key][temp_category]
                        
                plt.bar(range(len(name_list)), acc_list, tick_label=name_list)
                
                for temp_acc_record in acc_record_list:
                    if acc_category < temp_acc_record:
                        category_str = str(temp_acc_record)
                        break

                
            if args.gzsl_flag:
                for key in acc_error_dict_seen:
                    figure = plt.figure()

                    name_list = []
                    acc_list = []

                    init_flag = 1

                    for temp_category in acc_error_dict_seen[key]:
                        name_list.append(temp_category)
                        acc_list.append(acc_error_dict_seen[key][temp_category])

                        if init_flag:
                            init_flag = 0
                            acc_category = acc_error_dict_seen[key][temp_category]

                    for temp_acc_record in acc_record_list:
                        if acc_category <= temp_acc_record:
                            category_str = str(temp_acc_record)
                            break

                    plt.bar(range(len(name_list)), acc_list, tick_label=name_list)

                    
    if args.cs_list_flag:
    
        if args.task_name:
            if "cs info" not in data_task_info[key_task]:
                data_task_info[key_task]["cs info"] = {}
                data_task_info[key_task]["cs info"]["detail"] = {}

        cs_value_original = args.calibrated_stacking

        args.calibrated_stacking = 0 

        for cs_value in args.cs_list:
            args.calibrated_stacking = cs_value 

            logits_array_seen_temp = model.calibrated_stacking_func(logits_array_seen.clone().detach(), cs_value)
            acc_seen_temp = test_use_logits(args, logits_array_seen_temp, label_array_seen, mode="seen_test")

            logits_array_unseen_temp = model.calibrated_stacking_func(logits_array_unseen.clone().detach(), cs_value)
            acc_unseen_temp = test_use_logits(args, logits_array_unseen_temp, label_array_unseen, mode="unseen_test")

            acc_H_temp = compute_H(acc_seen_temp, acc_unseen_temp)

            if args.task_name:
                if str(cs_value) not in data_task_info[key_task]["cs info"]["detail"]:
                    data_task_info[key_task]["cs info"]["detail"][str(cs_value)] = {}

                data_task_info[key_task]["cs info"]["detail"][str(cs_value)]["acc_unseen"] = acc_unseen_temp
                data_task_info[key_task]["cs info"]["detail"][str(cs_value)]["acc_seen"] = acc_seen_temp
                data_task_info[key_task]["cs info"]["detail"][str(cs_value)]["acc_H"] = acc_H_temp

            if acc_H_temp > best_cs_H:
                best_cs_H = acc_H_temp
                best_cs_seen = acc_seen_temp 
                best_cs_unseen = acc_unseen_temp
                best_cs_value = cs_value

                if best_cs_H > best_best_cs_H:
                    best_best_cs_H = best_cs_H
                    best_best_cs_value = best_cs_value
                    best_best_cs_seen = best_cs_seen
                    best_best_cs_unseen = best_cs_unseen
                    best_best_epoch_gzsl = args.epoch_now
            
            best_cs_only_unseen = acc_unseen_temp    
            if best_cs_only_unseen > best_best_zsl_unseen:
                best_best_zsl_unseen = best_cs_only_unseen
                best_best_epoch_zsl = args.epoch_now
            
        if args.task_name:
            data_task_info[key_task]["cs info"]["best_cs_H"] = best_cs_H
            data_task_info[key_task]["cs info"]["best_cs_seen"] = best_cs_seen
            data_task_info[key_task]["cs info"]["best_cs_unseen"] = best_cs_unseen
            data_task_info[key_task]["cs info"]["best_cs_value"] = best_cs_value
            data_task_info[key_task]["cs info"]["best_cs_only_unseen"] = best_cs_only_unseen 

            data_task_info[key_task]["cs info"]["best_best_cs_H"] = best_best_cs_H
            data_task_info[key_task]["cs info"]["best_best_cs_value"] = best_best_cs_value
            data_task_info[key_task]["cs info"]["best_best_cs_seen"] = best_best_cs_seen
            data_task_info[key_task]["cs info"]["best_best_cs_unseen"] = best_best_cs_unseen
            data_task_info[key_task]["cs info"]["best_best_zsl_unseen"] = best_best_zsl_unseen
            data_task_info[key_task]["cs info"]["best_best_epoch_gzsl"] = best_best_epoch_gzsl
            data_task_info[key_task]["cs info"]["best_best_epoch_zsl"] = best_best_epoch_zsl
            
        new_cs_list = []
        update_cs_value_list = [-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04]
        for temp_value in update_cs_value_list:
            if best_cs_value + temp_value > 0:
                new_cs_list.append(best_cs_value + temp_value)

        for cs_value in args.cs_list:
            if cs_value < best_cs_value:
                continue
            if cs_value not in new_cs_list:
                new_cs_list.append(cs_value)
                
        args.cs_list = new_cs_list
        
        args.calibrated_stacking = cs_value_original
        
    if args.task_name and args.cs_list_flag:

        if data_task_info[key_task]["cs info"]["best_best_cs_H"] > data_task_info["best model"]["H"]:

            data_task_info["best model"]["H"] = data_task_info[key_task]["cs info"]["best_best_cs_H"]
            data_task_info["best model"]["S"] = data_task_info[key_task]["cs info"]["best_best_cs_seen"]
            data_task_info["best model"]["U"] = data_task_info[key_task]["cs info"]["best_best_cs_unseen"]
            data_task_info["best model"]["key_task_GZSL"] = key_task
            data_task_info["best model"]["epoch_gzsl"] = data_task_info[key_task]["cs info"]["best_best_epoch_gzsl"]
            
            state = {'state_dict': model.state_dict(), 
                     'best_seen': data_task_info["best model"]["S"],
                     'best_unseen': data_task_info["best model"]["U"],
                     'best_H': data_task_info["best model"]["H"] }
            torch.save(state, os.path.join(args.path_model_save, f"task_{args.task_name}_GZSL.pt"))

        if data_task_info[key_task]["cs info"]["best_best_zsl_unseen"] > data_task_info["best model"]["ZSL"]:
            data_task_info["best model"]["ZSL"] = data_task_info[key_task]["cs info"]["best_best_zsl_unseen"]
            data_task_info["best model"]["key_task_ZSL"] = key_task
            data_task_info["best model"]["epoch_zsl"] = data_task_info[key_task]["cs info"]["best_best_epoch_zsl"]

            state = {'state_dict': model.state_dict(), 
                     'best_unseen': data_task_info["best model"]["U"]}
            torch.save(state, os.path.join(args.path_model_save, f"task_{args.task_name}_ZSL.pt"))

        save_json(data_task_info, args.task_dir)

    if not args.debug:

        if args.gzsl_flag:
            update_info = {'best_seen': best_seen,
                           'best_unseen': best_unseen,
                           'best_H': best_H, 
                           'loss_test_unseen': loss_test_unseen, 
                           'loss_test_seen': loss_test_seen}
            
            if args.cs_list_flag and best_cs_H > 0:
                update_info["best_cs_H"] = best_cs_H
                update_info["best_cs_seen"] = best_cs_seen
                update_info["best_cs_unseen"] = best_cs_unseen
                update_info["best_cs_value"] = best_cs_value
                update_info["best_cs_only_unseen"] = best_cs_only_unseen

                update_info["best_best_cs_H"] = best_best_cs_H
                update_info["best_best_cs_value"] = best_best_cs_value
                update_info["best_best_cs_seen"] = best_best_cs_seen
                update_info["best_best_cs_unseen"] = best_best_cs_unseen
                update_info["best_best_zsl_unseen"] = best_best_zsl_unseen
            
            for key in acc_dict_unseen:
                update_info["z_unseen_" + key] = acc_dict_unseen[key]                
        else:
            
            update_info = {'best_unseen': best_unseen, 
                           'loss_test_unseen': loss_test_unseen}

            for key in acc_dict_unseen:
                update_info["z_unseen_" + key] = acc_dict_unseen[key]

        for key in update_info:
            writer.add_scalar(key, update_info[key], step) # /args.test_freq)
                
    
    return best_unseen, best_acc, best_top1_unseen, best_H, best_seen
