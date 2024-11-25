# ==================== Import ==================== #
import time
import sys
import os 

import numpy as np 

from torch.utils.data import DataLoader
from torchvision import transforms 

from pprint import pprint  

from data_process.util_dataset import read_mat_dataset 
from data_process.GUB_dataset import GBU_dataset, GBU_extracted_feature_dataset, collate_fn, GBU_image_definition_dataset
from utils.util_path import save_json

# ==================== Function ==================== #

def get_named_data(args, train_mode="default", train_no_shuffle_flag=False):
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_val = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
            ])
    
    
    transform_train = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ])
    
    if args.dataset.lower() == "awa2":
        path_att_mat = os.path.join(args.data_split_root, "xlsa17/data", "AWA2", "att_splits.mat")
        path_res_mat = os.path.join(args.data_split_root, "xlsa17/data", "AWA2", "res101.mat")
    elif args.dataset.lower() == "cub":
        path_att_mat = os.path.join(args.data_split_root, "xlsa17/data", "CUB", "att_splits.mat")
        path_res_mat = os.path.join(args.data_split_root, "xlsa17/data", "CUB", "res101.mat")
    elif args.dataset.lower() == "flo":
        path_att_mat = os.path.join(args.data_split_root, "xlsa17/data", "FLO", "att_splits.mat")
        path_res_mat = os.path.join(args.data_split_root, "xlsa17/data", "FLO", "res101.mat")
    else:
        sys.exit()

    train_image_files, train_label, test_unseen_image_files, test_unseen_label, \
    test_seen_image_files, test_seen_label, seen_classes_name_to_index_dict, unseen_classes_name_to_index_dict = read_mat_dataset(path_att_mat, path_res_mat, validation_flag=args.validation_flag, dataset=args.dataset.lower())

    train_old_label_to_new_label_dict = {}
    train_classes_name_to_index_dict = {}
    test_new_label_to_classes_name_unseen_dict = {}
    test_new_label_to_classes_name_seen_dict = {}

    index = 0 
    for class_name in seen_classes_name_to_index_dict:
        label = seen_classes_name_to_index_dict[class_name]

        train_old_label_to_new_label_dict[label] = index  
        train_classes_name_to_index_dict[class_name] = index 

        test_new_label_to_classes_name_seen_dict[str(index)] = class_name

        index += 1 

    test_unseen_old_label_to_new_label_dict = {}
    test_seen_old_label_to_new_label_dict = train_old_label_to_new_label_dict.copy()
    if args.gzsl_flag:
        test_classes_name_to_index_dict = train_classes_name_to_index_dict.copy()
        index = len(train_classes_name_to_index_dict)
        args.seen_classes_num = index
    else: 
        test_classes_name_to_index_dict = {}
        index = 0 

    for class_name in unseen_classes_name_to_index_dict: 
        label = unseen_classes_name_to_index_dict[class_name] 

        test_unseen_old_label_to_new_label_dict[label] = index
        test_classes_name_to_index_dict[class_name] = index 

        test_new_label_to_classes_name_unseen_dict[str(index)] = class_name

        index += 1

    if train_mode == "default":

        if args.text_logit_loss_flag and args.image_enhanced_model_flag:
            use_text_logit_flag = True 
        else:
            use_text_logit_flag = False

        train_set = GBU_dataset(args.data_root, train_image_files, train_label, train_old_label_to_new_label_dict, 
                                transform_train, args.use_extracted_features_flag, os.path.join(args.extracted_features_root, "train.npy"), 
                                use_text_logit_flag, args.data_select_logit_root, args.max_context_length)
    elif "image_definition" in train_mode: 
        train_set = GBU_image_definition_dataset(args.data_root, train_image_files, train_label, train_old_label_to_new_label_dict, args.data_image_definition_root, args.max_context_length, 
                                                 transform_train, args.use_extracted_features_flag, os.path.join(args.extracted_features_root, "train.npy"), args.d_text_glove_embedding)

    test_unseen_set = GBU_dataset(args.data_root, test_unseen_image_files, test_unseen_label, test_unseen_old_label_to_new_label_dict, 
                                  transform_val, args.use_extracted_features_flag, os.path.join(args.extracted_features_root, "unseen.npy"))
    

    train_val_set = GBU_dataset(args.data_root, test_seen_image_files, test_seen_label, test_seen_old_label_to_new_label_dict, transform_val, 
                                args.use_extracted_features_flag, os.path.join(args.extracted_features_root, "seen.npy"))
    
    if args.gzsl_flag:
        test_seen_set = GBU_dataset(args.data_root, test_seen_image_files, test_seen_label, test_seen_old_label_to_new_label_dict, transform_val, 
                                    args.use_extracted_features_flag, os.path.join(args.extracted_features_root, "seen.npy"))
    
    if train_no_shuffle_flag:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, 
                                num_workers=args.workers, pin_memory=False) 
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, 
                                num_workers=args.workers, pin_memory=False)
        
    test_unseen_loader = DataLoader(test_unseen_set, batch_size=args.batch_size, shuffle=False, 
                                    num_workers=args.workers, pin_memory=False)

    train_val_loader = DataLoader(train_val_set, batch_size=args.batch_size, shuffle=False, 
                                  num_workers=args.workers, pin_memory=False)

    if args.gzsl_flag:
        test_seen_loader = DataLoader(test_seen_set, batch_size=args.batch_size, shuffle=False, 
                                      num_workers=args.workers, pin_memory=False)

    if args.gzsl_flag:
        save_json(test_new_label_to_classes_name_unseen_dict, os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_unseen_gzsl.json"))
        save_json(test_new_label_to_classes_name_seen_dict, os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_seen_gzsl.json"))
    else:
        save_json(test_new_label_to_classes_name_unseen_dict, os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_unseen_zsl.json"))
        save_json(test_new_label_to_classes_name_seen_dict, os.path.join(args.path_dataset_output, "test_new_label_to_classes_name_seen_zsl.json"))

    args.print_freq = 1 # len(train_loader) // 3
    args.save_freq = 10

    if args.gzsl_flag:
        if not args.debug:
            print("\nGZSL Settings =>")
            print(f"train: {len(train_set)}, test_seen: {len(test_seen_set)}, test_unseen: {len(test_unseen_set)}")

        return train_loader, test_seen_loader, test_unseen_loader, train_classes_name_to_index_dict, test_classes_name_to_index_dict
    else: 
        if not args.debug:
            print("\nZSL Settings =>")
            print(f"train: {len(train_set)}, train val:{len(train_val_set)}, test_unseen: {len(test_unseen_set)}")

        return train_loader, test_unseen_loader, train_val_loader, train_classes_name_to_index_dict, test_classes_name_to_index_dict 
