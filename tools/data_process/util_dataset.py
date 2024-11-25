# ==================== Import ==================== #
import time
import sys
import os 

import numpy as np 
import scipy.io as scio
import json 

# ==================== Function ==================== #
def get_classes_name_to_label_dict(path, classes_name_to_label_dict=None):

    with open(path, 'r') as f:
        data = f.readlines()
    
    if classes_name_to_label_dict == None:
        classes_name_to_label_dict = {}
        index = 0 
    else: 
        index = len(classes_name_to_label_dict)

    for line in data:
        class_name = line.strip("\n").replace("+", " ")
        classes_name_to_label_dict[class_name] = index 

        index += 1

    return classes_name_to_label_dict 

def process_CUB_image_file(image_file_list):
    new_image_file_list = []

    for image_file in image_file_list:
        new_image_file_list.append(image_file[41:])
      
    return new_image_file_list

def process_FLO_image_file(image_file_list):
    new_image_file_list = []

    for image_file in image_file_list:
        new_image_file_list.append(image_file.split("/")[-1])
    
    return new_image_file_list

def read_mat_dataset(path_att_mat, path_res_mat, validation_flag=0, dataset="awa2", path_flo_label_to_category="data/FLO_label_to_category_name.json"):

    mat_content = scio.loadmat(path_res_mat)
    label = mat_content["labels"].astype(int).squeeze() - 1 
    image_files_original = mat_content["image_files"] 

    image_files = []
    for temp_path_image in image_files_original: 
        image_files.append(temp_path_image[0][0].split("//")[-1])
    image_files = np.array(image_files)

    mat_content = scio.loadmat(path_att_mat)

    if dataset == "flo":
        allclasses_name = []
        with open(path_flo_label_to_category, 'r') as f:
            label_to_category_dict = json.load(f)

        for temp_label in label_to_category_dict:
            allclasses_name.append(label_to_category_dict[temp_label])

    else:
        allclasses_name_original = mat_content["allclasses_names"]

        allclasses_name = []
        for temp_category in allclasses_name_original:
            allclasses_name.append(temp_category[0][0].replace("+", " "))
            
    trainval_loc = mat_content["trainval_loc"].squeeze() - 1
    train_loc = mat_content['train_loc'].squeeze() - 1
    val_unseen_loc = mat_content['val_loc'].squeeze() - 1
    test_seen_loc = mat_content['test_seen_loc'].squeeze() - 1
    test_unseen_loc = mat_content['test_unseen_loc'].squeeze() - 1

    if not validation_flag: 
        train_image_files = image_files[trainval_loc.astype(int)]
        train_label = label[trainval_loc.astype(int)]
        
        test_unseen_image_files = image_files[test_unseen_loc.astype(int)]
        test_unseen_label = label[test_unseen_loc.astype(int)]
        test_seen_image_files = image_files[test_seen_loc.astype(int)]
        test_seen_label = label[test_seen_loc.astype(int)]

        if dataset == "cub":
            train_image_files = process_CUB_image_file(train_image_files)
            test_unseen_image_files = process_CUB_image_file(test_unseen_image_files)
            test_seen_image_files = process_CUB_image_file(test_seen_image_files)
        elif dataset == "flo":
            train_image_files = process_FLO_image_file(train_image_files)
            test_unseen_image_files = process_FLO_image_file(test_unseen_image_files)
            test_seen_image_files = process_FLO_image_file(test_seen_image_files)

        seenclasses = np.unique(test_seen_label)
        unseenclasses = np.unique(test_unseen_label)

        seen_classes_name_to_index_dict = {}
        for index in seenclasses:
            seen_classes_name_to_index_dict[allclasses_name[index]] = index 

        unseen_classes_name_to_index_dict = {}
        for index in unseenclasses: 
            unseen_classes_name_to_index_dict[allclasses_name[index]] = index 

        return train_image_files, train_label, test_unseen_image_files, test_unseen_label, \
               test_seen_image_files, test_seen_label, seen_classes_name_to_index_dict, unseen_classes_name_to_index_dict

def get_AWA_classesname(path="data/Animals_with_Attributes2/classes.txt"):
    """get classes name in AWA dataset"""

    with open(path, 'r') as f:
        data = f.readlines()

    classes_name_list = []
    for line in data:
        classes_name_list.append(line.split("\t")[1].split("\n")[0].replace("+", " "))

    return classes_name_list

def get_seen_or_unseen_classes_name(path):
    with open(path, 'r') as f:
        data = f.readlines()

    classes_name_list = []
    for line in data:
        classes_name_list.append(line.split("\n")[0].replace("+", " "))

    return classes_name_list
