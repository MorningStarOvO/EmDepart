# ==================== Import ==================== #
import time
import sys
import os 
import psutil 

import numpy as np 
import shutil 
import json


# ==================== Functions ==================== #
def path_make_overwrite(path, overwrite=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    elif os.path.exists(path) and overwrite:
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

def path_make_times(path, args, times=0):
    
    if not os.path.exists(path):
        if not args.debug:
            os.makedirs(path, exist_ok=True)
    else:
        times += 1
        if times > 1:
            
            path = path[:-2]
        path = path + "_" + str(times)
        path = path_make_times(path, args, times) 
    
    return path 

def save_code(path_save):
    if os.path.exists(os.path.join(path_save, "tools_new")):
        shutil.rmtree(os.path.join(path_save, "tools_new"))

    shutil.copytree("tools_new", os.path.join(path_save, "tools_new"))

    pid = os.getpid()

    s = psutil.Process(pid)

    command_text = "" 
    for word in s.cmdline():
        command_text += word 
        command_text += " "
    path_text = os.path.join(path_save, "tools_command_text.txt")
    with open(path_text, 'w') as f:
        f.write(command_text)
        
def save_code_and_cmd(path, args):
    path_save_new = path_make_times(path, args)

    if not args.debug:
        shutil.copytree("tools_new", os.path.join(path_save_new, "tools_new"))

        pid = os.getpid()
        
        s = psutil.Process(pid)
    
        command_text = "" 
        for word in s.cmdline():
            command_text += word 
            command_text += " "
        
        path_text = os.path.join(path_save_new, "tools_command_text.txt")
        with open(path_text, 'w') as f:
            f.write(command_text)

        path_seed = os.path.join(path_save_new, "seed.txt")
        with open(path_seed, 'w') as f:
            f.write(str(args.seed))
    
        path_pid = os.path.join(path_save_new, "pid.txt") 
        with open(path_pid, 'w') as f:
            f.write(str(pid))

    return path_save_new

def save_json(dict, path_save):
    
    data_json = json.dumps(dict, indent=4) 
    file = open(path_save, 'w')
    file.write(data_json)
    file.close()

def sorted_dict(dict, reverse=True):
    dict_sorted = sorted(dict.items(), key=lambda d:d[1], reverse=reverse)

    dict_new  = {}
    for key in dict_sorted:
        dict_new[key[0]] = key[1]

    return dict_new
    
def sorted_dict_and_save_json(dict, path_save, reverse=True):
    
    dict_sorted = sorted(dict.items(), key=lambda d:d[1], reverse=reverse)

    dict_new  = {}
    for key in dict_sorted:
        dict_new[key[0]] = key[1]

    data_json = json.dumps(dict_new, indent=4) 
    file = open(path_save, 'w')
    file.write(data_json)
    file.close()

    return dict_new

def get_path_file(path, path_new=None):
    
    path_file_list = os.listdir(path)

    if path_new != None:
        path_new_list = []

    path_list = []
    for path_temp in path_file_list:
        path_list_temp = os.listdir(os.path.join(path, path_temp))
        for temp in path_list_temp:
            
            temp_old = os.path.join(path, path_temp, temp)
            path_list.append(temp_old)

            if path_new != None:
                if not os.path.exists(os.path.join(path_new, path_temp)):
                    os.makedirs(os.path.join(path_new, path_temp), exist_ok=True)
                temp = os.path.join(path_new, path_temp, temp)
                path_new_list.append(temp)
            
    if path_new != None:
        return path_list, path_new_list
    else:
        return path_list 

