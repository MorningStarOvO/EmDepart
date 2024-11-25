# ==================== Import ==================== #
import time
import sys
import os 
import shutil
import psutil

import numpy as np 


# ==================== Functions ==================== #
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def exit(self):
        self.log.close()


def get_command_test():
    
    pid = os.getpid()
    
    s = psutil.Process(pid)

    command_text = "" 
    for word in s.cmdline():
        command_text += word 
        command_text += " "
    print()
    print(command_text)

def build_save_file(args):

    path_dataset_output = os.path.join(args.checkpoint_path, args.dataset, args.task_name)
    if not os.path.exists(path_dataset_output):
        os.makedirs(path_dataset_output, exist_ok=True)
    args.path_dataset_output = path_dataset_output

    path_model_save = os.path.join(path_dataset_output, "model")
    if not os.path.exists(path_model_save):
        os.makedirs(path_model_save, exist_ok=True)
    args.path_model_save = path_model_save

    path_log_save = os.path.join(path_dataset_output, "log")
    if args.task_name:
        path_log_save = os.path.join(path_log_save, args.task_name)

    if not os.path.exists(path_log_save):
        os.makedirs(path_log_save, exist_ok=True)

    path_tensorboard_save = os.path.join(path_dataset_output, "TensorBoard")
    if not os.path.exists(path_tensorboard_save):
        os.makedirs(path_tensorboard_save, exist_ok=True)

    path_code_save = os.path.join(path_dataset_output, "code")
    if not os.path.exists(path_code_save):
        os.makedirs(path_code_save, exist_ok=True)

    path_task_save = os.path.join(args.checkpoint_path, args.dataset, "task")
    if not os.path.exists(path_task_save):
        os.makedirs(path_task_save, exist_ok=True)

    if args.model_name != None:
        model_name = args.model_name
    elif args.gzsl_flag:
        model_name = f"{args.semantic_embedding}_gzsl_seed{str(args.seed)}_bz{str(args.batch_size)}_epochs{str(args.epochs)}_{args.optim}_lr_{str(args.learning_rate)}_decay_{str(args.weight_decay)}_eps_{args.adam_epsilon}"
    else:
        model_name = f"{args.semantic_embedding}_seed{str(args.seed)}_bz{str(args.batch_size)}_epochs{str(args.epochs)}_{args.optim}_lr_{str(args.learning_rate)}_decay_{str(args.weight_decay)}_eps_{args.adam_epsilon}"

    args.model_name = model_name

    if not args.debug:        
        log_dir = os.path.join(path_log_save, f"{model_name}.log")

        if os.path.exists(log_dir):
            os.remove(log_dir)
        sys.stdout = Logger(filename=log_dir, stream=sys.stdout)

        code_dir = os.path.join(path_code_save, model_name)

        if os.path.exists(code_dir):
            shutil.rmtree(code_dir)
        shutil.copytree("tools", code_dir)

        tensorboard_dir = os.path.join(path_tensorboard_save, model_name)
        if not os.path.exists(tensorboard_dir):
            os.makedirs(tensorboard_dir)
        args.tensorboard_dir = tensorboard_dir

        task_dir = os.path.join(path_task_save, f"{args.task_name}.json")
        args.task_dir = task_dir
        
        get_command_test()
