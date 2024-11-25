# ==================== Import ==================== #
import time
import sys
import os 

import numpy as np 
import json 

import torch 
import torchvision.datasets as datasets 
import torchtext.vocab as vocab

from PIL import Image
from torch.utils.data import Dataset

from nltk.tokenize import word_tokenize

# ==================== Function ==================== #
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB') 

class GBU_dataset(Dataset):
    def __init__(self, data_root, image_files, labels, old_label_to_new_label_dict, transform=None, 
                 use_extracted_features_flag=False, extracted_features_root=None, 
                 use_text_logit_flag=False, data_select_logit_root=None, max_len=0):
        super().__init__()

        self.transform = transform

        self.data_root = data_root
        self.path_image_list = image_files
        self.label_list = []

        for old_label in labels:
            self.label_list.append(old_label_to_new_label_dict[old_label])

        self.use_extracted_features_flag = use_extracted_features_flag
        if use_extracted_features_flag:
            self.extracted_features = np.load(extracted_features_root, allow_pickle=True).item()

        self.use_text_logit_flag = use_text_logit_flag
        if use_text_logit_flag:
            data_text_logit = np.load(data_select_logit_root, allow_pickle=True).item()

            self.data_text_logit_dict = {}

            for image_file in data_text_logit:
                temp_logit = np.concatenate((data_text_logit[image_file], np.zeros(max_len - 1 - len(data_text_logit[image_file]))))
                self.data_text_logit_dict[image_file] = temp_logit

    def __getitem__(self, index):
        path_image = os.path.join(self.data_root, self.path_image_list[index])
        label = self.label_list[index]
        
        if not self.use_extracted_features_flag:
            img = pil_loader(path_image)
            if self.transform:
                img = self.transform(img)
        else:
            img = torch.from_numpy(self.extracted_features[path_image])

        label = torch.tensor(label)

        if self.use_text_logit_flag:
            text_logit = torch.from_numpy(self.data_text_logit_dict[self.path_image_list[index]]).float()
            return img, label, text_logit
        else:
            return img, label

    def __len__(self):
        return len(self.path_image_list)


def collate_fn(data):

    data = list(zip(*data))

    res = {"img" : data[0], 
           "path_image" : data[1]}
           
    del data
    return res


class GBU_extracted_feature_dataset(Dataset):
    def __init__(self, data_root, image_files, transform=None):
        super().__init__()

        self.transform = transform

        self.data_root = data_root
        self.path_image_list = image_files

    def __getitem__(self, index):
        path_image = os.path.join(self.data_root, self.path_image_list[index])

        img = pil_loader(path_image)
        if self.transform:
            img = self.transform(img)

        return img, path_image

    def __len__(self):
        return len(self.path_image_list)
    
class GBU_image_definition_dataset(Dataset):
    def __init__(self, data_root, image_files, labels, old_label_to_new_label_dict, data_image_definition_root, max_len, 
                 transform=None, use_extracted_features_flag=False, extracted_features_root=None, d_text_glove_embedding=300):
        super().__init__()

        self.transform = transform

        self.data_root = data_root
        self.path_image_list = image_files
        self.label_list = []
        for old_label in labels:
            self.label_list.append(old_label_to_new_label_dict[old_label])

        self.use_extracted_features_flag = use_extracted_features_flag
        if use_extracted_features_flag:
            self.extracted_features = np.load(extracted_features_root, allow_pickle=True).item()

        path_image_definition_embedding = data_image_definition_root.split(".")[0] + "_embedding.pt"
        path_image_definition_mask = data_image_definition_root.split(".")[0] + "_mask.pt"

        if not os.path.exists(path_image_definition_embedding):
            
            with open(data_image_definition_root, 'r') as f:
                data_image_definition = json.load(f)

            self.image_definition_embedding_list = torch.zeros(len(image_files), max_len-1, d_text_glove_embedding)
            self.image_definition_mask_list = []

            glove = vocab.GloVe(name='6B', dim=d_text_glove_embedding) 

            eos_embed = torch.zeros(d_text_glove_embedding)

            for i in range(len(image_files)):
                temp_image_file = image_files[i]
                temp_image_definition = data_image_definition[temp_image_file]

                doc = word_tokenize(temp_image_definition)

                len_text = 0 
                mask_matrix = [False]
                sentence_embed = 0 

                for temp_token in doc:
                    if temp_token in glove.stoi:
                        temp_vector =  glove.vectors[glove.stoi[temp_token]]
                        mask_matrix.append(False)
                    elif temp_token.lower() in glove.stoi:
                        temp_vector =  glove.vectors[glove.stoi[temp_token.lower()]]
                        mask_matrix.append(False)
                    else:
                        temp_vector = eos_embed
                        mask_matrix.append(True)
                    
                    if len_text == 0:
                        sentence_embed = temp_vector.unsqueeze(0)
                    else:
                        sentence_embed = torch.cat((sentence_embed, temp_vector.unsqueeze(0)), dim=0)

                    len_text += 1
                
                mask_matrix = torch.tensor(mask_matrix)
                
                if len_text < max_len - 1:
                    sentence_embed = torch.cat((sentence_embed, eos_embed.expand(max_len - 1 - len_text, eos_embed.shape[0])), dim=0)
                    mask_matrix = torch.cat((mask_matrix, torch.tensor(True).expand(max_len - 1 - len_text)))
                elif len_text > max_len - 1:
                    print(temp_image_file)
                    sys.exit()

                self.image_definition_embedding_list[i] = sentence_embed
                self.image_definition_mask_list.append(mask_matrix)

            torch.save(self.image_definition_embedding_list, path_image_definition_embedding)
            torch.save(self.image_definition_mask_list, path_image_definition_mask)
                 
        else:
            self.image_definition_embedding_list = torch.load(path_image_definition_embedding)
            self.image_definition_mask_list = torch.load(path_image_definition_mask)


    def __getitem__(self, index):
        path_image = os.path.join(self.data_root, self.path_image_list[index])
        label = self.label_list[index]

        if not self.use_extracted_features_flag:
            img = pil_loader(path_image)
            if self.transform:
                img = self.transform(img)
        else:
    
            img = torch.from_numpy(self.extracted_features[path_image])

        label = torch.tensor(label)

        image_definition_embedding = self.image_definition_embedding_list[index]
        image_definition_mask = self.image_definition_mask_list[index]

        return img, label, image_definition_embedding, image_definition_mask

    def __len__(self):
        return len(self.path_image_list)

