# ==================== Import ==================== #
import time
import sys
import os 

import numpy as np 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import timm  
from timm.loss import LabelSmoothingCrossEntropy
from torch import einsum 

from einops import rearrange, repeat

from models.model_util import Text_Encoder, MLP_text, MLP_image_layer_2, MLP_image_layer_3, load_text_embedding, Attention_layer, Attention_layer_true_i2mv, I2DFormer_logits_local, max_pool, l2norm
from models.SDM.aggregation_block import SDM 
from models.distance_util import SetwiseDistance
from models.norm_loss_util import diversity_loss

# ==================== Functions ==================== #
class EmDepart_model(nn.Module):
    def __init__(self, args, train_classes_name_to_index, test_classes_name_to_index):
        super().__init__()

        # ------------------------- #
        # ------------------------- #
        category_glove_embedding_dict = load_text_embedding(args.data_document_root, args.max_context_length, args.path_cache, args.d_text_glove_embedding)

        self.train_glove_embedding, self.train_mask_matrix = self.load_index_to_glove_and_mask_list(category_glove_embedding_dict, train_classes_name_to_index)
        self.test_glove_embedding, self.test_mask_matrix = self.load_index_to_glove_and_mask_list(category_glove_embedding_dict, test_classes_name_to_index)

        self.gzsl_flag = args.gzsl_flag
        if args.gzsl_flag:
            self.seen_label_index_list = []
            for key in train_classes_name_to_index:
                self.seen_label_index_list.append(train_classes_name_to_index[key])

            self.calibrated_stacking = args.calibrated_stacking
            if not self.calibrated_stacking:
                self.calibrated_stacking = torch.tensor(self.calibrated_stacking).cuda()

            self.cs_list_mode = args.cs_list_mode

        # ------------------------- #
        # ------------------------- #
        self.lambda_CLS = args.lambda_CLS 
        self.only_global_loss_flag = args.only_global_loss_flag 
        self.set_lambda_local_flag = args.set_lambda_local_flag
        self.lambda_local = args.lambda_local
        self.global_loss_mode = args.global_loss_mode
        
        self.lambda_diversity = args.lambda_diversity

        self.lambda_variance = args.lambda_variance

        if self.global_loss_mode == "triplet":
            self.margin = args.triplet_margin
            self.semi_hard_triplet = args.semi_hard_triplet_flag

        # ------------------------- # 
        # ------------------------- # 
        self.image_embedding = args.image_embedding
        if args.image_embedding == "vit-base-patch16-224":
            self.feature_extractor = timm.create_model('vit_base_patch16_224', pretrained=True) 

        # ------------------------- # 
        #     text transformer      # 
        # ------------------------- # 
        self.text_encoder = Text_Encoder(context_length=args.max_context_length, 
                                         transformer_width=args.d_semantic_space, 
                                         transformer_layers=args.num_text_layers, 
                                         transformer_heads=args.num_heads, 
                                         positional_embedd_flag=args.positional_embedd_flag)

        self.perceiver_softmax_mode = args.perceiver_softmax_mode

        self.set_prediction_module_text = SDM(dim=args.d_semantic_space, 
                                                             depth=args.perceiver_text_depth, 
                                                             heads=args.perceiver_text_head,
                                                             num_latents=args.perceiver_text_num_set,
                                                             ff_mult=args.perceiver_text_ff_mult, 
                                                             dim_head=args.perceiver_text_dim_head,
                                                             variance_constant=args.variance_constant, 
                                                             dropout=args.dropout_rate, 
                                                             more_drop_flag=args.perceiver_more_drop_flag, 
                                                             variance_after_softmax_flag=args.variance_after_softmax_flag)

        self.set_prediction_module_image = SDM(dim=args.d_semantic_space, 
                                                              depth=args.perceiver_image_depth, 
                                                              heads=args.perceiver_image_head,
                                                              num_latents=args.perceiver_image_num_set,
                                                              ff_mult=args.perceiver_image_ff_mult, 
                                                              dim_head=args.perceiver_image_dim_head,
                                                              variance_constant=args.variance_constant,
                                                              dropout=args.dropout_rate, 
                                                              more_drop_flag=args.perceiver_more_drop_flag,
                                                              variance_after_softmax_flag=args.variance_after_softmax_flag)

        # ------------------------- #
        #           Score           #
        # ------------------------- #
        self.similarity_func_mode = args.similarity_func_mode
        if self.global_loss_mode == "triplet":
            args.similarity_temperature = 1 

        self.global_fuse_mode = args.global_fuse_mode

        if self.global_fuse_mode == "default" or self.global_fuse_mode == "without_global":
            self.setwise_distance = SetwiseDistance(img_set_size=args.perceiver_image_num_set, txt_set_size=args.perceiver_text_num_set, temperature=args.similarity_temperature, scale=args.smooth_chamfer_scale, fixed_temperature_flag=args.fixed_temperature_flag)
        elif self.global_fuse_mode == "concat" or self.global_fuse_mode == "concat_and_perceiver":
            self.setwise_distance = SetwiseDistance(img_set_size=args.perceiver_image_num_set+1, txt_set_size=args.perceiver_text_num_set+1, temperature=args.similarity_temperature, scale=args.smooth_chamfer_scale, fixed_temperature_flag=args.fixed_temperature_flag)

        if self.similarity_func_mode == "avg_distance":
            self.setwise_distance_func = self.setwise_distance.avg_distance
        elif self.similarity_func_mode == 'max_distance':
            self.setwise_distance_func = self.setwise_distance.max_distance
        elif  self.similarity_func_mode == 'chamfer_distance':
            self.setwise_distance_func = self.setwise_distance.chamfer_distance
        elif  self.similarity_func_mode == 'smooth_chamfer_distance':
            self.setwise_distance_func = self.setwise_distance.smooth_chamfer_distance

        # ------------------------- # 
        #         projection        # 
        # ------------------------- # 
        self.text_projection = MLP_text(d_model_in=args.d_text_glove_embedding, 
                                        d_model_out=args.d_semantic_space, 
                                        dropout_rate=args.dropout_rate,
                                        d_model_mid=args.d_middle_space_text,
                                        text_mlp_res_flag=args.text_mlp_res_flag)
    
        if args.num_image_projection == 2:
            self.image_projection = MLP_image_layer_2(d_model_in=args.d_image_transformer, 
                                                      d_model_out=args.d_semantic_space, 
                                                      dropout_rate=args.dropout_rate,
                                                      d_model_mid=args.d_middle_space,
                                                      image_mlp_res_flag=args.image_mlp_res_flag)
        elif args.num_image_projection == 3:
            self.image_projection = MLP_image_layer_3(d_model_in=args.d_image_transformer, 
                                                      d_model_out=args.d_semantic_space, 
                                                      dropout_rate=args.dropout_rate,
                                                      d_model_mid=args.d_middle_space, 
                                                      image_mlp_res_flag=args.image_mlp_res_flag)


        # ------------------------- # 
        #        Prototypes          #
        # ------------------------- # 
        self.set_prediction_module_mode = args.set_prediction_module_mode 
        
        
        # ------------------------- #
        #     Layer Norm 相关       #
        # ------------------------- #
        self.norm_res_local_text = nn.LayerNorm(args.d_semantic_space)
        self.norm_res_local_image = nn.LayerNorm(args.d_semantic_space)
        

        # ------------------------- #
        # ------------------------- #
        if args.attention_layer_mode == "default":
            self.I2DFormer_attention_layer = Attention_layer(d_model=args.d_semantic_space)
        elif args.attention_layer_mode == "mode_i2mv":
            self.I2DFormer_attention_layer = Attention_layer_true_i2mv(d_model=args.d_semantic_space, heads=args.attention_layer_head)

        self.I2DFormer_logits_local = I2DFormer_logits_local(args.d_semantic_space, args.pooling_method)
    
    def encoder_image(self, image): 
        with torch.no_grad():

            if self.image_embedding == "vit-base-patch16-224":
                image_features = self.feature_extractor.forward_features(image)

        return image_features
    
    def compute_loss(self, image, label, mode="train", cs_value=None):
        
        outputs = {}

        # ----- logits ----- # 
        output_forward = self.forward(image, mode=mode)
        logits_per_image = output_forward["logits_per_image"]
        logits_per_image_local = output_forward["logits_per_image_local"]
        img_embs = output_forward["img_embs"]
        txt_embs = output_forward["txt_embs"]
        text_set_embeddings = output_forward["text_set_embeddings"]
        image_set_embeddings = output_forward["image_set_embeddings"]
        

        loss_diversity = (diversity_loss(text_set_embeddings, text_set_embeddings.shape[1]) + diversity_loss(image_set_embeddings, image_set_embeddings.shape[1])) / 2

        if self.global_loss_mode == "contrastive":
            loss_global = F.cross_entropy(logits_per_image, label)
        
        loss_local = F.cross_entropy(logits_per_image_local, label)

        if self.only_global_loss_flag:
            loss = loss_global
        elif self.set_lambda_local_flag:
            loss = self.lambda_CLS * loss_global + self.lambda_local * loss_local  
        else:
            loss = self.lambda_CLS * loss_global + self.lambda_local * loss_local 

        # diversity loss # 
        if self.lambda_diversity > 0:
            loss += self.lambda_diversity * loss_diversity
    
        if self.lambda_variance > 0:
            loss += self.lambda_variance * output_forward["loss_variance"]

        if "test" in mode and self.gzsl_flag and self.calibrated_stacking and self.cs_list_mode == "logits":
            logits_per_image = self.calibrated_stacking_func(logits_per_image, cs_value)

        logits_per_image_class = logits_per_image.softmax(dim=-1)
        
        if "test" in mode and self.gzsl_flag and self.calibrated_stacking and self.cs_list_mode == "after_softmax":
            logits_per_image_class = self.calibrated_stacking_func(logits_per_image_class, cs_value)
        
        outputs["logits_per_image"] = logits_per_image_class 
        outputs["logits_per_image_init"] = logits_per_image 
        outputs["loss_global"] = loss_global
        outputs["loss_local"] = loss_local
        outputs["loss_diversity"] = loss_diversity
        outputs["loss_variance"] = output_forward["loss_variance"]
        outputs["loss"] = loss
        
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        return outputs

    def forward(self, image, mode="train"):

        output_forward = {}

        image_patch_features = self.encoder_image(image) 
        image_projection_features = self.image_projection(image_patch_features)

        if mode == "train" or mode == "train_val":
            train_token_embedding = self.text_projection(self.train_glove_embedding)
            text_token_features = self.text_encoder(train_token_embedding, key_padding_mask=self.train_mask_matrix)
            mask_matrix = self.train_mask_matrix
        elif "test" in mode: 
            test_token_embedding = self.text_projection(self.test_glove_embedding)
            text_token_features = self.text_encoder(test_token_embedding, key_padding_mask=self.test_mask_matrix)
            mask_matrix = self.test_mask_matrix
    
        text_global_feature = text_token_features[:, 0, :]
        text_local_features = text_token_features[:, 1:, :] 
        
        image_global_feature = image_projection_features[:, 0, :]
        image_local_features = image_projection_features[:, 1:, :]

        local_attention_values = self.I2DFormer_attention_layer(image_local_features, text_local_features, key_padding_mask=mask_matrix[:, 1:])
        logits_per_image_local = self.I2DFormer_logits_local(local_attention_values)

        if self.global_fuse_mode == "without_global":
            text_set_embeddings, loss_variance_text = self.set_prediction_module_text(text_local_features, mask=mask_matrix[:, 1:], softmax_mode=self.perceiver_softmax_mode, variance_flag=True)
            image_set_embeddings, loss_variance_image = self.set_prediction_module_image(image_local_features, softmax_mode=self.perceiver_softmax_mode, variance_flag=True)
            
            text_res_set_embeddings = self.norm_res_local_text(text_set_embeddings)
            image_res_set_embeddings = self.norm_res_local_image(image_set_embeddings)
        elif self.global_fuse_mode == "default":
            text_set_embeddings, loss_variance_text = self.set_prediction_module_text(text_local_features, mask=mask_matrix[:, 1:], softmax_mode=self.perceiver_softmax_mode, variance_flag=True)
            image_set_embeddings, loss_variance_image = self.set_prediction_module_image(image_local_features, softmax_mode=self.perceiver_softmax_mode, variance_flag=True)
            
            text_res_set_embeddings = self.norm_res_local_text(text_set_embeddings + text_global_feature.unsqueeze(1))
            image_res_set_embeddings = self.norm_res_local_image(image_set_embeddings + image_global_feature.unsqueeze(1))
        
        elif self.global_fuse_mode == "concat":
            text_set_embeddings, loss_variance_text = self.set_prediction_module_text(text_local_features, mask=mask_matrix[:, 1:], softmax_mode=self.perceiver_softmax_mode, variance_flag=True)
            image_set_embeddings, loss_variance_image = self.set_prediction_module_image(image_local_features, softmax_mode=self.perceiver_softmax_mode, variance_flag=True)

            text_res_set_embeddings = torch.cat((text_global_feature.unsqueeze(1), text_set_embeddings), dim=1)
            image_res_set_embeddings = torch.cat((image_global_feature.unsqueeze(1), image_set_embeddings), dim=1)
        elif self.global_fuse_mode == "concat_and_perceiver":
            text_set_embeddings, loss_variance_text = self.set_prediction_module_text(text_token_features, mask=mask_matrix, softmax_mode=self.perceiver_softmax_mode, variance_flag=True)
            image_set_embeddings, loss_variance_image = self.set_prediction_module_image(image_projection_features, softmax_mode=self.perceiver_softmax_mode, variance_flag=True)

            text_res_set_embeddings = torch.cat((text_global_feature.unsqueeze(1), text_set_embeddings), dim=1)
            image_res_set_embeddings = torch.cat((image_global_feature.unsqueeze(1), image_set_embeddings), dim=1)

        loss_variance = (loss_variance_text + loss_variance_image) / 2

        text_res_set_embeddings = text_res_set_embeddings / (text_res_set_embeddings.norm(dim=-1, keepdim=True) + 1e-10)
        image_res_set_embeddings = image_res_set_embeddings / (image_res_set_embeddings.norm(dim=-1, keepdim=True))

        img_embs = image_res_set_embeddings.reshape(-1, image_res_set_embeddings.shape[-1])
        txt_embs = text_res_set_embeddings.reshape(-1, text_res_set_embeddings.shape[-1])

        logits_per_image = self.setwise_distance_func(img_embs, txt_embs)

        output_forward["logits_per_image"] = logits_per_image 
        output_forward["logits_per_image_local"] = logits_per_image_local
        output_forward["img_embs"] = img_embs
        output_forward["txt_embs"] = txt_embs
        output_forward["text_set_embeddings"] = text_set_embeddings
        output_forward["image_set_embeddings"] = image_set_embeddings
        output_forward["loss_variance"] = loss_variance

        return output_forward

    def calibrated_stacking_func(self, logits, cs_value):
        
        logits[:, self.seen_label_index_list] -= cs_value

        return logits

    def load_index_to_glove_and_mask_list(self, category_glove_embedding_dict, classes_name_to_index_dict):

        glove_embed = 0 
        mask_matrix = 0 

        index_to_category_dict = {}
        for category in classes_name_to_index_dict:
            index_to_category_dict[classes_name_to_index_dict[category]] = category

        for i in range(len(index_to_category_dict)):
            if i == 0:
                glove_embed = category_glove_embedding_dict[index_to_category_dict[i]]["glove_embed"].unsqueeze(0)
                mask_matrix = category_glove_embedding_dict[index_to_category_dict[i]]["mask_matrix"].unsqueeze(0)
            else: 
                temp_glove_embed = category_glove_embedding_dict[index_to_category_dict[i]]["glove_embed"].unsqueeze(0)
                temp_mask_matrix = category_glove_embedding_dict[index_to_category_dict[i]]["mask_matrix"].unsqueeze(0) 

                glove_embed = torch.cat((glove_embed, temp_glove_embed), dim=0) 
                mask_matrix = torch.cat((mask_matrix, temp_mask_matrix), dim=0) 

        return glove_embed.cuda(), mask_matrix.cuda()
