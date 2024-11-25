# ==================== Import ==================== #
import argparse 

from pprint import pprint

import numpy as np


# ==================== Function ==================== #
def parse_opt():

    parser = argparse.ArgumentParser(description='EmDepart Training !')

    # ----- dataset ----- # 
    parser.add_argument('--dataset', default='AWA2', help='which dataset to train')
    parser.add_argument('--data_root', default='data/Animals_with_Attributes2', help='path to dataset')
    parser.add_argument('--data_split_root', default='data/data_split', help='path to data split')
    parser.add_argument('--data_document_root', default='', help='path to document')
    parser.add_argument('--path_cache', default='vector_cache')
    parser.add_argument('--image_embedding', default='vit-base-patch16-224', choices=['vit-base-patch16-224'])
    parser.add_argument('--semantic_embedding', default='I2DFormer')


    parser.add_argument('--data_image_definition_root', default='data/get_single_image_definition/image_definition_AWA2_API.json', help='path to image definition !')


    parser.add_argument('--get_extracted_features_flag', action='store_true', default=False, help='get extracted features')
    parser.add_argument('--use_extracted_features_flag', action='store_true', default=False, help='use extracted features')
    parser.add_argument('--extracted_features_root', default='data/Animals_with_Attributes2/extracted_features', help='path to extracted features')


    parser.add_argument('--gzsl_flag', action='store_true', default=False, help='enable generalized zero-shot learning')
    parser.add_argument('--only_test_flag', action='store_true', default=False, help='only test mode.')
    parser.add_argument('--get_error_images_test_flag', action='store_true', default=False, help='get error images test mode.')
    parser.add_argument('--validation_flag', action='store_true', default=False, help='enable validation mode')
    parser.add_argument('--debug', dest='debug', action='store_true', help='debug mode')
    parser.add_argument('--constant_lr_flag', action='store_true', default=False, help='constant learning rate')
    parser.add_argument('--train_mode', default='default', choices=['default', 'image_definition', 'default_and_image_definition'])
    

    parser.add_argument('--calibrated_stacking', type=float, default=False,  help='calibrated stacking')
    parser.add_argument('--cs_list_flag', action='store_true', default=False, help='Enable list for calibrated stacking')
    parser.add_argument('--cs_list_mode', type=str, default="after_softmax",  choices=['logits', 'after_softmax'], help='choose cs list mode')
    parser.add_argument('--cs_list_logits', type=list, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],  help='list for calibrated stacking')
    parser.add_argument('--cs_list_after_softmax', type=list, default=[0.05, 0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.96, 0.98, 1.0, 2.0],  help='list for calibrated stacking')
    
    # ----- 设置「I2DFormer 的模型参数」相关 ----- # 
    parser.add_argument('--num_image_projection', default=3, type=int, choices=[2, 3])
    parser.add_argument('--num_token_projection', default=2, type=int, choices=[2])
    parser.add_argument('--num_text_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4) 
    parser.add_argument('--d_text_glove_embedding', type=int, default=300) 
    parser.add_argument('--d_image_transformer', type=int, default=768) 
    parser.add_argument('--d_semantic_space', type=int, default=256) 
    parser.add_argument('--d_middle_space', type=int, default=1024) 
    parser.add_argument('--d_middle_space_text', type=int, default=512) 
    parser.add_argument('--max_context_length', type=int, default=402) 
    parser.add_argument('--pooling_method', default='max', choices=['max', 'mean'])
    parser.add_argument('--positional_embedd_flag', action='store_true', default=False,  help='enable positional embedding')
    parser.add_argument('--image_mlp_res_flag', action='store_true', default=False,  help='enable res for Image MLP')
    parser.add_argument('--text_mlp_res_flag', action='store_true', default=False,  help='enable res for Image MLP')
    parser.add_argument('--only_global_loss_flag', action='store_true', default=False,  help='enable only global loss')
    

    parser.add_argument('--image_definition_mask_flag', action='store_true', default=False,  help='when compute loss, enable image definition mask')
    parser.add_argument('--image_CE_loss_flag', action='store_true', default=False,  help='enable image CE loss')
    parser.add_argument('--lambda_image_definition', type=float, default=0.5) 


    parser.add_argument('--prototype_model_flag', action='store_true', default=False,  help='enable prototype model')
    parser.add_argument('--prototype_model_mode', default='text', choices=['text', 'text_and_image', 'image'])
    parser.add_argument('--n_image_prototypes', type=int, default=8) 
    parser.add_argument('--n_text_prototypes', type=int, default=8) 
    parser.add_argument('--variance_constant', type=float, default=1)
    parser.add_argument('--variance_constant_encoder', type=float, default=1)
    parser.add_argument('--lambda_variance', type=float, default=0.1) 
    parser.add_argument('--lambda_variance_image', type=float, default=0.5) 
    parser.add_argument('--lambda_variance_prototype', type=float, default=0.1) 
    parser.add_argument('--prototype_variance_flag', action='store_true', default=False,  help='enable prototype model')
             
    
    parser.add_argument('--image_enhanced_model_flag', action='store_true', default=False,  help='enable image enhanced text model')
    parser.add_argument('--image_enhanced_mode', default='cross_attention', choices=['cross_attention', 'gate', 'dot_product'])
    parser.add_argument('--image_enhanced_feature_mode', default='global', choices=['global', 'local', 'all']) 
    parser.add_argument('--text_logit_loss_flag', action='store_true', default=False,  help='enable text logit loss')
    parser.add_argument('--lambda_text_logit_loss', type=float, default=0.2) 
    parser.add_argument('--data_select_logit_root', default='/root/autodl-tmp/data/faster_rcnn_object_detect/AWA2/all/faster_rcnn_select_logit.json.npy', help='path to select logit')
    parser.add_argument('--n_cross_attention_feed_forward', type=int, default=0) 

    
    parser.add_argument('--set_based_embedding_model_flag', action='store_true', default=False,  help='enable set based embedding model')
    parser.add_argument('--set_prediction_module_mode', default='linear', choices=['linear', 'attention_aggregation_block'])
    parser.add_argument('--similarity_func_mode', default='smooth_chamfer_distance', choices=['i2dformer', 'avg_distance', 'max_distance', 'chamfer_distance', 'smooth_chamfer_distance'])
    parser.add_argument('--image_projection_mode', default='mlp', choices=['mlp', 'perceiver_attention'])
    parser.add_argument('--lr_scale', type=float, default=0.1)
    parser.add_argument('--similarity_temperature', type=float, default=np.log(1/0.07))
    parser.add_argument('--similarity_temperature_local', type=float, default=np.log(1/0.07))
    parser.add_argument('--fixed_temperature_flag', action='store_true', default=False)
    parser.add_argument('--smooth_chamfer_scale', type=float, default=1)
    
    
    parser.add_argument('--perceiver_softmax_mode', default='default', choices=['default', 'slot'])

    parser.add_argument('--perceiver_text_depth', default=2, type=int, help='perceiver depth')
    parser.add_argument('--perceiver_text_dim_head', default=64, type=int)
    parser.add_argument('--perceiver_text_head', default=8, type=int)
    parser.add_argument('--perceiver_text_num_latents', default=64, type=int)
    parser.add_argument('--perceiver_text_num_set', default=4, type=int)
    parser.add_argument('--perceiver_text_ff_mult', default=4, type=int)

    parser.add_argument('--perceiver_image_depth', default=2, type=int, help='perceiver depth')
    parser.add_argument('--perceiver_image_dim_head', default=64, type=int)
    parser.add_argument('--perceiver_image_head', default=8, type=int)
    parser.add_argument('--perceiver_image_num_latents', default=64, type=int)
    parser.add_argument('--perceiver_image_num_set', default=4, type=int)
    parser.add_argument('--perceiver_image_ff_mult', default=4, type=int)

    
    parser.add_argument('--perceiver_more_drop_flag', action='store_true', default=False)

    
    parser.add_argument('--variance_after_softmax_flag', action='store_true', default=False)

    
    parser.add_argument('--global_fuse_mode', default='default', choices=['default', 'concat', 'concat_and_perceiver', 'without_global'])

    
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--max_epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--optim', default='adam', choices=['adamw', 'adam'])
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8) 
    parser.add_argument('--eta_min', type=float, default=1e-6) 
    parser.add_argument('--warmup_start_lr', type=float, default=0)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)


    
    parser.add_argument('--lambda_CLS', type=float, default=0.9) 
    parser.add_argument('--lambda_local', type=float, default=0.5) 
    parser.add_argument('--set_lambda_local_flag', action='store_true', default=False)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=0)
    parser.add_argument('--lambda_mmd', type=float, default=0) 
    parser.add_argument('--lambda_diversity', type=float, default=0) 
    parser.add_argument('--global_loss_mode', default='contrastive', choices=['contrastive', 'triplet'])

    parser.add_argument('--triplet_margin', type=float, default=0.2)     
    parser.add_argument('--semi_hard_triplet_flag', action='store_true', default=False)

    
    parser.add_argument('--attention_layer_mode', default='default', choices=['default', 'mode_i2mv', 'mode_true_att'])
    parser.add_argument('--attention_layer_head', default=1, type=int)

    
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 6)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')

     
    parser.add_argument('--model_name', default=None, type=str, help='model name')
    parser.add_argument('--checkpoint_path', default='checkpoint/EmDepart', type=str, metavar='PATH',
                        help='path to save output results')
    parser.add_argument('--checkpoint_name', default='', type=str, metavar='PATH',
                        help='save checkpoint name')
    parser.add_argument('--load_checkpoint_path', default=None, type=str, metavar='PATH',
                        help='path to load checkpoint')
    parser.add_argument('--task_name', default="default", type=str, help='task name')


    parser.add_argument('--test_freq', default=100, type=int,
                        metavar='N', help='test frequency (default: 100)')
    
    args = parser.parse_args() 

    if args.cs_list_mode == "after_softmax":
        args.cs_list = args.cs_list_after_softmax
    elif args.cs_list_mode == "logits":
        args.cs_list = args.cs_list_logits


    return args 
