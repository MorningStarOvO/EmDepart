CUDA_VISIBLE_DEVICES=0 python tools/main_EmDepart.py --model_name FLO --dataset FLO --data_document_root data/document/FLO.json \
       --data_root data/FLO/jpg --data_split_root data/data_split \
       --max_context_length 680 --epochs 40 --max_epochs 40  -b 48 --pooling_method max --grad_clip 1 --variance_after_softmax_flag \
       --d_middle_space 128 --d_middle_space_text 128  --num_image_projection 3  --d_semantic_space 128 \
       --perceiver_image_dim_head 64 --perceiver_text_dim_head 64  --smooth_chamfer_scale 1.5 \
       --image_mlp_res_flag  --set_based_embedding_model_flag --perceiver_softmax_mode default --similarity_func_mode smooth_chamfer_distance  --similarity_temperature 4  \
       --calibrated_stacking 0.05 --cs_list_flag --dropout_rate 0.12 --learning_rate 0.00048 \
       --lambda_CLS 0.5 --lambda_local 0.5 --lambda_diversity 3 --lambda_variance 1 --variance_constant 0.75  