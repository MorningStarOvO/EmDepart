CUDA_VISIBLE_DEVICES=0 python tools/main_EmDepart.py --model_name CUB --dataset CUB  --data_document_root data/document/CUB.json \
       --data_root data/CUB2002011 --data_split_root data/data_split \
       --max_context_length 440  --epochs 32 -b 40 --max_epochs 32  --warmup_epochs 2 --pooling_method mean  --num_image_projection 2 --grad_clip 10  \
       --d_middle_space 512 --d_middle_space_text 512 --d_semantic_space 64  --perceiver_image_dim_head 64 --perceiver_text_dim_head 64 \
       --attention_layer_mode mode_i2mv --attention_layer_head 1 --variance_constant 0.25 --smooth_chamfer_scale 1 --similarity_temperature 4.2 \
       --image_mlp_res_flag  --set_based_embedding_model_flag --perceiver_softmax_mode default --similarity_func_mode smooth_chamfer_distance --perceiver_text_num_set 5 --perceiver_image_num_set 5 \
       --calibrated_stacking 0.05 --cs_list_flag --dropout_rate 0.15 --learning_rate 0.00078 \
       --lambda_CLS 1.0 --lambda_local 1.0  --lambda_diversity 3 --lambda_variance 0.96 