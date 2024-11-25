CUDA_VISIBLE_DEVICES=0 python tools/main_EmDepart.py --model_name AWA2 --dataset AWA2 --data_document_root data/document/AWA2.json \
    --data_root data/Animals_with_Attributes2 --data_split_root data/data_split \
    --max_context_length 500 -b 64 --epochs 100 --max_epochs 32  --num_image_projection 2 --grad_clip 1 \
    --perceiver_image_dim_head 256 --perceiver_text_dim_head 256 --d_middle_space 256  --fixed_temperature_flag --similarity_temperature 32 \
    --image_mlp_res_flag  --set_based_embedding_model_flag --perceiver_softmax_mode default --similarity_func_mode smooth_chamfer_distance  \
    --calibrated_stacking 0.05 --cs_list_flag --dropout_rate 0.35 --learning_rate 0.0001 \
    --lambda_variance 5  --lambda_diversity 3 --lambda_CLS 0.9 --lambda_local 0.1 --variance_constant 0.1 