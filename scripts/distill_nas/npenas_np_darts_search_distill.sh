CUDA_VISIBLE_DEVICES=0,1,2 python train_multiple_gpus_open_domain.py \
    --gpus 3 --algorithm gin_predictor \
    --budget 100 \
    --save_dir ./train_output_npenas/npenas_open_domain_darts_distill/
