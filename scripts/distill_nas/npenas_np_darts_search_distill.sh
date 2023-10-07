CUDA_VISIBLE_DEVICES=0,1 python train_multiple_gpus_open_domain.py \
    --gpus 2 --algorithm gin_predictor \
    --budget 100 \
    --save_dir ./train_output_npenas/npenas_open_domain_darts_distill/
