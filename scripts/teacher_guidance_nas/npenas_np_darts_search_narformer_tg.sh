CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multiple_gpus_open_domain.py \
    --gpus 4 --algorithm narformer_distill \
    --budget 100 \
    --save_dir ./output/train_output_npenas/npenas_open_domain_darts_narformer_tg/
