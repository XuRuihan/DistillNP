CUDA_VISIBLE_DEVICES=0,1,2,3 python train_multiple_gpus_open_domain.py \
    --gpus 4 --algorithm gin_predictor_seg \
    --budget 100 \
    --save_dir ./output/train_output_npenas_voc/npenas_open_domain_darts_seg/
