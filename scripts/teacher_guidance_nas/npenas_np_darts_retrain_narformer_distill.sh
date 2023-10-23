CUDA_VISIBLE_DEVICES=1 python tools_open_domain/train_darts_cifar10_distill.py \
    --seed 1 \
    --model_name d6dbee72a15046507f231bc8fd879ffb3c86b227e1ddd0ff8f7787809cb5621e.pkl \
    --save_dir ./output/train_output_npenas/npenas_open_domain_darts_narformer_tg/
