tensor board
```bash
tensorboard --logdir=runs/20260118
```

训练模型
```bash
python run_swin_adapt.py \
    -model_name encoder_modified_2_0 \
    -epoch 300 \
    -channel_mode awgn \
    -link_qual 7.0 \
    -lpips_lambda 1.0 \
    -device cuda:3 \
    -use_encoder_pruning True \
    -use_decoder_early_exit False
```

python compare_traditional.py \
    -model_path models/JSCC_swin_adapt_lr_awgn_epoch_600_dataset_cifar10_link_qual_10.0_n_trans_feat_16_hidden_size_256_n_heads_8_n_layers_8_is_adapt_True_link_rng_3.0_min_trans_feat_1_max_trans_feat_6_unit_trans_feat_4_trg_trans_feat_6.pth \
    -dataset cifar10 \
    -device cuda:3 \
    -snr_min 5 -snr_max 15 -snr_step 2 \
    -bw_min 1 -bw_max 6 \
    -output_dir results \
    -use_encoder_pruning false \
    -use_decoder_early_exit false \
    -max_samples 5