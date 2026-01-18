tensor board
```bash
tensorboard --logdir=runs/20260118
```

训练模型
```bash
python run_swin_adapt.py \
    -model_name testLambda2 \
    -epoch 300 \
    -channel_mode awgn \
    -link_qual 7.0 \
    -lpips_lambda 2.0 \
    -device cuda:3 \
    -use_encoder_pruning True \
    -use_decoder_early_exit True
```

python compare_traditional.py \
    -model_path models/20260118/testLambda3_channel_awgn_epoch_200_link_qual_7.0_lpipsNet_alex_lpipsLambda_3_0.pth \
    -dataset cifar10 \
    -device cuda:3 \
    -snr_min 5 -snr_max 15 -snr_step 2 \
    -bw_min 1 -bw_max 6 \
    -output_dir results \
    -use_encoder_pruning true \
    -use_decoder_early_exit true \
    -max_samples 5