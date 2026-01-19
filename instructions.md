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

# 周日下午启动，增大模型参数量，loss的lambda为1，tmux session = 6
python run_swin_adapt.py \
    -model_name embed512_encoderModified \
    -epoch 200 \
    -channel_mode awgn \
    -link_qual 7.0 \
    -lpips_lambda 1.0 \
    -device cuda:3 \
    -use_encoder_pruning True \
    -use_decoder_early_exit False \
    -embed_size 512

python compare_traditional.py \
    -model_path models/20260118/encoder_modified_only_channel_awgn_epoch_200_link_qual_7.0_lpipsNet_alex.pth \
    -dataset cifar10 \
    -device cuda:3 \
    -snr_min 5 -snr_max 15 -snr_step 2 \
    -bw_min 1 -bw_max 6 \
    -output_dir results \
    -use_encoder_pruning true \
    -use_decoder_early_exit false \
    -max_samples 5 \
    -output_dir results \
    -embed_size 512

python compare_models.py \
  -json_paths results/no_pruning/encoder_modified_only_channel_awgn_epoch_200_link_qual_7.0_lpipsNet_alex_20260119_175310/results.json results/pruning/encoder_modified_only_channel_awgn_epoch_200_link_qual_7.0_lpipsNet_alex_20260119_175447/results.json \
  -labels "no prouning" "pruning" \