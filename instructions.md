tensor board
```bash
tensorboard --logdir=runs/20260117
```

训练模型
```bash
python run_swin_adapt.py \
    -model_name testLambda3 \
    -epoch 200 \
    -channel_mode awgn \
    -link_qual 7.0 \
    -lpips_lambda 3.0 \
    -device cuda:3 \
    -use_encoder_pruning True \
    -use_decoder_early_exit True
```