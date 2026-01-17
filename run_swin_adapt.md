训练模型指令

```bash
python run_swin_adapt.py \
    -model_name encoder_decoder_modified \
    -epoch 200 \
    -channel_mode awgn \
    -link_qual 7.0 \
    -device cuda:1 \
    -use_encoder_pruning True \
    -use_decoder_early_exit True
```