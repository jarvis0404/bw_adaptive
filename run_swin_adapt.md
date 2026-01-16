训练模型指令

```bash
python run_swin_adapt.py \
    -dataset cifar10 \
    -train_batch_size 32 \
    -epoch 1 \
    -channel_mode rayleigh \
    -link_qual 7.0 \
    -device cuda:0
```

**Loss Function Update:**
The training now uses a **compound loss** combining MSE and LPIPS (Learned Perceptual Image Patch Similarity):
- **Loss Formula**: `Loss = MSE + λ × LPIPS`
- **λ (lambda)**: Weight for LPIPS component (default: 0.1, hardcoded in `run_swin_adapt.py`)
- **LPIPS Network**: AlexNet-based (can be changed to VGG by modifying `net='alex'` to `net='vgg'`)
- **DWA Integration**: Dynamic Weight Adaptation (DWA) weights are applied to the compound loss, while DWA weight calculation continues to use PSNR based on MSE reconstruction quality
- **Input Normalization**: Images are automatically normalized from [0,1] to [-1,1] for LPIPS computation