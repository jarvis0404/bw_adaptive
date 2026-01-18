"""
Comparison Script: Semantic Communication Model vs. Traditional Image Compression (BPG, WebP)

This script compares the DeepJSCC semantic communication model with traditional 
image compression algorithms (BPG, WebP) using 5G channel coding (LDPC).

Key features:
1. Fair comparison by matching transmission bits
2. Traditional algorithms use 5G LDPC channel coding
3. Metrics: PSNR, SSIM, LPIPS
4. Output comparison plots for different SNR and bandwidth settings
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import os
import subprocess
import tempfile
import cv2
import lpips
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
import argparse
from datetime import datetime
from types import SimpleNamespace
from shutil import which

# Import from project modules
from swin_module_bw import Swin_Encoder, Swin_Decoder, Swin_JSCC
from patch_modules import ComputationAdaptiveJSCC
from dataset import CIFAR10, Kodak
from utils import as_img_array, calc_psnr, calc_ssim, load_weights, np_to_torch


##############################################################################
# Helper Functions (from evaluate.py)
##############################################################################

def args_from_weights(weights_path: str, use_encoder_pruning: bool = None, 
                       use_decoder_early_exit: bool = None) -> SimpleNamespace:
    """Parse hyperparameters from weight filename"""
    name = os.path.splitext(os.path.basename(weights_path))[0]
    tokens = name.split('_')

    defaults = {
        'dataset': 'cifar',
        'channel_mode': 'awgn',
        'image_dims': [32, 32],
        'depth': [2, 4],
        'embed_size': 256,
        'window_size': 8,
        'mlp_ratio': 4,
        'n_trans_feat': 16,
        'n_heads': 8,
        'hidden_size': 256,
        'n_layers': 8,
        'adapt': True,
        'link_qual': 7.0,
        'link_rng': 3.0,
        'min_trans_feat': 1,
        'max_trans_feat': 6,
        'unit_trans_feat': 4,
        'n_adapt_embed': 2,
        'use_encoder_pruning': False,  # Default to False for backward compatibility
        'use_decoder_early_exit': False,
    }

    parsed = {}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == 'dataset' and i + 1 < len(tokens):
            parsed['dataset'] = tokens[i + 1]; i += 2; continue
        if t == 'awgn':
            parsed['channel_mode'] = 'awgn'; i += 1; continue
        if t == 'rayleigh':
            parsed['channel_mode'] = 'rayleigh'; i += 1; continue
        if t == 'channel' and i + 1 < len(tokens):
            parsed['channel_mode'] = tokens[i + 1]; i += 2; continue
        if t == 'link' and i + 2 < len(tokens) and tokens[i + 1] == 'qual':
            parsed['link_qual'] = float(tokens[i + 2]); i += 3; continue
        if t == 'link' and i + 2 < len(tokens) and tokens[i + 1] == 'rng':
            parsed['link_rng'] = float(tokens[i + 2]); i += 3; continue
        if t == 'n' and i + 3 < len(tokens) and tokens[i + 1] == 'trans' and tokens[i + 2] == 'feat':
            parsed['n_trans_feat'] = int(tokens[i + 3]); i += 4; continue
        if t == 'hidden' and i + 2 < len(tokens) and tokens[i + 1] == 'size':
            parsed['hidden_size'] = int(tokens[i + 2]); i += 3; continue
        if t == 'n' and i + 2 < len(tokens) and tokens[i + 1] == 'heads':
            parsed['n_heads'] = int(tokens[i + 2]); i += 3; continue
        if t == 'n' and i + 2 < len(tokens) and tokens[i + 1] == 'layers':
            parsed['n_layers'] = int(tokens[i + 2]); i += 3; continue
        if t == 'is' and i + 2 < len(tokens) and tokens[i + 1] == 'adapt':
            parsed['adapt'] = tokens[i + 2] == 'True'; i += 3; continue
        if t == 'min' and i + 3 < len(tokens) and tokens[i + 1] == 'trans' and tokens[i + 2] == 'feat':
            parsed['min_trans_feat'] = int(tokens[i + 3]); i += 4; continue
        if t == 'max' and i + 3 < len(tokens) and tokens[i + 1] == 'trans' and tokens[i + 2] == 'feat':
            parsed['max_trans_feat'] = int(tokens[i + 3]); i += 4; continue
        if t == 'unit' and i + 3 < len(tokens) and tokens[i + 1] == 'trans' and tokens[i + 2] == 'feat':
            parsed['unit_trans_feat'] = int(tokens[i + 3]); i += 4; continue
        i += 1

    cfg = {**defaults, **parsed, 'device': None}
    
    # Override with explicit arguments if provided
    if use_encoder_pruning is not None:
        cfg['use_encoder_pruning'] = use_encoder_pruning
    if use_decoder_early_exit is not None:
        cfg['use_decoder_early_exit'] = use_decoder_early_exit
    
    return SimpleNamespace(**cfg)


def build_model(device: torch.device, weights_path: str, 
                use_encoder_pruning: bool = None, use_decoder_early_exit: bool = None):
    """
    Build and load model from weights path.
    
    Supports both Swin_JSCC and ComputationAdaptiveJSCC models.
    If use_encoder_pruning or use_decoder_early_exit is True, uses ComputationAdaptiveJSCC.
    """
    base_args = args_from_weights(weights_path, use_encoder_pruning, use_decoder_early_exit)
    base_args.device = device

    enc_kwargs = dict(
        args=base_args,
        n_trans_feat=base_args.n_trans_feat,
        img_size=(base_args.image_dims[0], base_args.image_dims[1]),
        embed_dims=[base_args.embed_size, base_args.embed_size],
        depths=[base_args.depth[0], base_args.depth[1]],
        num_heads=[base_args.n_heads, base_args.n_heads],
        window_size=base_args.window_size,
        mlp_ratio=base_args.mlp_ratio,
        qkv_bias=True,
        qk_scale=None,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )

    dec_kwargs = dict(
        args=base_args,
        n_trans_feat=base_args.n_trans_feat,
        img_size=(base_args.image_dims[0], base_args.image_dims[1]),
        embed_dims=[base_args.embed_size, base_args.embed_size],
        depths=[base_args.depth[1], base_args.depth[0]],
        num_heads=[base_args.n_heads, base_args.n_heads],
        window_size=base_args.window_size,
        mlp_ratio=base_args.mlp_ratio,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
    )

    encoder = Swin_Encoder(**enc_kwargs).to(device)
    decoder = Swin_Decoder(**dec_kwargs).to(device)
    
    # Determine which model type to use
    use_adaptive = base_args.use_encoder_pruning or base_args.use_decoder_early_exit
    
    if use_adaptive:
        # Use ComputationAdaptiveJSCC
        bottleneck_channels = encoder.max_trans_feat * encoder.unit_trans_feat
        bottleneck_spatial = (encoder.H, encoder.W)
        model = ComputationAdaptiveJSCC(
            base_args, encoder, decoder, bottleneck_channels, bottleneck_spatial,
            img_size=(base_args.image_dims[0], base_args.image_dims[1]),
            use_encoder_pruning=base_args.use_encoder_pruning,
            use_decoder_early_exit=base_args.use_decoder_early_exit
        ).to(device)
        model_type = 'ComputationAdaptiveJSCC'
    else:
        # Use standard Swin_JSCC
        model = Swin_JSCC(base_args, encoder, decoder).to(device)
        model_type = 'Swin_JSCC'
    
    # Load weights
    cp = torch.load(weights_path, map_location=device)
    model.load_state_dict(cp['jscc_model'], strict=False)
    
    model.eval()
    print(f'  Model type: {model_type}')
    print(f'  use_encoder_pruning: {base_args.use_encoder_pruning}')
    print(f'  use_decoder_early_exit: {base_args.use_decoder_early_exit}')
    
    return model, base_args


def tensor_to_bgr_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to BGR uint8 numpy array"""
    img = as_img_array(t).byte().cpu().numpy()  # C x H x W
    return np.transpose(img, (1, 2, 0))


def compute_metrics(pred: torch.Tensor, target: torch.Tensor):
    """Compute PSNR and SSIM between prediction and target"""
    pred_b = pred.unsqueeze(0)
    target_b = target.unsqueeze(0)
    psnr_val = calc_psnr([pred_b], [target_b])[0].item()
    ssim_val = calc_ssim([pred_b], [target_b])[0].item()
    return psnr_val, ssim_val


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, lpips_fn, device):
    """Compute LPIPS between prediction and target"""
    pred_b = pred.unsqueeze(0).to(device)
    target_b = target.unsqueeze(0).to(device)
    # LPIPS expects [-1, 1] range
    pred_norm = pred_b * 2.0 - 1.0
    target_norm = target_b * 2.0 - 1.0
    with torch.no_grad():
        lpips_val = lpips_fn(pred_norm, target_norm).item()
    return lpips_val


##############################################################################
# 5G LDPC / MCS Table (from evaluate.py)
##############################################################################

MCS_TABLE = [
    ("BPSK",  1, 1/2, -3.0),
    ("BPSK",  1, 2/3, -1.5),
    ("BPSK",  1, 4/5, -0.5),
    ("QPSK",  2, 1/2,  1.0),
    ("QPSK",  2, 2/3,  3.0),
    ("QPSK",  2, 3/4,  4.5),
    ("QPSK",  2, 5/6,  6.0),
    ("8PSK",  3, 2/3,  7.5),
    ("8PSK",  3, 3/4,  9.0),
    ("8PSK",  3, 5/6, 10.5),
    ("16QAM", 4, 1/2,  9.5),
    ("16QAM", 4, 2/3, 11.5),
    ("16QAM", 4, 3/4, 13.0),
    ("16QAM", 4, 5/6, 14.5),
]


def get_best_mcs(snr_db):
    """Get best MCS for given SNR"""
    best_mcs = None
    best_efficiency = -1.0
    for name, mod_bits, rate, min_snr in MCS_TABLE:
        if snr_db >= min_snr:
            efficiency = mod_bits * rate
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_mcs = (name, mod_bits, rate, min_snr)
    if best_mcs is None:
        return ("Outage", 0, 0, 0)
    return best_mcs


def get_ldpc_throughput_bytes(model, height, width, bw, snr_db):
    """Calculate equivalent bytes for traditional compression"""
    # Handle both Swin_JSCC (model.enc) and ComputationAdaptiveJSCC (model.encoder)
    encoder = getattr(model, 'enc', None) or getattr(model, 'encoder', None)
    if encoder is None:
        raise AttributeError("Model has neither 'enc' nor 'encoder' attribute")
    
    downsample_factor = 2 ** encoder.num_layers 
    n_patches_h = height // downsample_factor
    n_patches_w = width // downsample_factor
    n_patches = n_patches_h * n_patches_w
    unit_trans_feat = encoder.unit_trans_feat
    active_real_symbols = n_patches * bw * unit_trans_feat
    complex_channel_uses = active_real_symbols / 2.0
    
    mcs_name, mod_bits, code_rate, min_snr = get_best_mcs(snr_db)
    if code_rate == 0:
        return 0, "Outage"
        
    total_bits = complex_channel_uses * mod_bits * code_rate
    return total_bits / 8.0, f"{mcs_name} R={code_rate:.2f}"


##############################################################################
# Traditional Image Compression (BPG, WebP)
##############################################################################

def check_bpg_available():
    """Check if BPG tools are available"""
    return which('bpgenc') is not None and which('bpgdec') is not None


def check_webp_available():
    """Check if WebP support is available (via cv2)"""
    # cv2 usually has webp support built-in
    return True


def webp_compress_target_size(img_bgr, target_bytes):
    """Compress image with WebP to target byte size using binary search"""
    if target_bytes <= 0:
        return img_bgr, 0
    
    l, r = 1, 100
    best_res = None
    min_diff = float('inf')
    
    while l <= r:
        mid = (l + r) // 2
        q = max(1, mid)
        _, enc = cv2.imencode('.webp', img_bgr, [cv2.IMWRITE_WEBP_QUALITY, q])
        size = len(enc)
        diff = abs(size - target_bytes)
        
        if diff < min_diff:
            min_diff = diff
            best_res = (q, enc, size)
        
        if size < target_bytes:
            l = mid + 1
        elif size > target_bytes:
            r = mid - 1
        else:
            break
    
    if best_res is None:
        return img_bgr, 0
    
    q, enc, size = best_res
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return decoded, size


def bpg_compress_target_size(img_bgr, target_bytes):
    """Compress image with BPG to target byte size using binary search"""
    if target_bytes <= 0:
        return None, 0
    
    fd_in, temp_in = tempfile.mkstemp(suffix='.png')
    os.close(fd_in)
    fd_out, temp_out = tempfile.mkstemp(suffix='.bpg')
    os.close(fd_out)
    fd_dec, temp_dec = tempfile.mkstemp(suffix='.png')
    os.close(fd_dec)
    
    cv2.imwrite(temp_in, img_bgr)
    
    l, r = 0, 51
    best_res = None
    min_diff = float('inf')
    
    while l <= r:
        qp = (l + r) // 2
        cmd = ['bpgenc', '-q', str(qp), '-o', temp_out, temp_in]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            size = os.path.getsize(temp_out)
        except:
            size = 0
            
        diff = abs(size - target_bytes)
        if size > 0 and diff < min_diff:
            min_diff = diff
            try:
                subprocess.run(['bpgdec', '-o', temp_dec, temp_out], 
                             check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                decoded = cv2.imread(temp_dec)
                if decoded is not None:
                    best_res = (decoded, size)
            except:
                pass
        
        if size == 0:
            break
        if size > target_bytes:
            l = qp + 1
        elif size < target_bytes:
            r = qp - 1
        else:
            break
            
    # Cleanup
    if os.path.exists(temp_in):
        os.remove(temp_in)
    if os.path.exists(temp_out):
        os.remove(temp_out)
    if os.path.exists(temp_dec):
        os.remove(temp_dec)
    
    if best_res:
        return best_res
    return None, 0


def jpeg_compress_target_size(img_bgr, target_bytes):
    """Compress image with JPEG to target byte size using binary search"""
    if target_bytes <= 0:
        return img_bgr, 0
    
    l, r = 1, 100
    best_res = None
    min_diff = float('inf')
    
    while l <= r:
        mid = (l + r) // 2
        q = max(1, mid)
        _, enc = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        size = len(enc)
        diff = abs(size - target_bytes)
        
        if diff < min_diff:
            min_diff = diff
            best_res = (q, enc, size)
        
        if size < target_bytes:
            l = mid + 1
        elif size > target_bytes:
            r = mid - 1
        else:
            break
    
    if best_res is None:
        return img_bgr, 0
    
    q, enc, size = best_res
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return decoded, size


##############################################################################
# Plotting Functions
##############################################################################

def plot_snr_comparison(results, bw, output_dir):
    """Plot performance comparison across SNR values for a fixed bandwidth"""
    snrs = sorted(results['model'][bw].keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['psnr', 'ssim', 'lpips']
    titles = ['PSNR (dB)', 'SSIM', 'LPIPS']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        for method, method_results in results.items():
            if bw in method_results:
                values = [method_results[bw][snr][metric] for snr in snrs]
                
                if method == 'model':
                    label = 'Model'
                    marker = 'o'
                    linestyle = '-'
                elif method == 'bpg':
                    label = 'BPG + LDPC'
                    marker = 's'
                    linestyle = '--'
                elif method == 'webp':
                    label = 'WebP + LDPC'
                    marker = '^'
                    linestyle = '-.'
                else:
                    label = method.upper()
                    marker = 'x'
                    linestyle = ':'
                
                ax.plot(snrs, values, marker=marker, linestyle=linestyle, label=label, linewidth=2)
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs SNR (BW={bw})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if metric == 'lpips':
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_snr_bw{bw}.png'), dpi=150)
    plt.close()


def plot_bw_comparison(results, snr, output_dir):
    """Plot performance comparison across bandwidth values for a fixed SNR"""
    bws = sorted(results['model'].keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['psnr', 'ssim', 'lpips']
    titles = ['PSNR (dB)', 'SSIM', 'LPIPS']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        for method, method_results in results.items():
            values = []
            valid_bws = []
            
            for bw in bws:
                if bw in method_results and snr in method_results[bw]:
                    values.append(method_results[bw][snr][metric])
                    valid_bws.append(bw)
            
            if values:
                if method == 'model':
                    label = 'DeepJSCC (Ours)'
                    marker = 'o'
                    linestyle = '-'
                elif method == 'bpg':
                    label = 'BPG + LDPC'
                    marker = 's'
                    linestyle = '--'
                elif method == 'webp':
                    label = 'WebP + LDPC'
                    marker = '^'
                    linestyle = '-.'
                else:
                    label = method.upper()
                    marker = 'x'
                    linestyle = ':'
                
                ax.plot(valid_bws, values, marker=marker, linestyle=linestyle, label=label, linewidth=2)
        
        ax.set_xlabel('Bandwidth (BW)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs Bandwidth (SNR={snr}dB)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if metric == 'lpips':
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_bw_snr{int(snr)}.png'), dpi=150)
    plt.close()


##############################################################################
# Main Comparison Script
##############################################################################

def main():
    parser = argparse.ArgumentParser(description='Compare DeepJSCC with traditional compression')
    
    parser.add_argument('-model_path', type=str, required=True, help='Path to model checkpoint (.pth file)')
    parser.add_argument('-dataset', type=str, default='cifar10', choices=['cifar10', 'kodak'],
                        help='Dataset (cifar10, kodak)')
    parser.add_argument('-device', type=str, default='cuda:0', help='Device')
    parser.add_argument('-snr_min', type=float, default=0, help='Minimum SNR')
    parser.add_argument('-snr_max', type=float, default=20, help='Maximum SNR')
    parser.add_argument('-snr_step', type=float, default=4, help='SNR step')
    parser.add_argument('-bw_min', type=int, default=1, help='Minimum bandwidth')
    parser.add_argument('-bw_max', type=int, default=6, help='Maximum bandwidth')
    parser.add_argument('-output_dir', type=str, default='comparison_results', help='Output directory')
    parser.add_argument('-batch_size', type=int, default=1, help='Batch size (recommend 1 for fair comparison)')
    parser.add_argument('-lpips_net', type=str, default='alex', choices=['alex', 'vgg'], help='LPIPS network')
    parser.add_argument('-num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('-max_samples', type=int, default=None, help='Maximum number of samples to evaluate (None = all)')
    
    # Computation-adaptive module switches (must match training settings)
    parser.add_argument('-use_encoder_pruning', type=lambda x: x.lower() in ['true', '1', 'yes'], 
                        default=False, help='Use encoder spatial pruning (must match training)')
    parser.add_argument('-use_decoder_early_exit', type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default=False, help='Use decoder early exit (must match training)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    output_dir = os.path.join(args.output_dir, f'{model_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'Model path: {args.model_path}')
    print(f'Output directory: {output_dir}')
    print(f'Device: {device}')
    
    # Check available compression methods
    bpg_available = check_bpg_available()
    print(f'BPG available: {bpg_available}')
    
    # Load model
    print(f'\nLoading model from {args.model_path}...')
    model, base_args = build_model(
        device, args.model_path,
        use_encoder_pruning=args.use_encoder_pruning,
        use_decoder_early_exit=args.use_decoder_early_exit
    )
    print(f'Model loaded successfully')
    print(f'  - channel_mode: {base_args.channel_mode}')
    print(f'  - link_qual: {base_args.link_qual}')
    print(f'  - max_trans_feat: {base_args.max_trans_feat}')
    
    # Setup dataset
    print(f'\nLoading dataset: {args.dataset}')
    if args.dataset.lower() in ['cifar10', 'cifar-10']:
        dataset = CIFAR10('datasets/cifar-10-batches-py', 'EVALUATE')
        image_size = (32, 32)
    elif args.dataset.lower() == 'kodak':
        dataset = Kodak('datasets/kodak', args=None)
        image_size = None  # Variable size
    else:
        raise ValueError(f'Unknown dataset: {args.dataset}')
    
    loader = data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    print(f'Dataset loaded: {len(dataset)} samples')
    
    # Initialize LPIPS
    print(f'\nInitializing LPIPS ({args.lpips_net})...')
    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
    lpips_fn.eval()
    
    # Define SNR and bandwidth ranges
    snrs = np.arange(args.snr_min, args.snr_max + 1, args.snr_step).tolist()
    bws = list(range(args.bw_min, min(args.bw_max + 1, base_args.max_trans_feat + 1)))
    
    print(f'\nEvaluation parameters:')
    print(f'  SNR range: {snrs}')
    print(f'  BW range: {bws}')
    print(f'  Max samples: {args.max_samples if args.max_samples else "All"}')
    
    # Results storage
    results = defaultdict(lambda: defaultdict(dict))
    
    # Main evaluation loop
    print('\n' + '=' * 60)
    print('Starting Evaluation')
    print('=' * 60)
    
    for snr in snrs:
        for bw in bws:
            print(f'\n--- SNR={snr}dB, BW={bw} ---')
            
            # Metrics storage
            metrics = {
                'model_psnr': [], 'model_ssim': [], 'model_lpips': [],
                'webp_psnr': [], 'webp_ssim': [], 'webp_lpips': [],
                'bpg_psnr': [], 'bpg_ssim': [], 'bpg_lpips': [],
            }
            
            sample_count = 0
            for batch in tqdm(loader, desc=f'SNR={snr} BW={bw}', leave=False):
                if args.dataset.lower() == 'kodak':
                    images, fns = batch
                else:
                    images, labels = batch
                
                images = images.to(device).float()
                B, C, H, W = images.shape
                
                # Model inference
                with torch.no_grad():
                    outputs = model(images, bw=bw, snr=snr, plr=0.0)
                    # Handle dict output (from ComputationAdaptiveJSCC in some modes)
                    if isinstance(outputs, dict):
                        outputs = outputs['output_final']
                
                # Calculate target bytes for traditional methods
                target_bytes, mcs_desc = get_ldpc_throughput_bytes(model, H, W, bw, snr)
                target_bytes = int(target_bytes)
                
                # Minimum bytes to prevent degenerate compression
                min_bytes = 50 if args.dataset.lower() == 'cifar10' else 500
                target_bytes = max(target_bytes, min_bytes)
                
                # Process each image in batch
                for idx in range(B):
                    # Check if we've reached max_samples limit
                    if args.max_samples is not None and sample_count >= args.max_samples:
                        break
                    sample_count += 1
                    
                    orig_t = images[idx].cpu()
                    recon_t = outputs[idx].cpu()
                    orig_bgr = tensor_to_bgr_uint8(orig_t)
                    
                    # Model metrics
                    m_psnr, m_ssim = compute_metrics(recon_t, orig_t)
                    m_lpips = compute_lpips(recon_t, orig_t, lpips_fn, device)
                    metrics['model_psnr'].append(m_psnr)
                    metrics['model_ssim'].append(m_ssim)
                    metrics['model_lpips'].append(m_lpips)
                    
                    # WebP metrics
                    webp_img, webp_size = webp_compress_target_size(orig_bgr, target_bytes)
                    if webp_img is not None:
                        webp_t = np_to_torch(webp_img).float() / 255.0
                        w_psnr, w_ssim = compute_metrics(webp_t, orig_t)
                        w_lpips = compute_lpips(webp_t, orig_t, lpips_fn, device)
                        metrics['webp_psnr'].append(w_psnr)
                        metrics['webp_ssim'].append(w_ssim)
                        metrics['webp_lpips'].append(w_lpips)
                    
                    # BPG metrics
                    if bpg_available:
                        bpg_img, bpg_size = bpg_compress_target_size(orig_bgr, target_bytes)
                        if bpg_img is not None:
                            bpg_t = np_to_torch(bpg_img).float() / 255.0
                            b_psnr, b_ssim = compute_metrics(bpg_t, orig_t)
                            b_lpips = compute_lpips(bpg_t, orig_t, lpips_fn, device)
                            metrics['bpg_psnr'].append(b_psnr)
                            metrics['bpg_ssim'].append(b_ssim)
                            metrics['bpg_lpips'].append(b_lpips)
                
                # Break outer loop if we've reached max_samples
                if args.max_samples is not None and sample_count >= args.max_samples:
                    break
            
            # Aggregate results
            results['model'][bw][snr] = {
                'psnr': np.mean(metrics['model_psnr']) if metrics['model_psnr'] else 0,
                'ssim': np.mean(metrics['model_ssim']) if metrics['model_ssim'] else 0,
                'lpips': np.mean(metrics['model_lpips']) if metrics['model_lpips'] else 1.0,
            }
            results['webp'][bw][snr] = {
                'psnr': np.mean(metrics['webp_psnr']) if metrics['webp_psnr'] else 0,
                'ssim': np.mean(metrics['webp_ssim']) if metrics['webp_ssim'] else 0,
                'lpips': np.mean(metrics['webp_lpips']) if metrics['webp_lpips'] else 1.0,
            }
            if bpg_available:
                results['bpg'][bw][snr] = {
                    'psnr': np.mean(metrics['bpg_psnr']) if metrics['bpg_psnr'] else 0,
                    'ssim': np.mean(metrics['bpg_ssim']) if metrics['bpg_ssim'] else 0,
                    'lpips': np.mean(metrics['bpg_lpips']) if metrics['bpg_lpips'] else 1.0,
                }
            
            # Print current results
            print(f"  Model:  PSNR={results['model'][bw][snr]['psnr']:.2f}, "
                  f"SSIM={results['model'][bw][snr]['ssim']:.4f}, "
                  f"LPIPS={results['model'][bw][snr]['lpips']:.4f}")
            print(f"  WebP:   PSNR={results['webp'][bw][snr]['psnr']:.2f}, "
                  f"SSIM={results['webp'][bw][snr]['ssim']:.4f}, "
                  f"LPIPS={results['webp'][bw][snr]['lpips']:.4f}")
            if bpg_available:
                print(f"  BPG:    PSNR={results['bpg'][bw][snr]['psnr']:.2f}, "
                      f"SSIM={results['bpg'][bw][snr]['ssim']:.4f}, "
                      f"LPIPS={results['bpg'][bw][snr]['lpips']:.4f}")
    
    # Generate plots
    print('\nGenerating plots...')
    
    # SNR comparison for each BW
    for bw in bws:
        plot_snr_comparison(results, bw, output_dir)
    
    # BW comparison for each SNR
    for snr in snrs:
        plot_bw_comparison(results, snr, output_dir)
    
    print(f'\nAll plots saved to {output_dir}')
    
    # Print summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    
    mid_snr = snrs[len(snrs) // 2]
    mid_bw = bws[len(bws) // 2]
    
    print(f'\nAt SNR={mid_snr}dB, BW={mid_bw}:')
    for method in results.keys():
        if mid_bw in results[method] and mid_snr in results[method][mid_bw]:
            m = results[method][mid_bw][mid_snr]
            name = 'DeepJSCC' if method == 'model' else method.upper()
            print(f'  {name}: PSNR={m["psnr"]:.2f}, SSIM={m["ssim"]:.4f}, LPIPS={m["lpips"]:.4f}')


if __name__ == '__main__':
    main()
