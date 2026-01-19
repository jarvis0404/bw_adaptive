"""
Multi-Model Comparison Script

This script reads JSON results from multiple compare_traditional.py runs and
generates comparison plots across different models.

Usage:
    python compare_models.py -json_paths results1/results.json results2/results.json -output_dir comparison

Or use relative paths from a base directory:
    python compare_models.py -base_dir comparison_results -json_paths model1_*/results.json model2_*/results.json
"""

import json
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime


def load_json_results(json_path):
    """Load results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_all_json_paths(base_dir, patterns):
    """Get all JSON file paths matching patterns"""
    all_paths = []
    for pattern in patterns:
        if base_dir:
            pattern = os.path.join(base_dir, pattern)
        matched = glob.glob(pattern)
        all_paths.extend(matched)
    return sorted(set(all_paths))  # Remove duplicates


def extract_model_label(json_data, json_path):
    """Extract a readable label for the model"""
    metadata = json_data['metadata']
    model_name = metadata.get('model_name', os.path.basename(os.path.dirname(json_path)))
    
    # Try to extract key parameters
    parts = []
    if 'channel_mode' in metadata:
        parts.append(metadata['channel_mode'])
    if 'link_qual' in metadata:
        parts.append(f"SNR{metadata['link_qual']}")
    
    if parts:
        label = f"{model_name} ({', '.join(parts)})"
    else:
        label = model_name
    
    return label


def plot_multi_model_snr_comparison(models_data, bw, output_dir):
    """Plot performance comparison across models and traditional methods for fixed bandwidth vs SNR"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['psnr', 'ssim', 'lpips']
    titles = ['PSNR (dB)', 'SSIM', 'LPIPS']
    
    num_models = len(models_data)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_models)))
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Track if we've plotted WebP and BPG
        webp_plotted = False
        bpg_plotted = False
        
        for model_idx, (label, data) in enumerate(models_data.items()):
            results = data['results']
            bw_str = str(bw)
            color = colors[model_idx]
            
            # Plot model results
            if 'model' in results and bw_str in results['model']:
                snrs = sorted([float(s) for s in results['model'][bw_str].keys()])
                values = [results['model'][bw_str][str(s)][metric] for s in snrs]
                ax.plot(snrs, values, marker='o', linestyle='-', 
                       label=label, linewidth=2, color=color)
            
            # Plot WebP results only once
            if not webp_plotted and 'webp' in results and bw_str in results['webp']:
                snrs = sorted([float(s) for s in results['webp'][bw_str].keys()])
                values = [results['webp'][bw_str][str(s)][metric] for s in snrs]
                ax.plot(snrs, values, marker='^', linestyle='--', 
                       label='WebP', linewidth=2, color='gray', alpha=0.8)
                webp_plotted = True
            
            # Plot BPG results only once
            if not bpg_plotted and 'bpg' in results and bw_str in results['bpg']:
                snrs = sorted([float(s) for s in results['bpg'][bw_str].keys()])
                values = [results['bpg'][bw_str][str(s)][metric] for s in snrs]
                ax.plot(snrs, values, marker='s', linestyle=':', 
                       label='BPG', linewidth=2, color='black', alpha=0.8)
                bpg_plotted = True
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs SNR (BW={bw})', fontsize=14)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        if metric == 'lpips':
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_snr_bw{bw}.png'), dpi=150)
    plt.close()


def plot_multi_model_bw_comparison(models_data, snr, output_dir):
    """Plot performance comparison across models and traditional methods for fixed SNR vs bandwidth"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['psnr', 'ssim', 'lpips']
    titles = ['PSNR (dB)', 'SSIM', 'LPIPS']
    
    num_models = len(models_data)
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, num_models)))
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Track if we've plotted WebP and BPG
        webp_plotted = False
        bpg_plotted = False
        
        for model_idx, (label, data) in enumerate(models_data.items()):
            results = data['results']
            snr_str = str(snr)
            color = colors[model_idx]
            
            # Plot model results
            if 'model' in results:
                bws = sorted([int(b) for b in results['model'].keys()])
                values = []
                valid_bws = []
                
                for bw in bws:
                    bw_str = str(bw)
                    if snr_str in results['model'][bw_str]:
                        values.append(results['model'][bw_str][snr_str][metric])
                        valid_bws.append(bw)
                
                if values:
                    ax.plot(valid_bws, values, marker='o', linestyle='-',
                           label=label, linewidth=2, color=color)
            
            # Plot WebP results only once
            if not webp_plotted and 'webp' in results:
                bws = sorted([int(b) for b in results['webp'].keys()])
                values = []
                valid_bws = []
                
                for bw in bws:
                    bw_str = str(bw)
                    if bw_str in results['webp'] and snr_str in results['webp'][bw_str]:
                        values.append(results['webp'][bw_str][snr_str][metric])
                        valid_bws.append(bw)
                
                if values:
                    ax.plot(valid_bws, values, marker='^', linestyle='--',
                           label='WebP', linewidth=2, color='gray', alpha=0.8)
                    webp_plotted = True
            
            # Plot BPG results only once
            if not bpg_plotted and 'bpg' in results:
                bws = sorted([int(b) for b in results['bpg'].keys()])
                values = []
                valid_bws = []
                
                for bw in bws:
                    bw_str = str(bw)
                    if bw_str in results['bpg'] and snr_str in results['bpg'][bw_str]:
                        values.append(results['bpg'][bw_str][snr_str][metric])
                        valid_bws.append(bw)
                
                if values:
                    ax.plot(valid_bws, values, marker='s', linestyle=':',
                           label='BPG', linewidth=2, color='black', alpha=0.8)
                    bpg_plotted = True
        
        ax.set_xlabel('Bandwidth (BW)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs Bandwidth (SNR={int(snr)}dB)', fontsize=14)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        if metric == 'lpips':
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_bw_snr{int(snr)}.png'), dpi=150)
    plt.close()


def plot_method_comparison(models_data, method, bw, output_dir):
    """Plot comparison of a specific method (webp/bpg) across models"""
    if method not in ['webp', 'bpg']:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['psnr', 'ssim', 'lpips']
    titles = ['PSNR (dB)', 'SSIM', 'LPIPS']
    colors = plt.cm.tab10(np.linspace(0, 1, len(models_data)))
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        for model_idx, (label, data) in enumerate(models_data.items()):
            results = data['results']
            
            if method not in results:
                continue
            
            bw_str = str(bw)
            if bw_str not in results[method]:
                continue
            
            snrs = sorted([float(s) for s in results[method][bw_str].keys()])
            values = [results[method][bw_str][str(s)][metric] for s in snrs]
            
            ax.plot(snrs, values, marker='^', linestyle='--',
                   label=label, linewidth=2, color=colors[model_idx])
        
        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} vs SNR ({method.upper()}, BW={bw})', fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if metric == 'lpips':
            ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'multi_model_{method}_snr_bw{bw}.png'), dpi=150)
    plt.close()


def create_summary_table(models_data, output_dir):
    """Create a summary table comparing all models"""
    summary = []
    
    for label, data in models_data.items():
        metadata = data['metadata']
        results = data['results']
        
        # Find middle SNR and BW values
        if 'model' in results and results['model']:
            bws = sorted([int(b) for b in results['model'].keys()])
            mid_bw = bws[len(bws) // 2]
            
            snrs = sorted([float(s) for s in results['model'][str(mid_bw)].keys()])
            mid_snr = snrs[len(snrs) // 2]
            
            metrics = results['model'][str(mid_bw)][str(mid_snr)]
            
            summary.append({
                'Model': label,
                'Channel': metadata.get('channel_mode', 'N/A'),
                'Link Quality': metadata.get('link_qual', 'N/A'),
                'SNR': int(mid_snr),
                'BW': mid_bw,
                'PSNR': f"{metrics['psnr']:.2f}",
                'SSIM': f"{metrics['ssim']:.4f}",
                'LPIPS': f"{metrics['lpips']:.4f}",
            })
    
    # Write to text file
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write('=' * 100 + '\n')
        f.write('MULTI-MODEL COMPARISON SUMMARY\n')
        f.write('=' * 100 + '\n\n')
        
        # Header
        f.write(f"{'Model':<40} {'Channel':<10} {'LQ':<8} {'SNR':<6} {'BW':<4} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8}\n")
        f.write('-' * 100 + '\n')
        
        # Data rows
        for row in summary:
            f.write(f"{row['Model']:<40} {row['Channel']:<10} {str(row['Link Quality']):<8} "
                   f"{row['SNR']:<6} {row['BW']:<4} {row['PSNR']:<8} {row['SSIM']:<8} {row['LPIPS']:<8}\n")
        
        f.write('\n' + '=' * 100 + '\n')
    
    print(f'Summary table saved to {summary_path}')


def main():
    parser = argparse.ArgumentParser(description='Compare multiple models from JSON results')
    
    parser.add_argument('-json_paths', type=str, nargs='+', required=True,
                        help='List of JSON result file paths or glob patterns')
    parser.add_argument('-base_dir', type=str, default=None,
                        help='Base directory for relative paths (optional)')
    parser.add_argument('-output_dir', type=str, default='multi_model_comparison',
                        help='Output directory for comparison plots')
    parser.add_argument('-labels', type=str, nargs='+', default=None,
                        help='Custom labels for models (optional, must match number of JSON files)')
    
    args = parser.parse_args()
    
    # Get all JSON file paths
    json_paths = get_all_json_paths(args.base_dir, args.json_paths)
    
    if not json_paths:
        print(f'Error: No JSON files found matching patterns: {args.json_paths}')
        if args.base_dir:
            print(f'  Base directory: {args.base_dir}')
        return
    
    print(f'Found {len(json_paths)} JSON files:')
    for path in json_paths:
        print(f'  - {path}')
    
    # Load all results
    models_data = {}
    for idx, json_path in enumerate(json_paths):
        print(f'\nLoading {json_path}...')
        data = load_json_results(json_path)
        
        # Determine label
        if args.labels and idx < len(args.labels):
            label = args.labels[idx]
        else:
            label = extract_model_label(data, json_path)
        
        models_data[label] = data
        print(f'  Label: {label}')
        print(f'  Dataset: {data["metadata"].get("dataset", "N/A")}')
        print(f'  Channel: {data["metadata"].get("channel_mode", "N/A")}')
        print(f'  Samples: {data["metadata"].get("num_samples", "N/A")}')
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f'\nOutput directory: {output_dir}')
    
    # Find common SNR and BW ranges
    all_snrs = set()
    all_bws = set()
    
    for label, data in models_data.items():
        results = data['results']
        if 'model' in results:
            for bw_str, snr_dict in results['model'].items():
                all_bws.add(int(bw_str))
                for snr_str in snr_dict.keys():
                    all_snrs.add(float(snr_str))
    
    snrs = sorted(all_snrs)
    bws = sorted(all_bws)
    
    print(f'\nCommon SNR values: {snrs}')
    print(f'Common BW values: {bws}')
    
    # Generate plots
    print('\nGenerating comparison plots...')
    
    # SNR comparison for each BW (models + traditional methods)
    print('  - SNR comparison plots (models + traditional methods)...')
    for bw in bws:
        plot_multi_model_snr_comparison(models_data, bw, output_dir)
    
    # BW comparison for each SNR (models + traditional methods)
    print('  - BW comparison plots (models + traditional methods)...')
    for snr in snrs:
        plot_multi_model_bw_comparison(models_data, snr, output_dir)
    
    # Create summary table
    print('  - Summary table...')
    create_summary_table(models_data, output_dir)
    
    print(f'\n✓ All comparison plots and summary saved to {output_dir}')


if __name__ == '__main__':
    main()
