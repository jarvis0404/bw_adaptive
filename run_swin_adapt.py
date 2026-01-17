import numpy as np 
import torch
import torch.utils.data as data
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import lpips
import os
from datetime import datetime

from get_args import get_args
from swin_module_bw import *
from patch_modules import ComputationAdaptiveJSCC
from dataset import CIFAR10, ImageNet, Kodak
from utils import *

from torch.utils.tensorboard import SummaryWriter

###### Parameter Setting
args = get_args()
# Ensure ImageNet-specific args exist
if not hasattr(args, 'crop'):
    args.crop = 128

# Convert args.device (string) to torch.device and validate availability
requested = None
if hasattr(args, 'device'):
    requested = args.device

if isinstance(requested, str) and requested.startswith('cuda'):
    if torch.cuda.is_available():
        device = torch.device(requested)
    else:
        print(f"Requested device {requested} not available; falling back to CPU.")
        device = torch.device('cpu')
else:
    try:
        device = torch.device(requested) if requested is not None else torch.device('cpu')
    except Exception:
        device = torch.device('cpu')

args.device = device

# Build job_name with user-specified model_name as prefix for easy identification
job_name = args.model_name + '_channel_' + args.channel_mode + '_epoch_' + str(args.epoch) + '_link_qual_' + str(args.link_qual) + '_lpipsNet_' + args.lpips_net

print(args)
print(job_name)

# Create date-based subdirectory for TensorBoard logs (similar to model saving)
date_str = datetime.now().strftime('%Y%m%d')
runs_dir = os.path.join('runs', date_str)
if not os.path.exists(runs_dir):
    print('Creating TensorBoard runs directory: {}'.format(runs_dir))
    os.makedirs(runs_dir, exist_ok=True)

writter = SummaryWriter(os.path.join(runs_dir, job_name))

# Select dataset based on args.dataset
def _build_imagenet_file_list(root_dir):
    # Build list of relative paths under datasets/ for ImageNet
    import glob, os
    train_glob = os.path.join(root_dir, 'ILSVRC2012_img_train', '*', '*.JPEG')
    val_glob = os.path.join(root_dir, 'ILSVRC2012_img_val', '*.JPEG')
    fns = []
    for fn in glob.iglob(train_glob, recursive=True):
        # store path relative to datasets/
        fns.append(os.path.relpath(fn, 'datasets'))
    for fn in glob.iglob(val_glob, recursive=True):
        fns.append(os.path.relpath(fn, 'datasets'))
    return fns

# Collate function to drop None samples from datasets (e.g., small images)
def _safe_collate_fn(args):
    import torch as _torch
    def _fn(batch):
        batch = [b for b in batch if b is not None and b[0] is not None]
        if len(batch) == 0:
            empty = _torch.empty((0, 3, args.crop, args.crop))
            return empty, []
        imgs = [_torch.as_tensor(b[0]) for b in batch]
        labels = [b[1] for b in batch]
        images = _torch.stack(imgs, dim=0)
        return images, labels
    return _fn
if str(args.dataset).lower() in ['cifar10', 'cifar-10']:
    train_set = CIFAR10('datasets/cifar-10-batches-py', 'TRAIN')
    valid_set = CIFAR10('datasets/cifar-10-batches-py', 'VALIDATE')
    eval_set = CIFAR10('datasets/cifar-10-batches-py', 'EVALUATE')
elif str(args.dataset).lower() in ['imagenet', 'image-net']:
    # Expect ImageNet folder structure under datasets/imagenet
    imagenet_root = 'datasets/imagenet'
    imagenet_fns = _build_imagenet_file_list(imagenet_root)
    train_set = ImageNet(imagenet_fns, 'TRAIN', args)
    valid_set = ImageNet(imagenet_fns, 'VALIDATE', args)
    eval_set = ImageNet(imagenet_fns, 'EVALUATE', args)
elif str(args.dataset).lower() in ['kodak']:
    # Kodak is typically for evaluation; still provide splits for consistency
    train_set = Kodak('datasets/kodak', 'TRAIN')
    valid_set = Kodak('datasets/kodak', 'VALIDATE')
    eval_set = Kodak('datasets/kodak', 'EVALUATE')
else:
    raise ValueError(f"Unsupported dataset: {args.dataset}. Choose from CIFAR10, ImageNet, Kodak.")


###### The JSCC Model using Swin Transformer ######
enc_kwargs = dict(
        args = args, n_trans_feat = args.n_trans_feat, img_size=(args.image_dims[0], args.image_dims[1]),
        embed_dims=[args.embed_size, args.embed_size], depths=[args.depth[0], args.depth[1]], num_heads=[args.n_heads, args.n_heads],
        window_size=args.window_size, mlp_ratio=args.mlp_ratio, qkv_bias=True, qk_scale=None,
        norm_layer=nn.LayerNorm, patch_norm=True,
    )

dec_kwargs = dict(
        args = args, n_trans_feat = args.n_trans_feat, img_size=(args.image_dims[0], args.image_dims[1]),
        embed_dims=[args.embed_size, args.embed_size], depths=[args.depth[1], args.depth[0]], num_heads=[args.n_heads, args.n_heads],
        window_size=args.window_size, mlp_ratio=args.mlp_ratio, norm_layer=nn.LayerNorm, patch_norm=True,)

source_enc = Swin_Encoder(**enc_kwargs).to(args.device)
source_dec = Swin_Decoder(**dec_kwargs).to(args.device)

# Wrap Swin encoder/decoder with SNR-adaptive wrapper
bottleneck_channels = source_enc.max_trans_feat * source_enc.unit_trans_feat
bottleneck_spatial = (source_enc.H, source_enc.W)
jscc_model = ComputationAdaptiveJSCC(
    args, source_enc, source_dec, bottleneck_channels, bottleneck_spatial, 
    img_size=(args.image_dims[0], args.image_dims[1]),
    use_encoder_pruning=args.use_encoder_pruning,
    use_decoder_early_exit=args.use_decoder_early_exit
).to(args.device)


# load pre-trained
if args.resume == False:
    pass
else:
    _ = load_weights(job_name, jscc_model)

solver = optim.Adam(jscc_model.parameters(), lr=args.lr)
scheduler = LS.MultiplicativeLR(solver, lr_lambda=lambda x: 0.9)
es = EarlyStopping(mode='min', min_delta=0, patience=args.train_patience)

# Initialize LPIPS loss
lpips_loss_fn = lpips.LPIPS(net=args.lpips_net).to(args.device)
lpips_loss_fn.eval()  # Set to eval mode (no training for LPIPS)
lpips_lambda = 0.5  # Weight for LPIPS loss component

###### Dataloader
train_loader = data.DataLoader(
    dataset=train_set,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=_safe_collate_fn(args)
        )

valid_loader = data.DataLoader(
    dataset=valid_set,
    batch_size=args.val_batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=_safe_collate_fn(args)
        )

eval_loader = data.DataLoader(
    dataset=eval_set,
    batch_size=args.val_batch_size,
    shuffle=True,
    num_workers=2,
    collate_fn=_safe_collate_fn(args)
)

##### TARGET -> PSNR obtained by separate models
TARGET_PSNR = np.array([24.75, 27.85, 30.1526917,	32.01,	33.2777652,	34.55393814])
# TARGET_LPIPS: target LPIPS values (lower is better, typically 0.0-0.5 for good quality)
# These should be set based on your target model performance
# If not available, we'll use a relative approach
TARGET_LPIPS = None  # Will use relative LPIPS improvement if None

def dynamic_weight_adaption(current_psnr, current_lpips=None, psnr_weight=0.6, lpips_weight=0.4):
    """
    Dynamically change the weight based on PSNR and LPIPS combination
    
    Args:
        current_psnr: Current PSNR values for each bandwidth
        current_lpips: Current LPIPS values for each bandwidth (optional)
        psnr_weight: Weight for PSNR component in combined metric (default: 0.6)
        lpips_weight: Weight for LPIPS component in combined metric (default: 0.4)
    """
    # Calculate PSNR delta (PSNR: higher is better)
    delta_psnr = TARGET_PSNR - current_psnr
    
    if current_lpips is not None and len(current_lpips) > 0 and np.any(current_lpips > 0):
        # LPIPS: lower is better, so we need to handle it differently
        # Normalize PSNR delta to [0, 1] range (assuming max delta is around 10)
        normalized_psnr_delta = np.clip(delta_psnr / 10.0, 0, 1)
        
        # For LPIPS, if we have target values, use them; otherwise use relative improvement
        if TARGET_LPIPS is not None:
            delta_lpips = current_lpips - TARGET_LPIPS  # positive means worse (need improvement)
        else:
            # Use relative LPIPS: assume lower is better
            # Use current LPIPS value as "improvement needed" metric
            # Normalize by typical max LPIPS (around 0.5-1.0 for poor quality)
            max_lpips_ref = 0.5  # typical maximum LPIPS for reasonable quality
            delta_lpips = np.clip(current_lpips / max_lpips_ref, 0, 1)  # higher means worse
        
        # Normalize LPIPS delta to [0, 1] range (ensure it's positive, meaning need improvement)
        normalized_lpips_delta = np.clip(delta_lpips, 0, 1)
        
        # Combined delta: weighted combination (both normalized to [0,1], higher = more improvement needed)
        combined_delta = psnr_weight * normalized_psnr_delta + lpips_weight * normalized_lpips_delta
    else:
        # Fallback to PSNR only if LPIPS not available
        combined_delta = np.clip(delta_psnr / 10.0, 0, 1)  # normalize
    
    # Calculate weights based on combined delta
    new_weight = np.zeros_like(combined_delta)
    for i in range(len(combined_delta)):
        if combined_delta[i] <= args.threshold / 10.0:  # adjust threshold for normalized delta
            new_weight[i] = 0
        else:
            new_weight[i] = 2**(args.alpha * combined_delta[i] * 10.0)  # scale back for weight calculation

    clipped_weight = np.clip(new_weight, args.min_clip, args.max_clip)
    return clipped_weight


def train_epoch(loader, model, solvers, weight, lpips_fn, lpips_weight):

    model.train()
    lpips_fn.eval()  # Keep LPIPS in eval mode

    # Statistics for magnitude comparison
    mse_values = []
    lpips_values = []
    mse_lpips_ratios = []
    batch_count = 0

    with tqdm(loader, unit='batch') as tepoch:
        for _, (images, _) in enumerate(tepoch):
            
            epoch_postfix = OrderedDict()

            if images is None or (hasattr(images, 'size') and images.size(0) == 0):
                continue
            images = images.to(args.device).float()
            
            solvers.zero_grad()
            bw = np.random.randint(args.min_trans_feat, args.max_trans_feat+1)
            plr = np.random.uniform(args.min_plr, args.max_plr)
            outputs = model(images, bw, plr=plr)

            # If model returns a dict (multi-exit), use model.compute_loss
            if isinstance(outputs, dict) and hasattr(model, 'compute_loss'):
                total_loss, loss_dict = model.compute_loss(outputs, images, lpips_fn=lpips_fn, lpips_weight=lpips_weight)
                loss = total_loss * weight[bw-args.min_trans_feat]
                loss.backward()
                solvers.step()

                epoch_postfix['loss_final'] = '{:.4f}'.format(loss_dict['loss_final'])
                if 'loss_exit1' in loss_dict:
                    epoch_postfix['loss_exit1'] = '{:.4f}'.format(loss_dict['loss_exit1'])
                if 'loss_exit2' in loss_dict:
                    epoch_postfix['loss_exit2'] = '{:.4f}'.format(loss_dict['loss_exit2'])
                epoch_postfix['total_loss'] = '{:.4f}'.format(loss.item())
                
                # Add MSE and LPIPS magnitude comparison info
                if 'mse_final' in loss_dict and 'lpips_final' in loss_dict:
                    mse_val = loss_dict['mse_final']
                    lpips_val = loss_dict['lpips_final']
                    ratio = loss_dict.get('mse_lpips_ratio_final', mse_val / (lpips_val + 1e-8))
                    
                    mse_values.append(mse_val)
                    lpips_values.append(lpips_val)
                    mse_lpips_ratios.append(ratio)
                    
                    # Show detailed comparison every 10 batches
                    if batch_count % 10 == 0:
                        epoch_postfix['mse_raw'] = '{:.6f}'.format(mse_val)
                        epoch_postfix['lpips_raw'] = '{:.6f}'.format(lpips_val)
                        epoch_postfix['mse/lpips'] = '{:.2f}'.format(ratio)
            else:
                # legacy single-output path
                output = outputs
                mse_loss = nn.MSELoss()(output, images)
                output_normalized = output * 2.0 - 1.0
                images_normalized = images * 2.0 - 1.0
                # LPIPS needs gradients for training, so don't use torch.no_grad()
                lpips_loss = lpips_fn(output_normalized, images_normalized).mean()

                compound_loss = mse_loss + lpips_weight * lpips_loss
                loss = compound_loss * weight[bw-args.min_trans_feat]
                loss.backward()
                solvers.step()

                mse_val = mse_loss.item()
                lpips_val = lpips_loss.item()
                ratio = mse_val / (lpips_val + 1e-8)
                
                mse_values.append(mse_val)
                lpips_values.append(lpips_val)
                mse_lpips_ratios.append(ratio)

                epoch_postfix['mse_loss'] = '{:.4f}'.format(mse_val)
                epoch_postfix['lpips_loss'] = '{:.4f}'.format(lpips_val)
                epoch_postfix['total_loss'] = '{:.4f}'.format(loss.item())
                
                # Show detailed comparison every 10 batches
                if batch_count % 10 == 0:
                    epoch_postfix['mse_raw'] = '{:.6f}'.format(mse_val)
                    epoch_postfix['lpips_raw'] = '{:.6f}'.format(lpips_val)
                    epoch_postfix['mse/lpips'] = '{:.2f}'.format(ratio)

            batch_count += 1
            tepoch.set_postfix(**epoch_postfix)
    
    # Print summary statistics at the end of epoch
    if len(mse_values) > 0:
        avg_mse = np.mean(mse_values)
        avg_lpips = np.mean(lpips_values)
        avg_ratio = np.mean(mse_lpips_ratios)
        std_ratio = np.std(mse_lpips_ratios)
        
        print(f"\n[Loss Magnitude Analysis]")
        print(f"  Average MSE: {avg_mse:.6f} (order of magnitude: {10**np.floor(np.log10(avg_mse + 1e-10)):.2e})")
        print(f"  Average LPIPS: {avg_lpips:.6f} (order of magnitude: {10**np.floor(np.log10(avg_lpips + 1e-10)):.2e})")
        print(f"  Average MSE/LPIPS ratio: {avg_ratio:.2f} ± {std_ratio:.2f}")
        if abs(np.log10(avg_mse) - np.log10(avg_lpips)) < 1.0:
            print(f"  ✓ MSE and LPIPS are in the SAME order of magnitude (difference < 1.0)")
        else:
            print(f"  ✗ MSE and LPIPS are in DIFFERENT orders of magnitude (difference >= 1.0)")
            print(f"    Consider adjusting lpips_weight (current: {lpips_weight}) to balance the losses")
        print()

def validate_epoch(loader, model, bw = args.trg_trans_feat, disable = False, plr = None, lpips_fn = None):

    if plr is None:
        plr = args.plr

    model.eval()

    loss_hist = []
    psnr_hist = []
    ssim_hist = []
    lpips_hist = []
    #msssim_hist = []
    power = []

    with torch.no_grad():
        with tqdm(loader, unit='batch', disable = disable) as tepoch:
            for _, (images, _) in enumerate(tepoch):

                epoch_postfix = OrderedDict()

                if images is None or (hasattr(images, 'size') and images.size(0) == 0):
                    continue
                images = images.to(args.device).float()

                output = model(images, bw, snr = args.link_qual, plr = plr)
                
                loss = nn.MSELoss()(output, images)

                epoch_postfix['l2_loss'] = '{:.4f}'.format(loss.item())

                ######  Predictions  ######
                predictions = torch.chunk(output, chunks=output.size(0), dim=0)
                target = torch.chunk(images, chunks=images.size(0), dim=0)

                ######  PSNR/SSIM/etc  ######

                psnr_vals = calc_psnr(predictions, target)
                psnr_hist.extend(psnr_vals)
                epoch_postfix['psnr'] = torch.mean(torch.tensor(psnr_vals)).item()

                ssim_vals = calc_ssim(predictions, target)
                ssim_hist.extend(ssim_vals)
                epoch_postfix['ssim'] = torch.mean(torch.tensor(ssim_vals)).item()
                
                # LPIPS calculation if lpips_fn is provided
                if lpips_fn is not None:
                    output_normalized = output * 2.0 - 1.0
                    images_normalized = images * 2.0 - 1.0
                    lpips_vals = lpips_fn(output_normalized, images_normalized)
                    lpips_hist.extend(lpips_vals.detach().cpu().numpy().flatten().tolist())
                    epoch_postfix['lpips'] = torch.mean(lpips_vals).item()
                
                # Show the snr/loss/psnr/ssim/lpips
                tepoch.set_postfix(**epoch_postfix)

                loss_hist.append(loss.item())
            
            loss_mean = np.nanmean(loss_hist)

            psnr_hist = torch.tensor(psnr_hist)
            psnr_mean = torch.mean(psnr_hist).item()
            psnr_std = torch.sqrt(torch.var(psnr_hist)).item()

            ssim_hist = torch.tensor(ssim_hist)
            ssim_mean = torch.mean(ssim_hist).item()
            ssim_std = torch.sqrt(torch.var(ssim_hist)).item()

            # LPIPS statistics
            if len(lpips_hist) > 0:
                lpips_hist = torch.tensor(lpips_hist)
                lpips_mean = torch.mean(lpips_hist).item()
                lpips_std = torch.sqrt(torch.var(lpips_hist)).item()
            else:
                lpips_mean = None
                lpips_std = None

            predictions = torch.cat(predictions, dim=0)[:, [2, 1, 0]]
            target = torch.cat(target, dim=0)[:, [2, 1, 0]]

            #power = torch.tensor(power)      # (num, n_patches)
            #power = power.view(-1, args.max_trans_feat)
            #power = torch.mean(power, dim=0)

            return_aux = {'psnr': psnr_mean,
                            'ssim': ssim_mean,
                            'predictions': predictions,
                            'target': target,
                            'psnr_std': psnr_std,
                            'ssim_std': ssim_std,
                            'lpips': lpips_mean,
                            'lpips_std': lpips_std}

        
    return loss_mean, return_aux



if __name__ == '__main__':
    epoch = 0

    # initial weights
    weight = np.array([1.0 for _ in range(args.min_trans_feat, args.max_trans_feat+1)])

    while epoch < args.epoch and not args.resume:
        
        epoch += 1
        
        train_epoch(train_loader, jscc_model, solver, weight, lpips_loss_fn, lpips_lambda)

        valid_loss, valid_aux = validate_epoch(valid_loader, jscc_model, lpips_fn=lpips_loss_fn)

        writter.add_scalar('loss', valid_loss, epoch)
        writter.add_scalar('psnr', valid_aux['psnr'], epoch)
        if valid_aux['lpips'] is not None:
            writter.add_scalar('lpips', valid_aux['lpips'], epoch)

        if epoch % args.freq == 0:
            current_psnr = np.array([0.0 for _ in range(args.min_trans_feat, args.max_trans_feat+1)])
            current_lpips = np.array([-1.0 for _ in range(args.min_trans_feat, args.max_trans_feat+1)])  # -1 indicates invalid
            for i in range(len(weight)):
                # Evaluate with default plr (args.plr) for weight adaptation, or should we use 0? 
                # Let's use args.plr to align with target condition.
                _, valid_aux = validate_epoch(valid_loader, jscc_model, args.min_trans_feat + i, True, plr=args.plr, lpips_fn=lpips_loss_fn)   # verbose -> True
                current_psnr[i] = valid_aux['psnr']
                if valid_aux['lpips'] is not None:
                    current_lpips[i] = valid_aux['lpips']
            
            # update the weight using combined PSNR and LPIPS
            # Only use LPIPS if we have valid values for all bandwidths
            use_lpips = np.all(current_lpips >= 0)
            if use_lpips:
                weight = dynamic_weight_adaption(current_psnr, current_lpips, psnr_weight=0.6, lpips_weight=0.4)
            else:
                # Fallback to PSNR only if LPIPS not available
                weight = dynamic_weight_adaption(current_psnr, None, psnr_weight=1.0, lpips_weight=0.0)

            ''' 
            writter.add_scalars('all_psnr', {'bw1':current_psnr[0],'bw2':current_psnr[1],'bw3':current_psnr[2],\
                                'bw4':current_psnr[3]}, epoch)
            writter.add_scalars('weights', {'weight1':weight[0],'weight2':weight[1],'weight3':weight[2],\
                                'weight4':weight[3]}, epoch)
            '''  
            writter.add_scalars('all_psnr', {'bw1':current_psnr[0],'bw2':current_psnr[1],'bw3':current_psnr[2],\
                                'bw4':current_psnr[3], 'bw5':current_psnr[4],'bw6':current_psnr[5]}, epoch)
            if use_lpips:
                writter.add_scalars('all_lpips', {'bw1':current_lpips[0],'bw2':current_lpips[1],'bw3':current_lpips[2],\
                                    'bw4':current_lpips[3], 'bw5':current_lpips[4],'bw6':current_lpips[5]}, epoch)
            writter.add_scalars('weights', {'weight1':weight[0],'weight2':weight[1],'weight3':weight[2],\
                                'weight4':weight[3], 'weight5':weight[4],'weight6':weight[5]}, epoch)     


        flag, best, best_epoch, bad_epochs = es.step(torch.Tensor([valid_loss]), epoch)
        if flag:
            print('ES criterion met; loading best weights from epoch {}'.format(best_epoch))
            _ = load_weights(job_name, jscc_model)
            break
        else:
            # TODO put this in trainer
            if bad_epochs == 0:
                print('average l2_loss: ', valid_loss.item())
                save_nets(job_name, jscc_model, epoch)
                best_epoch = epoch
                print('saving best net weights...')
            elif bad_epochs % 20 == 0:
                scheduler.step()
                print('lr updated: {:.5f}'.format(scheduler.get_last_lr()[0]))



    print('evaluating...')
    print(job_name)
    #jscc_model.sr_link = 0
    ####### adjust the SNR --- fix sd_link = rd_link
    #jscc_model.sr_link = 8
    
    for tgt_trans_feat in range(1,7):
    #for link_qual in range(5,10):
        jscc_model.trg_trans_feat = tgt_trans_feat
        #args.link_qual = link_qual
        _, eval_aux = validate_epoch(eval_loader, jscc_model, bw = tgt_trans_feat, lpips_fn=lpips_loss_fn)
        print(f'BW {tgt_trans_feat} - PSNR: {eval_aux["psnr"]:.4f}, SSIM: {eval_aux["ssim"]:.4f}, LPIPS: {eval_aux["lpips"]:.4f}')
        #print(eval_aux['power'])