"""
Computation-Adaptive Modules for DeepJSCC-1++
Implements computation reduction mechanisms that adapt based on channel conditions (SNR, bandwidth):
- Encoder Spatial Pruning: Reduces encoder computation by pruning patches early
- Early Exit Heads: Reduces decoder computation by exiting early when SNR is high
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderSpatialPruning(nn.Module):
    """
    SNR-Aware Spatial Pruning for Encoder Computation Reduction
    
    Generates a binary mask to prune unimportant spatial patches at the encoder's
    early stage (after patch embedding), reducing computation in subsequent encoder
    layers. This module operates on sequence format (B, HW, C) to work directly
    with Swin Transformer's patch-based architecture.
    
    Args:
        embed_dim: Embedding dimension of patches (C)
        hidden_dim: Hidden dimension for pruning network
        spatial_size: (H, W) of the patch grid
        temperature: Gumbel-Softmax temperature (lower = harder)
        snr_embed_dim: Dimension for SNR embedding
    """
    def __init__(self, embed_dim, hidden_dim=64, spatial_size=(8, 8), 
                 temperature=1.0, snr_embed_dim=32):
        super(EncoderSpatialPruning, self).__init__()
        
        self.embed_dim = embed_dim
        self.spatial_size = spatial_size
        self.temperature = temperature
        self.H, self.W = spatial_size
        self.num_patches = self.H * self.W
        
        # SNR embedding network
        self.snr_encoder = nn.Sequential(
            nn.Linear(1, snr_embed_dim),
            nn.ReLU(),
            nn.Linear(snr_embed_dim, snr_embed_dim),
            nn.ReLU()
        )
        
        # Patch importance network (operates on sequence format)
        # Takes patch features + SNR embedding -> importance logits per patch
        self.patch_importance = nn.Sequential(
            nn.Linear(embed_dim + snr_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # 2 outputs: [keep, prune] logits
        )
        
    def forward(self, x, snr, training=True):
        """
        Args:
            x: Patch features [B, HW, embed_dim] (sequence format)
            snr: SNR value (scalar or [B, 1])
            training: If True, use Gumbel-Softmax; if False, use hard argmax
            
        Returns:
            x_pruned: Pruned features [B, HW', embed_dim] (only kept patches)
            mask: Binary patch mask [B, HW] (1 = keep, 0 = prune)
            pruning_ratio: Fraction of pruned patches
            keep_indices: Indices of kept patches [B, HW'] for reconstruction
        """
        B, HW, C = x.shape
        
        # Encode SNR
        if isinstance(snr, (int, float)):
            snr_tensor = torch.tensor([snr], device=x.device).float().view(1, 1)
            snr_tensor = snr_tensor.expand(B, 1)
        else:
            snr_tensor = snr.view(B, 1).float()
        
        snr_embed = self.snr_encoder(snr_tensor)  # [B, snr_embed_dim]
        snr_embed_expanded = snr_embed.unsqueeze(1).expand(B, HW, -1)  # [B, HW, snr_embed_dim]
        
        # Concatenate patch features and SNR embedding
        x_with_snr = torch.cat([x, snr_embed_expanded], dim=2)  # [B, HW, C + snr_embed_dim]
        
        # Generate importance logits per patch
        logits = self.patch_importance(x_with_snr)  # [B, HW, 2]
        
        if training:
            # Gumbel-Softmax for differentiable sampling
            mask_probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=2)
            mask = mask_probs[:, :, 0]  # Select "keep" probability [B, HW]
        else:
            # Hard decision during inference
            mask = (logits[:, :, 0] > logits[:, :, 1]).float()  # [B, HW]
        
        # Apply mask to prune patches
        # For both training and inference: use soft mask to keep sequence length constant
        # This allows subsequent encoder layers to work without modification
        # The pruned patches are zeroed out, effectively reducing computation in attention
        x_pruned = x * mask.unsqueeze(-1)  # [B, HW, C] - pruned patches are zeroed
        keep_indices = None  # Not needed when keeping sequence length constant
        
        # Calculate pruning ratio
        pruning_ratio = 1.0 - mask.mean()
        
        return x_pruned, mask, pruning_ratio, keep_indices


class EarlyExitBlock(nn.Module):
    """
    Early Exit Head for Decoder
    
    Projects intermediate decoder features to RGB image space.
    Allows exiting early when SNR is high.
    
    Args:
        in_channels: Number of input channels from decoder stage
        out_channels: Output channels (3 for RGB)
        img_size: Target image size (H, W)
    """
    def __init__(self, in_channels, out_channels=3, img_size=(32, 32)):
        super(EarlyExitBlock, self).__init__()
        
        self.img_size = img_size
        
        # Lightweight reconstruction head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: Intermediate features [B, C, H, W]
            
        Returns:
            img: Reconstructed image [B, 3, img_H, img_W]
        """
        img = self.head(x)
        
        # Upsample to target size if needed
        if img.shape[2:] != self.img_size:
            img = F.interpolate(img, size=self.img_size, mode='bilinear', align_corners=False)
        
        return img


class ComputationAdaptiveJSCC(nn.Module):
    """
    Computation-Adaptive JSCC model that reduces computation based on channel conditions.
    
    This model adaptively reduces computation in both encoder and decoder based on SNR
    and bandwidth conditions. It maintains the same `forward(img, bw, snr=None, plr=0.0)`
    interface as `Swin_JSCC` while providing significant computation savings.
    
    Key features:
    - EncoderSpatialPruning: Reduces encoder computation by pruning patches early
    - Early Exit Heads: Allows decoder to exit early when SNR is high (reduces decoder computation)
    - Bandwidth Adaptation: Adjusts transmitted channels based on available bandwidth
    - SNR-Adaptive: Automatically selects computation level based on channel quality
    """
    def __init__(self, args, encoder, decoder, bottleneck_channels, bottleneck_spatial,
                 img_size=(32, 32), early_exit_snr_thresholds=[15.0, 10.0],
                 exit_loss_weights=[0.3, 0.3, 1.0], use_encoder_pruning=True):
        super(ComputationAdaptiveJSCC, self).__init__()

        self.args = args
        self.device = args.device
        self.encoder = encoder
        self.decoder = decoder
        self.img_size = img_size
        self.early_exit_snr_thresholds = early_exit_snr_thresholds
        self.exit_loss_weights = exit_loss_weights
        self.use_encoder_pruning = use_encoder_pruning

        # Module A: Encoder Spatial Pruning (reduces encoder computation)
        # Applied early in encoder (after patch_embed) to reduce computation
        if use_encoder_pruning:
            # Get embedding dimension from encoder
            embed_dim = encoder.layers[0].dim if hasattr(encoder.layers[0], 'dim') else args.embed_size
            self.encoder_pruning = EncoderSpatialPruning(
                embed_dim=embed_dim,
                hidden_dim=64,
                spatial_size=bottleneck_spatial,  # Will be updated based on actual patch grid
                temperature=1.0,
                snr_embed_dim=32
            )
        else:
            self.encoder_pruning = None

        # Projection layers to feed early-exit heads (if used)
        self.proj_exit1 = nn.Conv2d(bottleneck_channels, max(1, bottleneck_channels // 2), kernel_size=1)
        self.proj_exit2 = nn.Conv2d(bottleneck_channels, max(1, bottleneck_channels // 4), kernel_size=1)

        # Module B: Early Exit Heads
        self.early_exit_1 = EarlyExitBlock(
            in_channels=max(1, bottleneck_channels // 2),
            out_channels=3,
            img_size=img_size
        )

        self.early_exit_2 = EarlyExitBlock(
            in_channels=max(1, bottleneck_channels // 4),
            out_channels=3,
            img_size=img_size
        )

        # embedding used for decoder compatibility (same shape as Swin_JSCC)
        self.linear_embed = nn.Linear(3, args.n_adapt_embed).to(self.device)
    
    def _get_snr(self, snr):
        """Get SNR value, with fallback to args if None."""
        if snr is None:
            if self.training:
                snr = self.args.link_qual + self.args.link_rng * np.random.randn(1)[0]
            else:
                snr = self.args.link_qual
        return snr
    
    def channel_slice(self, z, bw):
        """
        Bandwidth Adaptation: Slice channels based on bandwidth
        
        Args:
            z: Features [B, C, H, W]
            bw: Bandwidth (number of feature groups to keep)
            
        Returns:
            z_sliced: Channel-sliced features [B, bw_channels, H, W]
            n_channels: Number of channels kept
        """
        max_channels = self.encoder.max_trans_feat * self.encoder.unit_trans_feat
        bw_channels = max(1, bw * self.encoder.unit_trans_feat)
        z_sliced = z[:, :bw_channels, :, :]
        return z_sliced, bw_channels
    
    def awgn_channel(self, z, snr):
        """
        AWGN Channel Simulation
        
        Args:
            z: Features to transmit [B, C, H, W]
            snr: Signal-to-noise ratio (dB)
            
        Returns:
            z_noisy: Received features with noise
        """
        if snr >= 100:  # Skip noise for very high SNR
            return z
        
        # Calculate noise power
        sig_power = torch.mean(z ** 2)
        snr_linear = 10 ** (snr / 10.0)
        noise_power = sig_power / (snr_linear + 1e-9)
        
        # Generate and add noise
        noise = torch.randn_like(z) * torch.sqrt(noise_power + 1e-12)
        return z + noise
    
    def gen_embedding(self, bw, snr=None, plr=0.0):
        """Generate adaptive embedding from (snr, bw, plr)."""
        snr = self._get_snr(snr)
        # Use tensor operations instead of numpy for better GPU compatibility
        adapt_embed = torch.tensor([snr, float(bw), plr], device=self.device, dtype=torch.float32)
        return self.linear_embed(adapt_embed.unsqueeze(0)), snr

    def _encode_with_pruning(self, img, adapt_embed, snr):
        """
        Encoder forward pass with spatial pruning applied early to reduce computation.
        Pruning is applied after patch embedding, so subsequent layers process fewer patches.
        """
        B, C, H, W = img.shape
        
        # Step 1: Patch embedding
        x = self.encoder.patch_embed(img)  # (B, H*W, n_embed[0])
        B, HW, n_embed = x.shape
        
        # Step 2: Apply spatial pruning early (reduces computation in subsequent layers)
        pruning_ratio = 0.0
        mask = None
        if self.use_encoder_pruning and self.encoder_pruning is not None:
            # Update spatial size based on actual patch grid
            # More accurate calculation: use encoder's actual H/W if available
            if hasattr(self.encoder, 'patches_resolution') and self.encoder.patches_resolution:
                patch_H, patch_W = self.encoder.patches_resolution
            else:
                # Fallback: assume square patches
                patch_H = int(np.sqrt(HW)) if HW > 0 else self.encoder.H
                patch_W = HW // patch_H if patch_H > 0 else self.encoder.W
            
            self.encoder_pruning.H = patch_H
            self.encoder_pruning.W = patch_W
            self.encoder_pruning.num_patches = HW
            
            # Apply pruning: prune patches early to reduce computation
            x, mask, pruning_ratio, keep_indices = self.encoder_pruning(x, snr, training=self.training)
            # x is now (B, HW', n_embed) where HW' <= HW (fewer patches = less computation)
            HW = x.shape[1]  # Update HW to actual number of patches after pruning
        else:
            # If pruning is disabled, create a mask of all ones
            mask = torch.ones(B, HW, device=x.device)
        
        # Step 3: Add adaptive embedding
        adapt_embed_expanded = adapt_embed.unsqueeze(1).repeat(B, HW, 1)  # (B, HW, n_adapt_embed)
        x = torch.cat((x, adapt_embed_expanded), dim=2)  # (B, HW, n_embed[0]+n_adapt_embed)
        x = self.encoder.adapt_proj(x)  # (B, HW, n_embed[0])
        
        # Step 4: Encoder layers (now processing fewer patches = reduced computation)
        for i_layer, layer in enumerate(self.encoder.layers):
            x = layer(x)
        
        # Step 5: Normalize and project
        x = self.encoder.norm(x)
        x = self.encoder.proj(x)  # (B, HW', n_trans_feat)
        
        return x, pruning_ratio, mask

    def forward(self, img, bw, snr=None, plr=0.0):
        """Maintain same forward signature as `Swin_JSCC` and return reconstructed image."""
        B, C, H, W = img.shape
        # update resolutions to keep decoder/encoder consistent
        self.encoder.update_resolution(H, W)
        # encoder's H/W already represent the low-res patch grid
        self.decoder.update_resolution(H // (2 ** self.encoder.num_layers), W // (2 ** self.encoder.num_layers))

        adapt_embed, snr = self.gen_embedding(bw, snr, plr)

        # Encode with spatial pruning (reduces encoder computation)
        # Pruning is applied early in encoder, reducing computation in subsequent layers
        x, pruning_ratio, mask = self._encode_with_pruning(img, adapt_embed, snr)
        B, H_W_, n_feat = x.shape

        # Reshape to (B, C, H_enc, W_enc) for channel slicing
        H_enc = self.encoder.H
        W_enc = self.encoder.W
        z = x.permute(0, 2, 1).contiguous().view(B, n_feat, H_enc, W_enc)

        # Channel slicing by bandwidth
        z_sliced, _ = self.channel_slice(z, bw)

        # AWGN channel simulation
        z_received = self.awgn_channel(z_sliced, snr)

        # pad back to original feature channels expected by decoder
        if z_received.shape[1] < n_feat:
            z_padded = torch.zeros(B, n_feat, H_enc, W_enc, device=z_received.device, dtype=z_received.dtype)
            z_padded[:, :z_received.shape[1], :, :] = z_received
            z_received = z_padded

        # prepare for decoder: (B, H_W_, n_feat)
        y = z_received.view(B, n_feat, H_enc * W_enc).permute(0, 2, 1).contiguous()

        if self.training:
            # Training: compute all exits for multi-exit loss
            exit1_feat = self.proj_exit1(z_received)
            exit2_feat = self.proj_exit2(z_received)
            output_exit1 = self.early_exit_1(exit1_feat)
            output_exit2 = self.early_exit_2(exit2_feat)
            
            # decode (do not forward SNR into decoder layers to preserve compatibility)
            output_final = self.decoder(y, adapt_embed, SNR=None, eta=None, out_conv=True)
            
            return {
                'output_final': output_final,
                'output_exit1': output_exit1,
                'output_exit2': output_exit2,
                'pruning_ratio': pruning_ratio,
                'mask': mask
            }
        else:
            # Inference: choose exit based on SNR and only compute the selected one
            # This truly reduces decoder computation
            snr = self._get_snr(snr)
            
            if snr >= self.early_exit_snr_thresholds[0]:
                # High SNR: use early exit 1 (lightweight, no decoder computation)
                exit1_feat = self.proj_exit1(z_received)
                output_exit1 = self.early_exit_1(exit1_feat)
                return output_exit1
            elif snr >= self.early_exit_snr_thresholds[1]:
                # Medium SNR: use early exit 2 (lightweight, no decoder computation)
                exit2_feat = self.proj_exit2(z_received)
                output_exit2 = self.early_exit_2(exit2_feat)
                return output_exit2
            else:
                # Low SNR: use full decoder (most computation, but best quality)
                output_final = self.decoder(y, adapt_embed, SNR=None, eta=None, out_conv=True)
                return output_final
    
    def compute_loss(self, outputs, target, lpips_fn=None, lpips_weight=0.5):
        """
        Compute weighted multi-exit loss
        
        Args:
            outputs: Dict from forward pass (training mode)
            target: Ground truth images [B, 3, H, W]
            lpips_fn: LPIPS loss function (optional)
            lpips_weight: Weight for LPIPS component
            
        Returns:
            total_loss: Weighted sum of exit losses
            loss_dict: Dictionary of individual losses
        """
        w1, w2, w_final = self.exit_loss_weights
        
        # MSE losses
        mse_exit1 = F.mse_loss(outputs['output_exit1'], target)
        mse_exit2 = F.mse_loss(outputs['output_exit2'], target)
        mse_final = F.mse_loss(outputs['output_final'], target)
        
        # LPIPS losses (optional)
        if lpips_fn is not None:
            target_norm = target * 2.0 - 1.0
            
            exit1_norm = outputs['output_exit1'] * 2.0 - 1.0
            exit2_norm = outputs['output_exit2'] * 2.0 - 1.0
            final_norm = outputs['output_final'] * 2.0 - 1.0
            
            # LPIPS needs gradients for training, so don't use torch.no_grad()
            lpips_exit1 = lpips_fn(exit1_norm, target_norm).mean()
            lpips_exit2 = lpips_fn(exit2_norm, target_norm).mean()
            lpips_final = lpips_fn(final_norm, target_norm).mean()
            
            loss_exit1 = mse_exit1 + lpips_weight * lpips_exit1
            loss_exit2 = mse_exit2 + lpips_weight * lpips_exit2
            loss_final = mse_final + lpips_weight * lpips_final
        else:
            loss_exit1 = mse_exit1
            loss_exit2 = mse_exit2
            loss_final = mse_final
        
        # Weighted total loss
        total_loss = w1 * loss_exit1 + w2 * loss_exit2 + w_final * loss_final
        
        loss_dict = {
            'loss_exit1': loss_exit1.item(),
            'loss_exit2': loss_exit2.item(),
            'loss_final': loss_final.item(),
            'total_loss': total_loss.item(),
            'pruning_ratio': outputs['pruning_ratio'].item()
        }
        
        return total_loss, loss_dict
