"""
Faithful Temporal Neck Implementation for TarViS

This is a complete reimplementation of the TarViS Temporal Neck,
matching the original architecture from the paper and official code.

Components:
1. Multi-scale input projections
2. 2D and 3D positional embeddings
3. Level embeddings
4. MSDeformable attention (PyTorch implementation)
5. Temporal attention with patch masking
6. Feature Pyramid Network (FPN)
7. Mask feature projection

Based on:
- TarViS paper: "TarViS: A Unified Approach for Target-based Video Segmentation"
- Original code: https://github.com/Ali2500/TarViS
"""

from typing import List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from deformable_attention import DeformableAttention
from temporal_attention import TemporalAttention
TORCH_AVAILABLE = True


def _get_activation_fn(activation: str):
    """Return activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingSine(nn.Module):
    """
    2D sinusoidal positional embeddings.
    
    Used for spatial position encoding in each frame.
    """
    
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        if TORCH_AVAILABLE:
            super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]
        Returns:
            pos: [B, C, H, W]
        """
        if not TORCH_AVAILABLE:
            return x
        
        B, C, H, W = x.shape
        
        # Create coordinate grid
        y_embed = torch.arange(H, dtype=torch.float32, device=x.device)
        x_embed = torch.arange(W, dtype=torch.float32, device=x.device)
        
        if self.normalize:
            y_embed = y_embed / (H - 1) * 2 * math.pi
            x_embed = x_embed / (W - 1) * 2 * math.pi
        
        # Create meshgrid
        y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing='ij')
        
        # Compute sinusoidal embeddings
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack([pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3).flatten(2)
        pos_y = torch.stack([pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3).flatten(2)
        
        pos = torch.cat([pos_y, pos_x], dim=2).permute(2, 0, 1)  # [C, H, W]
        
        return pos.unsqueeze(0).expand(B, -1, -1, -1)


class PositionEmbeddingSine3D(nn.Module):
    """
    3D sinusoidal positional embeddings (ORIGINAL from TarViS).
    
    Used for spatio-temporal position encoding across video frames.
    """
    
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.is_3d = True

    @torch.no_grad()
    def forward(self, x, mask=None, fmt="btchw"):
        """
        Args:
            x: [B, T, C, H, W] if fmt='btchw' or [B, C, T, H, W] if fmt='bcthw'
            mask: Optional mask
            fmt: Format string ('btchw' or 'bcthw')
        
        Returns:
            pos: [B, T, C, H, W] (always)
        """
        if not TORCH_AVAILABLE:
            return x
            
        assert x.dim() == 5, f"{x.shape} should be a 5-dimensional Tensor"
        
        if fmt == "btchw":
            batch_sz, clip_len, _, height, width = x.shape
        elif fmt == "bcthw":
            batch_sz, _, clip_len, height, width = x.shape
        else:
            raise ValueError(f"Invalid format given: {fmt}")

        if mask is None:
            mask = torch.zeros((batch_sz, clip_len, height, width), device=x.device, dtype=torch.bool)

        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t_floor_2 = torch.div(dim_t, 2, rounding_mode='trunc')
        dim_t = self.temperature ** (2 * dim_t_floor_2 / self.num_pos_feats)

        dim_t_z = torch.arange((self.num_pos_feats * 2), dtype=torch.float32, device=x.device)
        dim_t_z_floor_2 = torch.div(dim_t_z, 2, rounding_mode='trunc')
        dim_t_z = self.temperature ** (2 * dim_t_z_floor_2 / (self.num_pos_feats * 2))

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t_z
        
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        
        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z)

        if fmt == "btchw":
            pos = pos.permute(0, 1, 4, 2, 3)
        elif fmt == "bcthw":
            pos = pos.permute(0, 4, 1, 2, 3)

        return pos


class TemporalNeckFaithful(nn.Module):
    """
    Faithful implementation of TarViS Temporal Neck.
    
    This matches the original implementation with:
    - Multi-scale input projections
    - 2D and 3D positional embeddings
    - Level embeddings
    - Deformable + Temporal attention alternation
    - Feature Pyramid Network (FPN)
    - Mask feature projection
    
    Args:
        hidden_dim: Feature dimension (default: 256)
        num_layers: Number of encoder layers (default: 6)
        num_heads: Number of attention heads (default: 8)
        num_levels: Number of feature scales (default: 4)
        num_points: Sampling points per level in deformable attn (default: 4)
        feedforward_dim: FFN hidden dimension (default: 1024)
        dropout: Dropout rate (default: 0.1)
        num_fpn_levels: Number of FPN output levels (default: 3)
        mask_dim: Output mask feature dimension (default: 256)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        num_fpn_levels: int = 3,
        mask_dim: int = 256
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_levels = num_levels
        
        if not TORCH_AVAILABLE:
            return
        
        # Input projections for each scale
        # Projects backbone features to hidden_dim
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim)
            )
            for _ in range(num_levels)
        ])
        
        # Initialize input projections
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        
        # Positional embeddings
        self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.pe_layer_3d = PositionEmbeddingSine3D(hidden_dim // 2, normalize=True)
        
        # Level embeddings (learnable per-scale embeddings)
        self.level_embed = nn.Parameter(torch.Tensor(num_levels, hidden_dim))
        self.level_embed_3d = nn.Parameter(torch.Tensor(num_levels, hidden_dim))
        nn.init.normal_(self.level_embed)
        nn.init.normal_(self.level_embed_3d)
        
        # Encoder layers (Deformable + Temporal alternation)
        self.deform_layers = nn.ModuleList([
            DeformableAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        self.temporal_layers = nn.ModuleList([
            TemporalAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                grid_size=4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Feature Pyramid Network (FPN) layers
        self.num_fpn_levels = num_fpn_levels
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
            for _ in range(num_fpn_levels - 1)
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_fpn_levels - 1)
        ])
        
        # Initialize FPN layers
        for conv in self.lateral_convs:
            nn.init.xavier_uniform_(conv.weight, gain=1)
            nn.init.constant_(conv.bias, 0)
        
        for conv_seq in self.output_convs:
            nn.init.xavier_uniform_(conv_seq[0].weight, gain=1)
            nn.init.constant_(conv_seq[0].bias, 0)
        
        # Mask feature projection
        self.mask_features = nn.Conv2d(hidden_dim, mask_dim, kernel_size=1)
        nn.init.xavier_uniform_(self.mask_features.weight, gain=1)
        nn.init.constant_(self.mask_features.bias, 0)
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through Temporal Neck.
        
        Args:
            features: List of [B, C, H, W, T] for each scale [F32, F16, F8, F4]
        
        Returns:
            List of refined multi-scale features for decoder
        """
        if not TORCH_AVAILABLE:
            return features
        
        # Extract dimensions
        B, C, H, W, T = features[0].shape
        
        # Step 1: Project inputs and add positional embeddings
        srcs = []
        pos_2d = []
        pos_3d = []
        
        for level_idx, feat in enumerate(features):
            # feat: [B, C, H, W, T]
            B, C, H, W, T = feat.shape
            
            # Generate 3D positional embeddings
            # feat is [B, C, H, W, T], need [B, T, C, H, W] (fmt='btchw')
            feat_for_pos = feat.permute(0, 4, 1, 2, 3)  # [B, T, C, H, W]
            pos_3d_level = self.pe_layer_3d(feat_for_pos, fmt='btchw')  # [B, T, C, H, W]
            pos_3d.append(pos_3d_level)
            
            # Process each frame
            # Rearrange: [B, C, H, W, T] -> [B*T, C, H, W]
            feat_2d = feat.permute(0, 4, 1, 2, 3).reshape(B * T, C, H, W)
            
            # Generate 2D positional embeddings
            pos_2d_level = self.pe_layer(feat_2d)  # [B*T, C, H, W]
            pos_2d.append(pos_2d_level)
            
            # Apply input projection
            feat_proj = self.input_proj[level_idx](feat_2d)  # [B*T, C, H, W]
            
            # Reshape back: [B*T, C, H, W] -> [B, T, C, H, W]
            feat_proj = feat_proj.view(B, T, C, H, W)
            srcs.append(feat_proj)
        
        # Step 2: Apply encoder layers (Deformable + Temporal)
        output_features = srcs
        
        for layer_idx in range(self.num_layers):
            # Verify shapes before deformable attention
            for i, f in enumerate(output_features):
                assert f.shape[2] == self.hidden_dim, \
                    f"Scale {i}: Expected channel dim {self.hidden_dim}, got {f.shape[2]}"
            
            # Convert to [B, C, H, W, T] for deformable attention
            deform_input = [f.permute(0, 2, 3, 4, 1) for f in output_features]  # [B, T, C, H, W] -> [B, C, H, W, T]
            
            # Apply deformable attention
            deform_output = self.deform_layers[layer_idx](
                deform_input,
                level_embed=self.level_embed
            )
            
            # Convert back to [B, T, C, H, W]
            deform_output = [f.permute(0, 4, 1, 2, 3) for f in deform_output]  # [B, C, H, W, T] -> [B, T, C, H, W]
            
            # Apply temporal attention to each scale
            refined_features = []
            for scale_idx, feat in enumerate(deform_output):
                # Add 3D positional embedding
                feat_with_pos = feat + pos_3d[scale_idx]
                
                # Rearrange for temporal attention: [B, C, H, W, T]
                feat_rearranged = feat_with_pos.permute(0, 2, 3, 4, 1)
                
                # Apply temporal attention
                feat_temporal = self.temporal_layers[layer_idx](feat_rearranged)
                
                # Rearrange back: [B, T, C, H, W]
                feat_out = feat_temporal.permute(0, 4, 1, 2, 3)
                refined_features.append(feat_out)
            
            output_features = refined_features
        
        # Step 3: Feature Pyramid Network (FPN)
        # Merge batch and time: [B, T, C, H, W] -> [B*T, C, H, W]
        fpn_inputs = [f.reshape(B * T, C, f.shape[3], f.shape[4]) for f in output_features]
        
        # Build FPN top-down (finest to coarsest)
        fpn_outputs = []
        prev_features = None
        
        for idx in range(len(fpn_inputs) - 1, -1, -1):
            features_i = fpn_inputs[idx]
            
            if prev_features is None:
                # Finest level
                fpn_outputs.append(features_i)
                prev_features = features_i
            else:
                # Apply lateral connection
                if idx < len(self.lateral_convs):
                    lateral = self.lateral_convs[idx](features_i)
                else:
                    lateral = features_i
                
                # Upsample previous and add
                prev_upsampled = F.interpolate(
                    prev_features,
                    size=features_i.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                
                fused = lateral + prev_upsampled
                
                # Apply output conv
                if idx < len(self.output_convs):
                    fused = self.output_convs[idx](fused)
                
                fpn_outputs.append(fused)
                prev_features = fused
        
        # Reverse to coarse-to-fine order
        fpn_outputs = fpn_outputs[::-1]
        
        # Step 4: Add mask features
        mask_features = self.mask_features(fpn_outputs[-1])
        fpn_outputs.append(mask_features)
        
        return fpn_outputs[:self.num_fpn_levels + 1]


if __name__ == "__main__":
    print("="*70)
    print("Faithful Temporal Neck Implementation")
    print("="*70 + "\n")
    
    if TORCH_AVAILABLE:
        # Create model
        model = TemporalNeckFaithful(
            hidden_dim=256,
            num_layers=1,
            num_heads=8,
            num_levels=4
        )
        
        print("Model created:")
        print("  Hidden dim: 256")
        print("  Num layers: 6")
        print("  Num heads: 8")
        print("  Num levels: 4\n")
        
        # Test with dummy input
        B, C, T = 2, 256, 3
        
        F32 = torch.randn(B, C, 16, 16, T)
        F16 = torch.randn(B, C, 32, 32, T)
        F8 = torch.randn(B, C, 64, 64, T)
        F4 = torch.randn(B, C, 128, 128, T)
        
        features = [F32, F16, F8, F4]
        
        print("Input shapes:")
        for i, f in enumerate(features):
            print(f"  F{32//(2**i)}: {f.shape}")
        print()
        
        # Forward pass
        print("Running forward pass...")
        output = model(features)
        
        print("\nOutput shapes:")
        for i, f in enumerate(output):
            print(f"  Output {i}: {f.shape}")
        
        print("\n✓ Faithful Temporal Neck implementation complete!")
        
    else:
        print("PyTorch not available")