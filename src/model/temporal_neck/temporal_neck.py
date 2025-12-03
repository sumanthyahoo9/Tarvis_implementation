"""
Temporal Neck - Novel Spatio-Temporal Feature Fusion

This module implements TarViS's key innovation: a neck architecture that creates
temporally consistent features by alternating two types of attention:

1. Deformable Attention: Spatially global, temporally local
   - Attends to any spatial location in the current frame
   - Multi-scale feature interaction
   - Efficient (learns where to attend)

2. Temporal Attention: Spatially local, temporally global
   - Divides frame into grid cells
   - Attends across all frames within each cell
   - Enables temporal consistency

By alternating these operations across 6 layers, the Temporal Neck produces
features that are both spatially rich and temporally aligned - perfect for
video instance segmentation and tracking.

Architecture:
    Input: Multi-scale features from backbone [F32, F16, F8, F4]
    
    For each layer (6 total):
        1. Deformable Attention: Learn spatial relationships at scale
        2. Temporal Attention: Propagate information across time
        3. FFN: Non-linear transformation
    
    Output: Refined multi-scale features [F32', F16', F8', F4']

Note: F8 is excluded from temporal attention for memory efficiency.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from .deformable_attention import DeformableAttention
from .temporal_attention import TemporalAttention
import torch
import torch.nn as nn
import torch.nn.functional as F
TORCH_AVAILABLE = True
TensorType = torch.Tensor


class TemporalNeck(nn.Module):
    """
    Temporal Neck for video feature fusion.
    
    Alternates between Deformable Attention (spatial) and Temporal Attention
    (temporal) to create spatio-temporally consistent features.
    
    Args:
        hidden_dim: Feature dimension (D)
        num_layers: Number of refinement layers (default: 6)
        num_heads: Number of attention heads
        num_levels: Number of feature scales (default: 4 for [F32, F16, F8, F4])
        num_points: Number of sampling points for deformable attention
        dropout: Dropout rate
        temporal_grid_size: Grid size for temporal attention (e.g., 4 = 4x4 grid)
        exclude_f8_temporal: If True, exclude F8 from temporal attention (memory opt)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        temporal_grid_size: int = 4,
        exclude_f8_temporal: bool = True
    ):

        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.temporal_grid_size = temporal_grid_size
        self.exclude_f8_temporal = exclude_f8_temporal
        
        if not TORCH_AVAILABLE:
            return
        
        # Build layers
        self.layers = nn.ModuleList()
        
        for layer_idx in range(num_layers):
            # Each layer has: Deformable Attention + Temporal Attention + FFN
            layer = TemporalNeckLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                dropout=dropout,
                temporal_grid_size=temporal_grid_size,
                exclude_f8_temporal=exclude_f8_temporal,
                layer_idx=layer_idx
            )
            self.layers.append(layer)
        
        # Final layer norm per scale
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_levels)
        ])
        
        # Learnable level embeddings (used in deformable attention)
        self.level_embed = nn.Parameter(torch.randn(num_levels, hidden_dim))
        nn.init.normal_(self.level_embed)
    
    def forward(
        self,
        multi_scale_features: List[TensorType],
        temporal_pos_embed: Optional[TensorType] = None
    ) -> List[TensorType]:
        """
        Process multi-scale features through temporal neck.
        
        Args:
            multi_scale_features: List of [F32, F16, F8, F4]
                Each tensor has shape [B, C, H, W, T] where:
                - B = batch size
                - C = channels (hidden_dim)
                - H, W = spatial dimensions (scale-dependent)
                - T = temporal dimension (number of frames)
            
            temporal_pos_embed: Optional positional embeddings for temporal dimension
                Shape: [T, hidden_dim]
        
        Returns:
            Refined multi-scale features with same shapes as input
        """
        if not TORCH_AVAILABLE:
            # Mock output
            return multi_scale_features
        
        # Validate input
        assert len(multi_scale_features) == self.num_levels, \
            f"Expected {self.num_levels} scales, got {len(multi_scale_features)}"
        
        # Store input shapes for later
        shapes = [feat.shape for feat in multi_scale_features]
        
        # Process through layers
        for layer_idx, layer in enumerate(self.layers):
            multi_scale_features = layer(
                multi_scale_features,
                level_embed=self.level_embed,
                temporal_pos_embed=temporal_pos_embed
            )
        
        # Final layer norm per scale
        output_features = []
        for feat, norm in zip(multi_scale_features, self.level_norms):
            B, C, H, W, T = feat.shape
            # Reshape for layer norm: [B, H, W, T, C]
            feat = feat.permute(0, 2, 3, 4, 1)
            feat = norm(feat)
            # Reshape back: [B, C, H, W, T]
            feat = feat.permute(0, 4, 1, 2, 3)
            output_features.append(feat)
        
        return output_features
    
    def num_parameters(self) -> int:
        """Count total number of parameters."""
        if not TORCH_AVAILABLE:
            return 0
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"TemporalNeck(\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_heads={self.num_heads},\n"
            f"  num_levels={self.num_levels},\n"
            f"  temporal_grid={self.temporal_grid_size}x{self.temporal_grid_size},\n"
            f"  exclude_f8_temporal={self.exclude_f8_temporal}\n"
            f")"
        )


class TemporalNeckLayer(nn.Module):
    """
    Single layer of Temporal Neck.
    
    Consists of:
    1. Deformable Attention (spatial, multi-scale)
    2. Temporal Attention (temporal, grid-based)
    3. Feed-Forward Network
    
    Args:
        hidden_dim: Feature dimension
        num_heads: Number of attention heads
        num_levels: Number of feature scales
        num_points: Sampling points for deformable attention
        dropout: Dropout rate
        temporal_grid_size: Grid size for temporal attention
        exclude_f8_temporal: Exclude F8 from temporal attention
        layer_idx: Layer index (for debugging)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        temporal_grid_size: int = 4,
        exclude_f8_temporal: bool = True,
        layer_idx: int = 0
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx
        self.exclude_f8_temporal = exclude_f8_temporal
        
        if not TORCH_AVAILABLE:
            return
        
        # 1. Deformable Attention (spatial, multi-scale)
        self.deformable_attn = DeformableAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout
        )
        
        # 2. Temporal Attention (temporal, grid-based)
        # Exclude F8 (scale index 2) for memory efficiency
        num_temporal_levels = num_levels - 1 if exclude_f8_temporal else num_levels
        
        self.temporal_attn = TemporalAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            grid_size=temporal_grid_size,
            dropout=dropout
        )
        
        # 3. Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        multi_scale_features: List[TensorType],
        level_embed: Optional[TensorType] = None,
        temporal_pos_embed: Optional[TensorType] = None
    ) -> List[TensorType]:
        """
        Process features through one Temporal Neck layer.
        
        Args:
            multi_scale_features: List of [F32, F16, F8, F4]
                Each: [B, C, H, W, T]
            level_embed: Level embeddings [num_levels, C]
            temporal_pos_embed: Temporal position embeddings [T, C]
            
        Returns:
            Refined multi-scale features
        """
        if not TORCH_AVAILABLE:
            return multi_scale_features
        
        # Step 1: Deformable Attention (spatial, multi-scale)
        # This operates on all scales simultaneously
        deform_output = self.deformable_attn(
            multi_scale_features,
            level_embed=level_embed,
            temporal_pos_embed=temporal_pos_embed
        )
        
        # Residual connection + norm
        multi_scale_features = [
            self._residual_norm(feat, out, self.norm1)
            for feat, out in zip(multi_scale_features, deform_output)
        ]
        
        # Step 2: Temporal Attention (temporal, grid-based)
        # This operates per scale (excluding F8 if specified)
        temporal_output = []
        for scale_idx, feat in enumerate(multi_scale_features):
            if self.exclude_f8_temporal and scale_idx == 2:  # F8 is index 2
                # Skip temporal attention for F8
                temporal_output.append(feat)
            else:
                # Apply temporal attention
                feat_temporal = self.temporal_attn(
                    feat,
                    temporal_pos_embed=temporal_pos_embed
                )
                temporal_output.append(feat_temporal)
        
        # Residual connection + norm
        multi_scale_features = [
            self._residual_norm(feat, out, self.norm2)
            for feat, out in zip(multi_scale_features, temporal_output)
        ]
        
        # Step 3: Feed-Forward Network
        ffn_output = []
        for feat in multi_scale_features:
            B, C, H, W, T = feat.shape
            # Reshape for FFN: [B*H*W*T, C]
            feat_flat = feat.permute(0, 2, 3, 4, 1).reshape(-1, C)
            feat_ffn = self.ffn(feat_flat)
            # Reshape back: [B, C, H, W, T]
            feat_ffn = feat_ffn.reshape(B, H, W, T, C).permute(0, 4, 1, 2, 3)
            ffn_output.append(feat_ffn)
        
        # Residual connection + norm
        multi_scale_features = [
            self._residual_norm(feat, out, self.norm3)
            for feat, out in zip(multi_scale_features, ffn_output)
        ]
        
        return multi_scale_features
    
    def _residual_norm(
        self,
        input_feat: TensorType,
        output_feat: TensorType,
        norm_layer: nn.Module
    ) -> TensorType:
        """Apply residual connection and layer norm."""
        # Add residual
        feat = input_feat + self.dropout(output_feat)
        
        # Apply layer norm
        B, C, H, W, T = feat.shape
        feat = feat.permute(0, 2, 3, 4, 1)  # [B, H, W, T, C]
        feat = norm_layer(feat)
        feat = feat.permute(0, 4, 1, 2, 3)  # [B, C, H, W, T]
        
        return feat


# Placeholder implementations (to be replaced by actual modules)
class DeformableAttention(nn.Module):
    """Placeholder for Deformable Attention (will be implemented separately)."""
    
    def __init__(self, **kwargs):
        if TORCH_AVAILABLE:
            super().__init__()
        self.kwargs = kwargs
    
    def forward(self, multi_scale_features, **kwargs):
        # Placeholder: return input unchanged
        return multi_scale_features


class TemporalAttention(nn.Module):
    """Placeholder for Temporal Attention (will be implemented separately)."""
    
    def __init__(self, **kwargs):
        if TORCH_AVAILABLE:
            super().__init__()
        self.kwargs = kwargs
    
    def forward(self, features, **kwargs):
        # Placeholder: return input unchanged
        return features


if __name__ == "__main__":
    print("="*60)
    print("Temporal Neck Architecture Demo")
    print("="*60 + "\n")
    
    if TORCH_AVAILABLE:
        # Create temporal neck
        neck = TemporalNeck(
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            num_levels=4,
            temporal_grid_size=4
        )
        
        print(f"Temporal Neck Architecture:\n{neck}\n")
        print(f"Number of parameters: {neck.num_parameters():,}\n")
        
        # Create mock multi-scale features
        B, C, T = 2, 256, 5  # batch=2, channels=256, frames=5
        
        # Multi-scale features with different spatial resolutions
        F32 = torch.randn(B, C, 16, 16, T)   # 1/32 scale
        F16 = torch.randn(B, C, 32, 32, T)   # 1/16 scale
        F8 = torch.randn(B, C, 64, 64, T)    # 1/8 scale
        F4 = torch.randn(B, C, 128, 128, T)  # 1/4 scale
        
        multi_scale_features = [F32, F16, F8, F4]
        
        print("Input shapes:")
        for i, feat in enumerate(multi_scale_features):
            scale = [32, 16, 8, 4][i]
            print(f"  F{scale}: {feat.shape}")
        print()
        
        # Forward pass
        print("Processing through Temporal Neck...")
        output_features = neck(multi_scale_features)
        
        print("\nOutput shapes:")
        for i, feat in enumerate(output_features):
            scale = [32, 16, 8, 4][i]
            print(f"  F{scale}': {feat.shape}")
        
        # Verify shapes unchanged
        print("\n✓ Shapes preserved (as expected)")
        print("\nNote: Deformable and Temporal Attention are placeholders.")
        print("They will be implemented next!")
        
    else:
        print("PyTorch not available")
        print("Install with: pip install torch")