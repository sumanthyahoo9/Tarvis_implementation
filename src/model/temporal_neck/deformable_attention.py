"""
Deformable Attention for Multi-Scale Spatial Feature Interaction

Deformable Attention is an efficient attention mechanism that:
1. Learns WHERE to attend (sampling offsets)
2. Attends to multiple scales simultaneously
3. Uses only K sampling points instead of all H×W pixels

Key properties:
- Spatially unrestricted: Can attend anywhere in the frame
- Temporally restricted: Operates on single frame at a time
- Multi-scale: Aggregates information across [F32, F16, F8, F4]

Based on:
"Deformable DETR: Deformable Transformers for End-to-End Object Detection"
https://arxiv.org/abs/2010.04159

Implementation note:
In the full TarViS, this uses custom CUDA kernels for efficiency.
Here we provide a PyTorch-native implementation for understanding/testing.
"""

from typing import List, Optional, Tuple, Any, Union
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    TensorType = torch.Tensor
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    TensorType = Any


class DeformableAttention(nn.Module if TORCH_AVAILABLE else object):
    """
    Multi-scale Deformable Attention.
    
    Instead of attending to all spatial locations (expensive), learns to attend
    to K sampling points per reference point, with learnable offsets.
    
    For each query point:
    1. Predict K×L sampling offsets (K points per L levels)
    2. Predict K×L attention weights
    3. Sample features at offset locations
    4. Aggregate with attention weights
    
    Args:
        embed_dim: Feature dimension (D)
        num_heads: Number of attention heads (M)
        num_levels: Number of feature scales (L, typically 4)
        num_points: Number of sampling points per level (K, typically 4)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        
        if not TORCH_AVAILABLE:
            return
        
        # Check divisibility
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Predict sampling offsets
        # For each query: K points × L levels × 2 coordinates (x, y)
        self.sampling_offsets = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points * 2
        )
        
        # Predict attention weights
        # For each query: K points × L levels
        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * num_levels * num_points
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        # Initialize offsets to form a grid
        # This gives a good initialization: offsets start near a regular grid
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        
        # Initialize with uniform grid offsets
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32
        ) * (2.0 * math.pi / self.num_heads)
        
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2)
        grid_init = grid_init.repeat(1, self.num_levels, self.num_points, 1)
        
        # Scale down the offsets
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= (i + 1)
        
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(
                grid_init.view(-1)
            )
        
        # Initialize attention weights uniformly
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        # Initialize projections
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.query_proj.bias, 0.0)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.constant_(self.output_proj.bias, 0.0)
    
    def forward(
        self,
        multi_scale_features: List[TensorType],
        level_embed: Optional[TensorType] = None,
        temporal_pos_embed: Optional[TensorType] = None
    ) -> List[TensorType]:
        """
        Apply deformable attention across multiple scales.
        
        Args:
            multi_scale_features: List of [F32, F16, F8, F4]
                Each: [B, C, H, W, T] where T = number of frames
            level_embed: Level embeddings [L, C]
            temporal_pos_embed: Temporal position embeddings [T, C]
            
        Returns:
            Attended features for each scale (same shapes as input)
        """
        if not TORCH_AVAILABLE:
            return multi_scale_features
        
        # Process each frame independently (temporally local)
        B, C, H0, W0, T = multi_scale_features[0].shape
        
        output_features = []
        
        for feat_scale in multi_scale_features:
            B, C, H, W, T = feat_scale.shape
            
            # Process each frame
            frame_outputs = []
            for t in range(T):
                # Extract single frame: [B, C, H, W]
                frame = feat_scale[:, :, :, :, t]
                
                # Apply deformable attention on this frame
                attended_frame = self._forward_single_frame(
                    query_features=frame,
                    multi_scale_features=[
                        f[:, :, :, :, t] for f in multi_scale_features
                    ],
                    level_embed=level_embed
                )
                
                frame_outputs.append(attended_frame)
            
            # Stack frames back: [B, C, H, W, T]
            output_feat = torch.stack(frame_outputs, dim=-1)
            output_features.append(output_feat)
        
        return output_features
    
    def _forward_single_frame(
        self,
        query_features: TensorType,
        multi_scale_features: List[TensorType],
        level_embed: Optional[TensorType] = None
    ) -> TensorType:
        """
        Apply deformable attention on a single frame.
        
        Args:
            query_features: [B, C, H, W] - features for current scale
            multi_scale_features: List of [B, C, H_l, W_l] - all scales
            level_embed: [L, C] - level embeddings
            
        Returns:
            Attended features [B, C, H, W]
        """
        B, C, H, W = query_features.shape
        
        # Flatten spatial dimensions: [B, H*W, C]
        query_flat = query_features.flatten(2).permute(0, 2, 1)
        
        # Project queries
        queries = self.query_proj(query_flat)  # [B, H*W, C]
        
        # Get reference points (normalized coordinates for each spatial location)
        reference_points = self._get_reference_points(H, W, query_features.device)
        # reference_points: [H*W, 2] in range [0, 1]
        
        # Predict sampling offsets
        # [B, H*W, num_heads * num_levels * num_points * 2]
        sampling_offsets = self.sampling_offsets(queries)
        sampling_offsets = sampling_offsets.view(
            B, H * W, self.num_heads, self.num_levels, self.num_points, 2
        )
        
        # Predict attention weights
        # [B, H*W, num_heads * num_levels * num_points]
        attention_weights = self.attention_weights(queries)
        attention_weights = attention_weights.view(
            B, H * W, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            B, H * W, self.num_heads, self.num_levels, self.num_points
        )
        
        # Sample features at offset locations
        # This is the core deformable operation
        sampled_features = self._sample_features(
            multi_scale_features,
            reference_points,
            sampling_offsets
        )  # [B, H*W, num_heads, num_levels*num_points, head_dim]
        
        # Aggregate with attention weights
        # Reshape for weighted sum
        sampled_features = sampled_features.view(
            B, H * W, self.num_heads, self.num_levels, self.num_points, self.head_dim
        )
        
        # Weighted sum: [B, H*W, num_heads, head_dim]
        output = (sampled_features * attention_weights.unsqueeze(-1)).sum(dim=[3, 4])
        
        # Concatenate heads and project
        output = output.flatten(-2)  # [B, H*W, C]
        output = self.output_proj(output)
        output = self.dropout(output)
        
        # Reshape back to [B, C, H, W]
        output = output.permute(0, 2, 1).view(B, C, H, W)
        
        return output
    
    def _get_reference_points(
        self,
        H: int,
        W: int,
        device: torch.device
    ) -> TensorType:
        """
        Generate normalized reference points for each spatial location.
        
        Returns:
            reference_points: [H*W, 2] in range [0, 1]
        """
        # Create grid
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        
        # Normalize to [0, 1]
        y = (y + 0.5) / H
        x = (x + 0.5) / W
        
        # Stack and flatten: [H*W, 2]
        reference_points = torch.stack([x, y], dim=-1).flatten(0, 1)
        
        return reference_points
    
    def _sample_features(
        self,
        multi_scale_features: List[TensorType],
        reference_points: TensorType,
        sampling_offsets: TensorType
    ) -> TensorType:
        """
        Sample features at offset locations using bilinear interpolation.
        
        Args:
            multi_scale_features: List of [B, C, H_l, W_l]
            reference_points: [H*W, 2] normalized coordinates
            sampling_offsets: [B, H*W, num_heads, num_levels, num_points, 2]
            
        Returns:
            Sampled features: [B, H*W, num_heads, num_levels*num_points, head_dim]
        """
        B, _, _, _ = multi_scale_features[0].shape
        N = reference_points.shape[0]  # H*W
        
        # Project value features for all scales
        value_features = []
        for feat in multi_scale_features:
            # feat: [B, C, H_l, W_l]
            val = self.value_proj(feat.flatten(2).permute(0, 2, 1))
            # Reshape to multi-head: [B, H_l*W_l, num_heads, head_dim]
            val = val.view(B, -1, self.num_heads, self.head_dim)
            value_features.append(val)
        
        # Sample from each level
        sampled_list = []
        
        for level_idx, (feat, val) in enumerate(zip(multi_scale_features, value_features)):
            B, C, H_l, W_l = feat.shape
            
            # Get offsets for this level: [B, H*W, num_heads, num_points, 2]
            offsets = sampling_offsets[:, :, :, level_idx, :, :]
            
            # Expand reference points for heads and points
            # [H*W, 2] -> [B, H*W, num_heads, num_points, 2]
            ref_pts = reference_points.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            ref_pts = ref_pts.expand(B, N, self.num_heads, self.num_points, 2)
            
            # Add offsets (normalized by level size)
            sampling_locations = ref_pts + offsets / torch.tensor(
                [W_l, H_l], device=feat.device
            )
            
            # Clip to valid range
            sampling_locations = sampling_locations.clamp(0, 1)
            
            # Convert to grid_sample format: [-1, 1]
            sampling_grid = sampling_locations * 2 - 1
            
            # Reshape for grid_sample
            # [B, H*W*num_heads*num_points, 1, 2]
            sampling_grid = sampling_grid.flatten(1, 3).unsqueeze(2)
            
            # Sample features using grid_sample
            # feat: [B, C, H_l, W_l]
            # Need to expand for each head separately
            feat_expanded = feat.unsqueeze(1).expand(B, self.num_heads, C, H_l, W_l)
            feat_expanded = feat_expanded.flatten(0, 1)  # [B*num_heads, C, H_l, W_l]
            
            # Adjust sampling grid
            # Reshape to [B*num_heads, N, num_points, 1, 2]
            grid_per_head = sampling_grid.view(
                B, N, self.num_heads, self.num_points, 1, 2
            ).permute(0, 2, 1, 3, 4, 5).flatten(0, 1)  # [B*num_heads, N, num_points, 1, 2]
            
            # Flatten spatial dimensions for grid_sample
            # grid_sample expects [N, H_out, W_out, 2], so flatten to [N*num_points, 1, 2]
            grid_flat = grid_per_head.view(B * self.num_heads, N * self.num_points, 1, 2)
            
            # Sample
            sampled = F.grid_sample(
                feat_expanded,
                grid_flat,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # [B*num_heads, C, N*num_points, 1]
            
            # Remove last dimension and reshape
            sampled = sampled.squeeze(-1)  # [B*num_heads, C, N*num_points]
            
            # Reshape: [B, num_heads, C, N, num_points]
            sampled = sampled.view(
                B, self.num_heads, C, N, self.num_points
            )
            
            # Rearrange: [B, N, num_heads, num_points, C]
            sampled = sampled.permute(0, 3, 1, 4, 2)
            
            # Split into heads: [B, N, num_heads, num_points, head_dim]
            # Note: C might not equal embed_dim if using value projection
            # For now, use actual C dimension
            sampled_list.append(sampled)
        
        # Concatenate all levels: [B, N, num_heads, num_levels*num_points, C]
        # Stack along level dimension then reshape
        output = torch.stack(sampled_list, dim=3)  # [B, N, num_heads, num_levels, num_points, C]
        output = output.flatten(3, 4)  # [B, N, num_heads, num_levels*num_points, C]
        
        return output


if __name__ == "__main__":
    print("="*60)
    print("Deformable Attention Demo")
    print("="*60 + "\n")
    
    if TORCH_AVAILABLE:
        # Create module
        deform_attn = DeformableAttention(
            embed_dim=256,
            num_heads=8,
            num_levels=4,
            num_points=4
        )
        
        print(f"Deformable Attention:")
        print(f"  Embed dim: {deform_attn.embed_dim}")
        print(f"  Num heads: {deform_attn.num_heads}")
        print(f"  Num levels: {deform_attn.num_levels}")
        print(f"  Num points: {deform_attn.num_points}\n")
        
        # Create mock multi-scale features (single frame)
        B, C, T = 2, 256, 3
        
        F32 = torch.randn(B, C, 16, 16, T)
        F16 = torch.randn(B, C, 32, 32, T)
        F8 = torch.randn(B, C, 64, 64, T)
        F4 = torch.randn(B, C, 128, 128, T)
        
        multi_scale = [F32, F16, F8, F4]
        
        print("Input shapes:")
        for i, feat in enumerate(multi_scale):
            print(f"  Scale {i}: {feat.shape}")
        print()
        
        # Forward pass
        print("Applying deformable attention...")
        output = deform_attn(multi_scale)
        
        print("\nOutput shapes:")
        for i, feat in enumerate(output):
            print(f"  Scale {i}: {feat.shape}")
        
        print("\n✓ Deformable attention complete!")
        print("✓ Shapes preserved (spatially unrestricted, temporally local)")
        
    else:
        print("PyTorch not available")
        print("Install with: pip install torch")