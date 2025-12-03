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

from typing import List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
TORCH_AVAILABLE = True
TensorType = torch.Tensor


class DeformableAttention(nn.Module):
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
        )  # [B, N, num_heads, num_levels*num_points, head_dim]
        
        # sampled_features has shape [B, N, num_heads, L*K, head_dim]
        # We need to reshape to [B, N, num_heads, L, K, head_dim] for weighting
        B, N, M, LK, head_dim = sampled_features.shape
        sampled_features = sampled_features.view(
            B, N, M, self.num_levels, self.num_points, head_dim
        )
        
        # Aggregate with attention weights
        # attention_weights: [B, N, num_heads, num_levels, num_points]
        # Expand last dim for broadcasting: [B, N, M, L, K, 1]
        attn_weights_expanded = attention_weights.unsqueeze(-1)
        
        # Weighted sum over levels and points: [B, N, M, head_dim]
        output = (sampled_features * attn_weights_expanded).sum(dim=[3, 4])
        
        # Flatten heads: [B, N, M*head_dim] = [B, N, embed_dim]
        output = output.reshape(B, N, self.embed_dim)
        
        # Project (identity since already embed_dim)
        output = self.output_proj(output)  # [B, N, embed_dim]
        output = self.dropout(output)
        
        # Reshape back to [B, C, H, W]
        output = output.permute(0, 2, 1).view(B, self.embed_dim, H, W)
        
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
        
        This is the core of deformable attention:
        1. For each query point, predict K sampling offsets per level
        2. Sample features at those offset locations (bilinear interpolation)
        3. Return sampled features for aggregation with attention weights
        
        Args:
            multi_scale_features: List of [B, C, H_l, W_l] for each level
            reference_points: [N, 2] normalized coordinates (one per query)
            sampling_offsets: [B, N, num_heads, num_levels, num_points, 2]
            
        Returns:
            Sampled features: [B, N, num_heads, num_levels*num_points, C]
        """
        B = multi_scale_features[0].shape[0]
        C = self.embed_dim
        N = reference_points.shape[0]
        
        # Collect sampled features from all levels
        all_sampled = []
        
        for level_idx, feat in enumerate(multi_scale_features):
            # feat: [B, C, H_l, W_l]
            _, _, H_l, W_l = feat.shape
            
            # Get sampling offsets for this level
            # offsets: [B, N, num_heads, num_points, 2]
            offsets = sampling_offsets[:, :, :, level_idx, :, :]
            
            # Reference points: [N, 2] -> [B, N, 1, 1, 2]
            ref_pts = reference_points[None, :, None, None, :].expand(
                B, N, self.num_heads, self.num_points, 2
            )
            
            # Add offsets (normalize by spatial size)
            # Offsets are in normalized coordinates [-1, 1] range
            sampling_locations = ref_pts + offsets / torch.tensor(
                [W_l, H_l], dtype=torch.float32, device=feat.device
            )
            
            # Clip to valid range [0, 1]
            sampling_locations = torch.clamp(sampling_locations, 0, 1)
            
            # Convert to grid_sample format: [0, 1] -> [-1, 1]
            grid = sampling_locations * 2.0 - 1.0
            
            # Reshape for grid_sample: [B*num_heads, N*num_points, 1, 2]
            grid = grid.permute(0, 2, 1, 3, 4)  # [B, num_heads, N, num_points, 2]
            grid = grid.reshape(B * self.num_heads, N * self.num_points, 1, 2)
            
            # Project features with value projection first
            # feat: [B, C, H_l, W_l]
            feat_flat = feat.flatten(2).permute(0, 2, 1)  # [B, H_l*W_l, C]
            feat_value = self.value_proj(feat_flat)  # [B, H_l*W_l, C]
            
            # Split into heads: [B, H_l*W_l, num_heads, head_dim]
            feat_value = feat_value.view(B, H_l * W_l, self.num_heads, self.head_dim)
            
            # Rearrange for grid_sample: [B*num_heads, head_dim, H_l, W_l]
            feat_value = feat_value.permute(0, 2, 3, 1)  # [B, num_heads, head_dim, H_l*W_l]
            feat_value = feat_value.reshape(B * self.num_heads, self.head_dim, H_l, W_l)
            
            # Sample using bilinear interpolation
            # grid_sample input: [B*num_heads, head_dim, H_l, W_l]
            # grid_sample grid: [B*num_heads, N*num_points, 1, 2]
            # output: [B*num_heads, head_dim, N*num_points, 1]
            sampled = torch.nn.functional.grid_sample(
                feat_value,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )
            
            # Remove extra dimension: [B*num_heads, head_dim, N*num_points]
            sampled = sampled.squeeze(-1)
            
            # Reshape: [B, num_heads, head_dim, N*num_points]
            sampled = sampled.reshape(B, self.num_heads, self.head_dim, N * self.num_points)
            
            # Rearrange: [B, N, num_heads, num_points, head_dim]
            sampled = sampled.reshape(B, self.num_heads, self.head_dim, N, self.num_points)
            sampled = sampled.permute(0, 3, 1, 4, 2)  # [B, N, num_heads, num_points, head_dim]
            
            all_sampled.append(sampled)
        
        # Stack all levels: [B, N, num_heads, num_levels, num_points, head_dim]
        output = torch.stack(all_sampled, dim=3)
        
        # Flatten levels and points: [B, N, num_heads, num_levels*num_points, head_dim]
        output = output.reshape(B, N, self.num_heads, -1, self.head_dim)
        
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
        
        print("Deformable Attention:")
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