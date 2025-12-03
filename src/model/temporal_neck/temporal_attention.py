"""
Temporal Attention for Grid-Based Temporal Feature Interaction

Temporal Attention enables features to interact across time while maintaining
spatial locality. Key properties:

1. Spatially restricted: Divides frame into grid cells
2. Temporally unrestricted: Attends across all frames within each cell
3. Efficient: Reduces H×W×T attention to (H/G)×(W/G) cells × G²×T points

Strategy:
- Divide frame into G×G grid (typically 4×4 = 16 cells)
- Within each cell, apply standard multi-head attention across time
- Features in cell (i,j) attend to same cell (i,j) in all frames

This is inspired by "Is Space-Time Attention All You Need for Video Understanding?"
(Bertasius et al., ICML 2021) but adapted for dense prediction.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
TORCH_AVAILABLE = True
TensorType = torch.Tensor


class TemporalAttention(nn.Module):
    """
    Grid-based Temporal Attention.
    
    Divides spatial dimensions into grid cells and applies temporal
    self-attention within each cell across all frames.
    
    For a grid cell at position (i, j):
    - Contains G×G spatial points (where G = H/grid_size)
    - Each point attends to corresponding points across T frames
    - Total attention: G²×T tokens per cell
    
    Args:
        embed_dim: Feature dimension (D)
        num_heads: Number of attention heads (M)
        grid_size: Grid divisions (G×G cells)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        grid_size: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()  
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.grid_size = grid_size
        
        if not TORCH_AVAILABLE:
            return
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.qkv.bias, 0.0)
        nn.init.constant_(self.proj.bias, 0.0)
    
    def forward(
        self,
        features: TensorType,
        temporal_pos_embed: Optional[TensorType] = None
    ) -> TensorType:
        """
        Apply temporal attention to features.
        
        Args:
            features: [B, C, H, W, T]
            temporal_pos_embed: Optional temporal position embeddings [T, C]
            
        Returns:
            Attended features: [B, C, H, W, T]
        """
        if not TORCH_AVAILABLE:
            return features
        
        B, C, H, W, T = features.shape
        
        # Check if spatial dimensions are divisible by grid_size
        assert H % self.grid_size == 0 and W % self.grid_size == 0, \
            f"Spatial dims ({H}, {W}) must be divisible by grid_size ({self.grid_size})"
        
        # Grid cell size
        cell_h = H // self.grid_size
        cell_w = W // self.grid_size
        
        # Reshape into grid cells
        # [B, C, H, W, T] -> [B, C, grid_size, cell_h, grid_size, cell_w, T]
        features = features.view(
            B, C, self.grid_size, cell_h, self.grid_size, cell_w, T
        )
        
        # Rearrange: [B, grid_size, grid_size, cell_h, cell_w, T, C]
        features = features.permute(0, 2, 4, 3, 5, 6, 1)
        
        # Reshape for attention within each cell
        # [B * grid_size * grid_size, cell_h * cell_w * T, C]
        num_cells = B * self.grid_size * self.grid_size
        features = features.reshape(num_cells, cell_h * cell_w * T, C)
        
        # Apply multi-head attention
        attended = self._attention(features, temporal_pos_embed, cell_h, cell_w, T)
        
        # Reshape back to grid
        # [B, grid_size, grid_size, cell_h, cell_w, T, C]
        attended = attended.view(
            B, self.grid_size, self.grid_size, cell_h, cell_w, T, C
        )
        
        # Rearrange back: [B, C, grid_size, cell_h, grid_size, cell_w, T]
        attended = attended.permute(0, 6, 1, 3, 2, 4, 5)
        
        # Merge grid cells: [B, C, H, W, T]
        attended = attended.reshape(B, C, H, W, T)
        
        return attended
    
    def _attention(
        self,
        features: TensorType,
        temporal_pos_embed: Optional[TensorType],
        cell_h: int,
        cell_w: int,
        T: int
    ) -> TensorType:
        """
        Apply multi-head attention within cells.
        
        Args:
            features: [num_cells, cell_h*cell_w*T, C]
            temporal_pos_embed: [T, C]
            cell_h, cell_w: Cell dimensions
            T: Number of frames
            
        Returns:
            Attended features: [num_cells, cell_h*cell_w*T, C]
        """
        num_cells, N, C = features.shape
        
        # Add temporal positional embeddings if provided
        if temporal_pos_embed is not None:
            # Expand temporal embeddings for all spatial positions
            # [T, C] -> [cell_h*cell_w, T, C]
            temp_embed = temporal_pos_embed.unsqueeze(0).expand(
                cell_h * cell_w, T, C
            )
            # Flatten: [cell_h*cell_w*T, C]
            temp_embed = temp_embed.reshape(-1, C)
            # Add to features
            features = features + temp_embed.unsqueeze(0)
        
        # Project to Q, K, V
        # [num_cells, N, 3*C]
        qkv = self.qkv(features)
        
        # Reshape to separate heads
        # [num_cells, N, 3, num_heads, head_dim]
        qkv = qkv.reshape(num_cells, N, 3, self.num_heads, self.head_dim)
        
        # Permute: [3, num_cells, num_heads, N, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Split into Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        # [num_cells, num_heads, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        # [num_cells, num_heads, N, head_dim]
        out = attn @ v
        
        # Concatenate heads
        # [num_cells, N, num_heads, head_dim]
        out = out.transpose(1, 2)
        
        # Flatten heads: [num_cells, N, C]
        out = out.reshape(num_cells, N, C)
        
        # Output projection
        out = self.proj(out)
        out = self.dropout(out)
        
        return out
    
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, "
            f"grid_size={self.grid_size}×{self.grid_size}"
        )


class AdaptiveTemporalAttention(nn.Module):
    """
    Adaptive Temporal Attention with learnable grid size.
    
    Instead of fixed grid, learns to group spatial locations dynamically.
    This is an advanced variant - the base TemporalAttention is sufficient
    for most cases.
    
    Args:
        embed_dim: Feature dimension
        num_heads: Number of attention heads
        max_grid_size: Maximum grid size
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        max_grid_size: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_grid_size = max_grid_size
        
        if not TORCH_AVAILABLE:
            return
        
        # Base temporal attention
        self.temporal_attn = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            grid_size=max_grid_size,
            dropout=dropout
        )
        
        # Learnable grid size predictor (advanced feature)
        self.grid_predictor = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(embed_dim, max_grid_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(
        self,
        features: TensorType,
        temporal_pos_embed: Optional[TensorType] = None
    ) -> TensorType:
        """
        Apply adaptive temporal attention.
        
        For simplicity, this implementation uses fixed grid.
        A full implementation would dynamically adjust grouping.
        """
        # For now, just use base temporal attention
        return self.temporal_attn(features, temporal_pos_embed)


if __name__ == "__main__":
    print("="*60)
    print("Temporal Attention Demo")
    print("="*60 + "\n")
    
    if TORCH_AVAILABLE:
        # Create module
        temporal_attn = TemporalAttention(
            embed_dim=256,
            num_heads=8,
            grid_size=4
        )
        
        print("Temporal Attention:")
        print(f"  {temporal_attn.extra_repr()}\n")
        
        # Create mock features
        B, C, H, W, T = 2, 256, 64, 64, 5
        features = torch.randn(B, C, H, W, T)
        
        print(f"Input shape: {features.shape}")
        print(f"  Batch: {B}")
        print(f"  Channels: {C}")
        print(f"  Spatial: {H}×{W}")
        print(f"  Temporal: {T} frames\n")
        
        print("Grid division:")
        print("  Grid size: 4×4 = 16 cells")
        print(f"  Cell size: {H//4}×{W//4}")
        print(f"  Tokens per cell: {(H//4)*(W//4)*T} = {H//4}×{W//4} spatial × {T} temporal\n")
        
        # Optional: temporal positional embeddings
        temporal_pos_embed = torch.randn(T, C)
        
        # Forward pass
        print("Applying temporal attention...")
        output = temporal_attn(features, temporal_pos_embed)
        
        print(f"\nOutput shape: {output.shape}")
        print("✓ Shape preserved!")
        
        print("\n" + "="*60)
        print("Key Properties:")
        print("="*60)
        print("✓ Spatially restricted: Attention within grid cells")
        print("✓ Temporally unrestricted: Attention across all frames")
        print("✓ Efficient: Reduces from H×W×T to smaller cell-based attention")
        print("="*60)
        
    else:
        print("PyTorch not available")
        print("Install with: pip install torch")