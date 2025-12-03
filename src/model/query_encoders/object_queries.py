"""
Object Query Encoder for VOS/PET Tasks

This module implements query encoding for Video Object Segmentation (VOS) and
Point Exemplar-guided Tracking (PET). Unlike semantic queries which are learned
embeddings, object queries are dynamically encoded from:
- VOS: First-frame object masks
- PET: Point coordinates inside objects

The encoder uses an iterative refinement process with self-attention (queries
attend to each other) and cross-attention (queries attend to image features).

Key difference from HODOR: Uses hard-masked attention with pmax points per object
instead of soft-masked attention over entire image.
"""

from typing import Dict, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
TORCH_AVAILABLE = True
TensorType = torch.Tensor


class ObjectEncoder(nn.Module):
    """
    Encodes objects from masks or points into query embeddings.
    
    Uses iterative refinement with:
    1. Self-attention: Queries attend to each other
    2. Cross-attention: Queries attend to masked image features
    
    Args:
        hidden_dim: Query embedding dimension
        num_layers: Number of refinement layers
        num_heads: Number of attention heads
        pmax: Maximum feature points per object (for efficiency)
        queries_per_object: Number of queries per object (qo in paper)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        pmax: int = 1024,
        queries_per_object: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pmax = pmax
        self.queries_per_object = queries_per_object
        
        if not TORCH_AVAILABLE:
            return
        
        # Encoder layers with self-attention and cross-attention
        self.layers = nn.ModuleList([
            ObjectEncoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
    
    def _initialize_queries_from_masks(
        self,
        masks: TensorType,
        features: TensorType
    ) -> TensorType:
        """
        Initialize object queries by spatial pooling features inside masks.
        
        For VOS, we divide each mask into qo spatial segments and pool separately.
        
        Args:
            masks: Object masks [B, O, H, W]
            features: Image features [B, C, H, W]
            
        Returns:
            Initial queries [B, O*qo, hidden_dim]
        """
        B, O, H, W = masks.shape
        _, C, Hf, Wf = features.shape
        
        # Resize masks to match feature spatial size
        if H != Hf or W != Wf:
            masks = F.interpolate(
                masks.float(),
                size=(Hf, Wf),
                mode='bilinear',
                align_corners=False
            )
        
        queries = []
        
        for o in range(O):
            # Get mask for this object [B, 1, Hf, Wf]
            obj_mask = masks[:, o:o+1, :, :]
            
            if self.queries_per_object == 1:
                # Single query: average pool entire mask
                # Masked features [B, C, Hf, Wf]
                masked_features = features * obj_mask
                
                # Sum and normalize
                mask_sum = obj_mask.sum(dim=[2, 3], keepdim=True).clamp(min=1e-5)
                query = masked_features.sum(dim=[2, 3]) / mask_sum.squeeze()
                
                queries.append(query)  # [B, C]
            else:
                # Multiple queries: divide mask into grid
                grid_size = int(math.sqrt(self.queries_per_object))
                
                # Split mask into grid
                h_split = Hf // grid_size
                w_split = Wf // grid_size
                
                for i in range(grid_size):
                    for j in range(grid_size):
                        h_start = i * h_split
                        h_end = (i + 1) * h_split if i < grid_size - 1 else Hf
                        w_start = j * w_split
                        w_end = (j + 1) * w_split if j < grid_size - 1 else Wf
                        
                        # Extract grid cell
                        cell_mask = obj_mask[:, :, h_start:h_end, w_start:w_end]
                        cell_features = features[:, :, h_start:h_end, w_start:w_end]
                        
                        # Pool
                        masked_cell = cell_features * cell_mask
                        cell_sum = cell_mask.sum(dim=[2, 3], keepdim=True).clamp(min=1e-5)
                        cell_query = masked_cell.sum(dim=[2, 3]) / cell_sum.squeeze()
                        
                        queries.append(cell_query)  # [B, C]
        
        # Stack queries [B, O*qo, C]
        queries = torch.stack(queries, dim=1)
        
        return queries
    
    def _initialize_queries_from_points(
        self,
        points: TensorType,
        features: TensorType
    ) -> TensorType:
        """
        Initialize object queries from point coordinates.
        
        For PET, we extract feature at point location.
        
        Args:
            points: Point coordinates [B, O, 2] in (x, y) format, normalized [0, 1]
            features: Image features [B, C, H, W]
            
        Returns:
            Initial queries [B, O, hidden_dim]
        """
        B, O, _ = points.shape
        _, C, H, W = features.shape
        
        # Convert normalized coordinates to feature map coordinates
        # points are in [0, 1], convert to [-1, 1] for grid_sample
        grid = points * 2.0 - 1.0  # [B, O, 2]
        grid = grid.unsqueeze(2)  # [B, O, 1, 2]
        
        # Extract features at point locations
        # grid_sample expects [B, C, H, W] and [B, H, W, 2]
        features_expanded = features.unsqueeze(1).expand(B, O, C, H, W)
        features_flat = features_expanded.reshape(B * O, C, H, W)
        
        grid_flat = grid.reshape(B * O, 1, 1, 2)
        
        # Sample features [B*O, C, 1, 1]
        sampled = F.grid_sample(
            features_flat,
            grid_flat,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        
        # Reshape to [B, O, C]
        queries = sampled.squeeze(-1).squeeze(-1).reshape(B, O, C)
        
        return queries
    
    def _extract_masked_features(
        self,
        masks: TensorType,
        features: TensorType
    ) -> Tuple[TensorType, TensorType]:
        """
        Extract features within masks, subsampling to pmax points per object.
        
        Args:
            masks: Object masks [B, O, H, W]
            features: Image features [B, C, H, W]
            
        Returns:
            masked_features: [B, O, pmax, C]
            attention_mask: [B, O, pmax] (1 for valid, 0 for padding)
        """
        B, O, H, W = masks.shape
        _, C, Hf, Wf = features.shape
        
        # Resize masks
        if H != Hf or W != Wf:
            masks = F.interpolate(
                masks.float(),
                size=(Hf, Wf),
                mode='bilinear',
                align_corners=False
            )
        
        masked_feats_list = []
        attn_mask_list = []
        
        for b in range(B):
            for o in range(O):
                # Get mask for this object [Hf, Wf]
                obj_mask = masks[b, o] > 0.5
                
                # Get feature coordinates where mask is True
                coords = obj_mask.nonzero()  # [N, 2] where N = number of True pixels
                
                if coords.shape[0] == 0:
                    # Empty mask - use padding
                    masked_feats_list.append(torch.zeros(self.pmax, C, device=features.device))
                    attn_mask_list.append(torch.zeros(self.pmax, device=features.device))
                elif coords.shape[0] <= self.pmax:
                    # Fewer points than pmax - use all + padding
                    feats = features[b, :, coords[:, 0], coords[:, 1]].T  # [N, C]
                    
                    # Pad to pmax
                    padding = torch.zeros(
                        self.pmax - coords.shape[0], C,
                        device=features.device
                    )
                    feats_padded = torch.cat([feats, padding], dim=0)
                    
                    # Attention mask
                    attn_mask = torch.ones(self.pmax, device=features.device)
                    attn_mask[coords.shape[0]:] = 0
                    
                    masked_feats_list.append(feats_padded)
                    attn_mask_list.append(attn_mask)
                else:
                    # More points than pmax - subsample
                    indices = torch.randperm(coords.shape[0], device=features.device)[:self.pmax]
                    selected_coords = coords[indices]
                    
                    feats = features[b, :, selected_coords[:, 0], selected_coords[:, 1]].T
                    attn_mask = torch.ones(self.pmax, device=features.device)
                    
                    masked_feats_list.append(feats)
                    attn_mask_list.append(attn_mask)
        
        # Stack into tensors
        masked_features = torch.stack(masked_feats_list, dim=0).reshape(B, O, self.pmax, C)
        attention_mask = torch.stack(attn_mask_list, dim=0).reshape(B, O, self.pmax)
        
        return masked_features, attention_mask
    
    def forward(
        self,
        features: TensorType,
        masks: Optional[TensorType] = None,
        points: Optional[TensorType] = None
    ) -> TensorType:
        """
        Encode objects from masks or points into query embeddings.
        
        Args:
            features: Image features [B, C, H, W]
            masks: Object masks [B, O, H, W] for VOS (mutually exclusive with points)
            points: Point coordinates [B, O, 2] for PET (mutually exclusive with masks)
            
        Returns:
            Refined object queries [B, O*qo, hidden_dim] (qo=1 for PET, qo=4 for VOS)
        """
        if not TORCH_AVAILABLE:
            # Mock output
            B, O = 1, 3
            qo = self.queries_per_object if masks is not None else 1
            return [[0.0] * self.hidden_dim] * (O * qo)
        
        assert (masks is not None) ^ (points is not None), \
            "Provide either masks (VOS) or points (PET), not both"
        
        # Initialize queries
        if masks is not None:
            # VOS: Initialize from masks
            queries = self._initialize_queries_from_masks(masks, features)
            B, O_qo, C = queries.shape
            O = masks.shape[1]
            
            # Extract masked features for cross-attention
            masked_features, attn_mask = self._extract_masked_features(masks, features)
        else:
            # PET: Initialize from points
            queries = self._initialize_queries_from_points(points, features)
            B, O, C = queries.shape
            O_qo = O
            
            # For PET, use entire feature map (no masking)
            _, _, H, W = features.shape
            masked_features = features.permute(0, 2, 3, 1).reshape(B, 1, H * W, C)
            masked_features = masked_features.expand(B, O, H * W, C)
            attn_mask = torch.ones(B, O, H * W, device=features.device)
        
        # Iterative refinement
        for layer in self.layers:
            queries = layer(queries, masked_features, attn_mask)
        
        # Final normalization
        queries = self.norm(queries)
        
        return queries


class ObjectEncoderLayer(nn.Module):
    """
    Single encoder layer with self-attention and cross-attention.
    
    Args:
        hidden_dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        if not TORCH_AVAILABLE:
            return
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        queries: TensorType,
        features: TensorType,
        attn_mask: Optional[TensorType] = None
    ) -> TensorType:
        """
        Args:
            queries: [B, N, C] where N = O*qo
            features: [B, O, P, C] where P = pmax or H*W
            attn_mask: [B, O, P] (1 for valid, 0 for padding)
            
        Returns:
            Refined queries [B, N, C]
        """
        # Self-attention
        queries2 = self.self_attn(queries, queries, queries)[0]
        queries = queries + self.dropout(queries2)
        queries = self.norm1(queries)
        
        # Cross-attention with masked features
        # For simplicity, we cross-attend each query to all object features
        # In practice, you'd want to attend query_i to features_i
        B, N, C = queries.shape
        B, O, P, C = features.shape
        
        # Reshape for cross-attention
        features_flat = features.reshape(B, O * P, C)
        
        # Create attention mask for cross-attention
        if attn_mask is not None:
            # attn_mask [B, O, P] -> [B, N, O*P]
            cross_attn_mask = attn_mask.reshape(B, O * P)
            cross_attn_mask = cross_attn_mask.unsqueeze(1).expand(B, N, O * P)
            # Convert to boolean (True = ignore)
            cross_attn_mask = ~cross_attn_mask.bool()
            # Repeat for num_heads: [B*num_heads, N, O*P]
            cross_attn_mask = cross_attn_mask.repeat_interleave(self.cross_attn.num_heads, dim=0)
        else:
            cross_attn_mask = None
        
        queries2 = self.cross_attn(
            queries,
            features_flat,
            features_flat,
            attn_mask=cross_attn_mask
        )[0]
        queries = queries + self.dropout(queries2)
        queries = self.norm2(queries)
        
        # FFN
        queries2 = self.ffn(queries)
        queries = queries + queries2
        queries = self.norm3(queries)
        
        return queries


class ObjectQueryEncoder(nn.Module):
    """
    Complete object query encoder for VOS/PET including background queries.
    
    This is the main interface that combines ObjectEncoder with background
    query generation.
    
    Args:
        hidden_dim: Query embedding dimension
        num_bg_queries: Number of background queries
        encoder_layers: Number of object encoder layers
        encoder_heads: Number of attention heads in encoder
        pmax: Maximum feature points per object
        queries_per_object: Number of queries per object (4 for VOS, 1 for PET)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_bg_queries: int = 16,
        encoder_layers: int = 3,
        encoder_heads: int = 8,
        pmax: int = 1024,
        queries_per_object: int = 4  # 4 for VOS, 1 for PET
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_bg_queries = num_bg_queries
        self.queries_per_object = queries_per_object
        
        if not TORCH_AVAILABLE:
            return
        
        # Object encoder
        self.object_encoder = ObjectEncoder(
            hidden_dim=hidden_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            pmax=pmax,
            queries_per_object=queries_per_object
        )
        
        # Background queries (learned)
        self.background_queries = nn.Parameter(
            torch.randn(num_bg_queries, hidden_dim)
        )
        
        nn.init.xavier_uniform_(self.background_queries)
    
    def forward(
        self,
        features: TensorType,
        masks: Optional[TensorType] = None,
        points: Optional[TensorType] = None,
        batch_size: int = 1
    ) -> Dict[str, Union[TensorType, int]]:
        """
        Generate complete query set (object + background).
        
        Args:
            features: Image features [B, C, H, W]
            masks: Object masks [B, O, H, W] for VOS
            points: Point coordinates [B, O, 2] for PET
            batch_size: Batch size (for background queries)
            
        Returns:
            Dictionary with:
                - 'queries': Concatenated [Qobj, Qbg]
                - 'num_objects': Number of object queries
                - 'num_background': Number of background queries
        """
        if not TORCH_AVAILABLE:
            O = 3
            return {
                'queries': [[0.0] * self.hidden_dim] * (O * self.queries_per_object + self.num_bg_queries),
                'num_objects': O * self.queries_per_object,
                'num_background': self.num_bg_queries
            }
        
        # Encode objects
        object_queries = self.object_encoder(features, masks, points)
        
        # Add background queries
        bg_queries = self.background_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Concatenate
        queries = torch.cat([object_queries, bg_queries], dim=1)
        
        return {
            'queries': queries,
            'num_objects': object_queries.shape[1],
            'num_background': self.num_bg_queries
        }


if __name__ == "__main__":
    print("=== Object Query Encoder Demo ===\n")
    
    if TORCH_AVAILABLE:
        # VOS example
        print("VOS Example:")
        encoder_vos = ObjectQueryEncoder(
            hidden_dim=256,
            queries_per_object=4  # Multiple queries per object
        )
        
        features = torch.randn(2, 256, 32, 32)
        masks = torch.rand(2, 3, 128, 128) > 0.5  # 3 objects
        
        output = encoder_vos(features, masks=masks.float(), batch_size=2)
        print(f"  Queries shape: {output['queries'].shape}")
        print(f"  Object queries: {output['num_objects']}")
        print(f"  Background queries: {output['num_background']}\n")
        
        # PET example
        print("PET Example:")
        encoder_pet = ObjectQueryEncoder(
            hidden_dim=256,
            queries_per_object=1  # Single query per object
        )
        
        points = torch.rand(2, 3, 2)  # 3 objects, (x, y) coordinates
        
        output = encoder_pet(features, points=points, batch_size=2)
        print(f"  Queries shape: {output['queries'].shape}")
        print(f"  Object queries: {output['num_objects']}")
        print(f"  Background queries: {output['num_background']}")
    else:
        print("PyTorch not available")
        print("Install PyTorch to see full demo:")
        print("  pip install torch torchvision")