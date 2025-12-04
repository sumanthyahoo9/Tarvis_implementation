"""
src/model/decoder/transformer_decoder.py

Transformer Decoder for TarViS video segmentation.

Applies L layers of:
- Masked cross-attention (queries → features)
- Self-attention (queries → queries)
- Feed-forward network

Following Mask2Former architecture adapted for video.
"""

import math
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    Tensor = None


class MLP(nn.Module):
    """
    Multi-layer perceptron.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_layers: Number of layers (minimum 2)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through MLP."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Single Transformer decoder layer.
    
    Contains:
    - Masked cross-attention
    - Self-attention
    - Feed-forward network
    - Layer normalizations
    
    Args:
        d_model: Feature dimension (256)
        nhead: Number of attention heads (8)
        dim_feedforward: FFN hidden dimension (2048)
        dropout: Dropout rate (0.1)
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Cross-attention: queries attend to features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Self-attention: queries attend to each other
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.activation = F.gelu
    
    def with_pos_embed(self, tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        """Add positional embedding if provided."""
        return tensor if pos is None else tensor + pos
    
    def forward(
        self,
        queries: Tensor,  # [N, C]
        features: Tensor,  # [B*T*H*W, C]
        query_pos: Optional[Tensor] = None,  # [N, C]
        attn_mask: Optional[Tensor] = None  # [N, B*T*H*W]
    ) -> Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            queries: Query embeddings [N, C]
            features: Flattened video features [B*T*H*W, C]
            query_pos: Query positional embeddings [N, C]
            attn_mask: Attention mask [N, B*T*H*W]
        
        Returns:
            Refined queries [N, C]
        """
        # Cross-attention: queries attend to features
        q = k = self.with_pos_embed(queries, query_pos)
        v = queries
        
        queries2, _ = self.cross_attn(
            query=q.unsqueeze(0),  # [1, N, C]
            key=features.unsqueeze(0),  # [1, B*T*H*W, C]
            value=features.unsqueeze(0),  # [1, B*T*H*W, C]
            attn_mask=attn_mask  # [N, B*T*H*W] if provided
        )
        queries2 = queries2.squeeze(0)  # [N, C]
        queries = queries + self.dropout1(queries2)
        queries = self.norm1(queries)
        
        # Self-attention: queries attend to each other
        q = k = self.with_pos_embed(queries, query_pos)
        v = queries
        
        queries2, _ = self.self_attn(
            query=q.unsqueeze(0),  # [1, N, C]
            key=k.unsqueeze(0),  # [1, N, C]
            value=v.unsqueeze(0)  # [1, N, C]
        )
        queries2 = queries2.squeeze(0)  # [N, C]
        queries = queries + self.dropout2(queries2)
        queries = self.norm2(queries)
        
        # Feed-forward network
        queries2 = self.linear2(self.dropout3(self.activation(self.linear1(queries))))
        queries = queries + self.dropout4(queries2)
        queries = self.norm3(queries)
        
        return queries


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder for TarViS.
    
    Applies L layers of masked cross-attention, self-attention, and FFN
    to iteratively refine query embeddings.
    
    Args:
        hidden_dim: Feature dimension (256)
        num_layers: Number of decoder layers (6)
        num_heads: Number of attention heads (8)
        dim_feedforward: FFN hidden dimension (2048)
        dropout: Dropout rate (0.1)
        return_intermediate: Return outputs from all layers
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        return_intermediate: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Mask prediction head (for masked cross-attention)
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        queries: Tensor,  # [N, C]
        features: Tensor,  # [B*T, C, H, W]
        query_pos: Optional[Tensor] = None  # [N, C]
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Forward pass through decoder.
        
        Args:
            queries: Initial query embeddings [N, C]
            features: Video features from temporal neck [B*T, C, H, W]
            query_pos: Query positional embeddings [N, C]
        
        Returns:
            output_queries: Refined queries [N, C]
            intermediate_outputs: List of queries from each layer (if return_intermediate)
        """
        if not TORCH_AVAILABLE:
            # Mock mode for CPU testing
            return queries, [queries] * self.num_layers
        
        # Flatten features: [B*T, C, H, W] -> [B*T*H*W, C]
        B_T, C, H, W = features.shape
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B*T, H*W, C]
        features_flat = features_flat.reshape(-1, C)  # [B*T*H*W, C]
        
        output = queries
        intermediate = []
        
        for layer in self.layers:
            # Predict attention mask from current queries
            # Inner product: [N, C] x [C, B*T*H*W] -> [N, B*T*H*W]
            mask_embed = self.mask_embed(output)  # [N, C]
            attn_mask = torch.einsum('nc,mc->nm', mask_embed, features_flat)  # [N, B*T*H*W]
            
            # Apply sigmoid to get soft attention mask
            attn_mask = attn_mask.sigmoid() < 0.5  # [N, B*T*H*W]
            attn_mask = attn_mask.detach()  # Don't backprop through mask
            
            # Apply decoder layer
            output = layer(
                queries=output,
                features=features_flat,
                query_pos=query_pos,
                attn_mask=attn_mask.float() * -1e9  # Convert to additive mask
            )
            
            if self.return_intermediate:
                intermediate.append(output)
        
        if self.return_intermediate:
            return output, intermediate
        else:
            return output, [output]


class MaskPredictor(nn.Module):
    """
    Mask prediction head.
    
    Predicts masks by computing inner product between queries and features.
    
    Args:
        hidden_dim: Feature dimension (256)
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
    
    def forward(self, queries: Tensor, features: Tensor) -> Tensor:
        """
        Predict masks.
        
        Args:
            queries: Query embeddings [N, C]
            features: Video features [B*T, C, H, W]
        
        Returns:
            Mask logits [B*T, N, H, W]
        """
        if not TORCH_AVAILABLE:
            # Mock mode
            B_T, C, H, W = features.shape
            N = queries.shape[0]
            return torch.zeros(B_T, N, H, W)
        
        # Project queries
        mask_embed = self.mask_embed(queries)  # [N, C]
        
        # Inner product: [N, C] x [B*T, C, H, W] -> [B*T, N, H, W]
        B_T, C, H, W = features.shape
        features_flat = features.flatten(2)  # [B*T, C, H*W]
        mask_logits = torch.einsum('nc,bcp->bnp', mask_embed, features_flat)  # [B*T, N, H*W]
        mask_logits = mask_logits.reshape(B_T, -1, H, W)  # [B*T, N, H, W]
        
        return mask_logits


class ClassPredictor(nn.Module):
    """
    Class prediction head (for VIS/VPS).
    
    Predicts classes by computing inner product between instance and semantic queries.
    
    Args:
        hidden_dim: Feature dimension (256)
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
    
    def forward(self, instance_queries: Tensor, semantic_queries: Tensor) -> Tensor:
        """
        Predict classes.
        
        Args:
            instance_queries: Instance query embeddings [I, C]
            semantic_queries: Semantic query embeddings [C+1, C] (classes + background)
        
        Returns:
            Class logits [I, C+1]
        """
        if not TORCH_AVAILABLE:
            # Mock mode
            I = instance_queries.shape[0]
            C_plus_1 = semantic_queries.shape[0]
            return torch.zeros(I, C_plus_1)
        
        # Inner product: [I, C] x [C+1, C] -> [I, C+1]
        class_logits = torch.einsum('ic,kc->ik', instance_queries, semantic_queries)
        
        return class_logits


def build_decoder(config: Optional[Dict] = None) -> TransformerDecoder:
    """
    Build Transformer Decoder from config.
    
    Args:
        config: Configuration dictionary with keys:
            - hidden_dim: Feature dimension (default: 256)
            - num_layers: Number of decoder layers (default: 6)
            - num_heads: Number of attention heads (default: 8)
            - dim_feedforward: FFN dimension (default: 2048)
            - dropout: Dropout rate (default: 0.1)
    
    Returns:
        TransformerDecoder instance
    """
    if config is None:
        config = {}
    
    return TransformerDecoder(
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        dim_feedforward=config.get('dim_feedforward', 2048),
        dropout=config.get('dropout', 0.1),
        return_intermediate=config.get('return_intermediate', True)
    )


# Example usage
if __name__ == "__main__":
    if TORCH_AVAILABLE:
        # Create decoder
        decoder = TransformerDecoder(
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            dim_feedforward=2048
        )
        
        # Create dummy inputs
        queries = torch.randn(10, 256)  # 10 queries
        features = torch.randn(5, 256, 30, 40)  # [B*T, C, H, W]
        
        # Forward pass
        output, intermediate = decoder(queries, features)
        
        print(f"Input queries: {queries.shape}")
        print(f"Input features: {features.shape}")
        print(f"Output queries: {output.shape}")
        print(f"Intermediate outputs: {len(intermediate)} layers")
        
        # Test mask predictor
        mask_predictor = MaskPredictor(hidden_dim=256)
        mask_logits = mask_predictor(output, features)
        print(f"Mask logits: {mask_logits.shape}")  # [B*T, N, H, W]
        
        # Test class predictor
        class_predictor = ClassPredictor(hidden_dim=256)
        instance_queries = output[:5]  # First 5 queries as instances
        semantic_queries = torch.randn(41, 256)  # 40 classes + background
        class_logits = class_predictor(instance_queries, semantic_queries)
        print(f"Class logits: {class_logits.shape}")  # [I, C+1]
    else:
        print("PyTorch not available - skipping example")