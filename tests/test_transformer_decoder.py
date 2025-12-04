"""
tests/test_transformer_decoder.py

Unit tests for Transformer Decoder.
"""

import pytest
import sys
from pathlib import Path
from src.model.decoder.transformer_decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    MaskPredictor,
    ClassPredictor,
    MLP,
    build_decoder
)

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestMLP:
    """Test MLP module."""
    
    def test_mlp_init(self):
        """Test MLP initialization."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        mlp = MLP(input_dim=256, hidden_dim=512, output_dim=256, num_layers=3)
        assert mlp.num_layers == 3
        assert len(mlp.layers) == 3
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        mlp = MLP(input_dim=256, hidden_dim=512, output_dim=256, num_layers=3)
        x = torch.randn(10, 256)
        output = mlp(x)
        
        assert output.shape == (10, 256)


class TestTransformerDecoderLayer:
    """Test single decoder layer."""
    
    def test_layer_init(self):
        """Test layer initialization."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        layer = TransformerDecoderLayer(d_model=256, nhead=8)
        assert layer.cross_attn.embed_dim == 256
        assert layer.self_attn.embed_dim == 256
    
    def test_layer_forward(self):
        """Test layer forward pass."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        layer = TransformerDecoderLayer(d_model=256, nhead=8)
        
        queries = torch.randn(10, 256)  # 10 queries
        features = torch.randn(600, 256)  # Flattened features
        
        output = layer(queries, features)
        
        assert output.shape == (10, 256)
    
    def test_layer_with_pos_embed(self):
        """Test layer with positional embeddings."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        layer = TransformerDecoderLayer(d_model=256, nhead=8)
        
        queries = torch.randn(10, 256)
        features = torch.randn(600, 256)
        query_pos = torch.randn(10, 256)
        
        output = layer(queries, features, query_pos=query_pos)
        
        assert output.shape == (10, 256)
    
    def test_layer_with_attn_mask(self):
        """Test layer with attention mask."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        layer = TransformerDecoderLayer(d_model=256, nhead=8)
        
        queries = torch.randn(10, 256)
        features = torch.randn(600, 256)
        attn_mask = torch.zeros(10, 600)  # No masking
        
        output = layer(queries, features, attn_mask=attn_mask)
        
        assert output.shape == (10, 256)


class TestTransformerDecoder:
    """Test full Transformer Decoder."""
    
    def test_decoder_init(self):
        """Test decoder initialization."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        decoder = TransformerDecoder(
            hidden_dim=256,
            num_layers=6,
            num_heads=8
        )
        
        assert decoder.hidden_dim == 256
        assert decoder.num_layers == 6
        assert len(decoder.layers) == 6
    
    def test_decoder_forward(self):
        """Test decoder forward pass."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        decoder = TransformerDecoder(hidden_dim=256, num_layers=3)
        
        queries = torch.randn(10, 256)  # 10 queries
        features = torch.randn(5, 256, 30, 40)  # [B*T, C, H, W]
        
        output, intermediate = decoder(queries, features)
        
        assert output.shape == (10, 256)
        assert len(intermediate) == 3  # 3 layers
        for inter in intermediate:
            assert inter.shape == (10, 256)
    
    def test_decoder_with_query_pos(self):
        """Test decoder with query positional embeddings."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        decoder = TransformerDecoder(hidden_dim=256, num_layers=3)
        
        queries = torch.randn(10, 256)
        features = torch.randn(5, 256, 30, 40)
        query_pos = torch.randn(10, 256)
        
        output, intermediate = decoder(queries, features, query_pos=query_pos)
        
        assert output.shape == (10, 256)
        assert len(intermediate) == 3
    
    def test_decoder_no_intermediate(self):
        """Test decoder without intermediate outputs."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        decoder = TransformerDecoder(
            hidden_dim=256,
            num_layers=3,
            return_intermediate=False
        )
        
        queries = torch.randn(10, 256)
        features = torch.randn(5, 256, 30, 40)
        
        output, intermediate = decoder(queries, features)
        
        assert output.shape == (10, 256)
        assert len(intermediate) == 1  # Only final output
    
    def test_decoder_different_sizes(self):
        """Test decoder with different input sizes."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        decoder = TransformerDecoder(hidden_dim=128, num_layers=2)
        
        # Small
        queries = torch.randn(5, 128)
        features = torch.randn(2, 128, 15, 20)
        output, _ = decoder(queries, features)
        assert output.shape == (5, 128)
        
        # Large
        queries = torch.randn(100, 128)
        features = torch.randn(10, 128, 60, 80)
        output, _ = decoder(queries, features)
        assert output.shape == (100, 128)
    
    def test_decoder_mock_mode(self):
        """Test decoder in mock mode (no PyTorch)."""
        decoder = TransformerDecoder(hidden_dim=256, num_layers=3)
        
        # Mock inputs
        queries = type('Tensor', (), {'shape': (10, 256)})()
        features = type('Tensor', (), {'shape': (5, 256, 30, 40)})()
        
        # Should not crash in mock mode
        # (returns mock data when TORCH_AVAILABLE=False)


class TestMaskPredictor:
    """Test mask prediction head."""
    
    def test_mask_predictor_init(self):
        """Test mask predictor initialization."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        predictor = MaskPredictor(hidden_dim=256)
        assert predictor.mask_embed.num_layers == 3
    
    def test_mask_predictor_forward(self):
        """Test mask prediction."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        predictor = MaskPredictor(hidden_dim=256)
        
        queries = torch.randn(10, 256)  # 10 objects
        features = torch.randn(5, 256, 30, 40)  # [B*T, C, H, W]
        
        mask_logits = predictor(queries, features)
        
        assert mask_logits.shape == (5, 10, 30, 40)  # [B*T, N, H, W]
    
    def test_mask_predictor_single_query(self):
        """Test mask prediction with single query."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        predictor = MaskPredictor(hidden_dim=256)
        
        queries = torch.randn(1, 256)
        features = torch.randn(3, 256, 20, 30)
        
        mask_logits = predictor(queries, features)
        
        assert mask_logits.shape == (3, 1, 20, 30)


class TestClassPredictor:
    """Test class prediction head."""
    
    def test_class_predictor_init(self):
        """Test class predictor initialization."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        predictor = ClassPredictor(hidden_dim=256)
        assert predictor.hidden_dim == 256
    
    def test_class_predictor_forward(self):
        """Test class prediction."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        predictor = ClassPredictor(hidden_dim=256)
        
        instance_queries = torch.randn(10, 256)  # 10 instances
        semantic_queries = torch.randn(41, 256)  # 40 classes + background
        
        class_logits = predictor(instance_queries, semantic_queries)
        
        assert class_logits.shape == (10, 41)  # [I, C+1]
    
    def test_class_predictor_single_instance(self):
        """Test class prediction with single instance."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        predictor = ClassPredictor(hidden_dim=256)
        
        instance_queries = torch.randn(1, 256)
        semantic_queries = torch.randn(21, 256)  # 20 classes + background
        
        class_logits = predictor(instance_queries, semantic_queries)
        
        assert class_logits.shape == (1, 21)


class TestBuildDecoder:
    """Test decoder builder function."""
    
    def test_build_decoder_default(self):
        """Test building decoder with default config."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        decoder = build_decoder()
        
        assert decoder.hidden_dim == 256
        assert decoder.num_layers == 6
    
    def test_build_decoder_custom(self):
        """Test building decoder with custom config."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        config = {
            'hidden_dim': 128,
            'num_layers': 4,
            'num_heads': 4,
            'dim_feedforward': 1024,
            'dropout': 0.2
        }
        
        decoder = build_decoder(config)
        
        assert decoder.hidden_dim == 128
        assert decoder.num_layers == 4
        assert len(decoder.layers) == 4


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_decoder_to_masks(self):
        """Test full pipeline: decoder -> masks."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        # Create components
        decoder = TransformerDecoder(hidden_dim=256, num_layers=2)
        mask_predictor = MaskPredictor(hidden_dim=256)
        
        # Inputs
        queries = torch.randn(10, 256)
        features = torch.randn(5, 256, 30, 40)
        
        # Forward pass
        refined_queries, _ = decoder(queries, features)
        mask_logits = mask_predictor(refined_queries, features)
        
        assert refined_queries.shape == (10, 256)
        assert mask_logits.shape == (5, 10, 30, 40)
    
    def test_decoder_to_masks_and_classes(self):
        """Test full pipeline: decoder -> masks + classes."""
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        # Create components
        decoder = TransformerDecoder(hidden_dim=256, num_layers=2)
        mask_predictor = MaskPredictor(hidden_dim=256)
        class_predictor = ClassPredictor(hidden_dim=256)
        
        # Inputs
        semantic_queries = torch.randn(40, 256)  # 40 classes
        instance_queries = torch.randn(10, 256)  # 10 instances
        bg_query = torch.randn(1, 256)  # Background
        
        queries = torch.cat([semantic_queries, instance_queries, bg_query], dim=0)  # [51, 256]
        features = torch.randn(5, 256, 30, 40)
        
        # Forward pass
        refined_queries, _ = decoder(queries, features)
        
        # Split queries
        refined_semantic = refined_queries[:40]
        refined_instance = refined_queries[40:50]
        refined_bg = refined_queries[50:]
        
        # Predict masks (only for instances)
        mask_logits = mask_predictor(refined_instance, features)
        
        # Predict classes
        semantic_with_bg = torch.cat([refined_semantic, refined_bg], dim=0)  # [41, 256]
        class_logits = class_predictor(refined_instance, semantic_with_bg)
        
        assert mask_logits.shape == (5, 10, 30, 40)
        assert class_logits.shape == (10, 41)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])