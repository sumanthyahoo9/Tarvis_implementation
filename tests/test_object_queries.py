"""
Tests for Object Query Encoder

Tests cover:
- Initialization from masks (VOS)
- Initialization from points (PET)
- Iterative refinement
- Background queries
- Edge cases
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.model.query_encoders.object_queries import (
    ObjectEncoder,
    ObjectEncoderLayer,
    ObjectQueryEncoder
)


class TestObjectEncoder:
    """Test cases for ObjectEncoder."""
    
    def test_initialization(self):
        """Test basic encoder initialization."""
        encoder = ObjectEncoder(
            hidden_dim=256,
            num_layers=3,
            num_heads=8,
            pmax=1024,
            queries_per_object=4
        )
        
        assert encoder.hidden_dim == 256
        assert encoder.num_layers == 3
        assert encoder.pmax == 1024
        assert encoder.queries_per_object == 4
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_vos_initialization_single_query(self):
        """Test VOS with single query per object."""
        encoder = ObjectEncoder(
            hidden_dim=256,
            queries_per_object=1
        )
        
        features = torch.randn(2, 256, 32, 32)
        masks = torch.rand(2, 3, 128, 128) > 0.5  # 3 objects
        
        queries = encoder(features, masks=masks.float())
        
        # Should have 3 queries (1 per object)
        assert queries.shape == (2, 3, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_vos_initialization_multiple_queries(self):
        """Test VOS with multiple queries per object."""
        encoder = ObjectEncoder(
            hidden_dim=256,
            queries_per_object=4
        )
        
        features = torch.randn(2, 256, 32, 32)
        masks = torch.rand(2, 3, 128, 128) > 0.5  # 3 objects
        
        queries = encoder(features, masks=masks.float())
        
        # Should have 12 queries (4 per object × 3 objects)
        assert queries.shape == (2, 12, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pet_initialization(self):
        """Test PET initialization from points."""
        encoder = ObjectEncoder(
            hidden_dim=256,
            queries_per_object=1  # PET uses 1 query per object
        )
        
        features = torch.randn(2, 256, 32, 32)
        points = torch.rand(2, 3, 2)  # 3 objects, (x, y) normalized [0, 1]
        
        queries = encoder(features, points=points)
        
        # Should have 3 queries (1 per point)
        assert queries.shape == (2, 3, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_empty_mask_handling(self):
        """Test handling of empty masks."""
        encoder = ObjectEncoder(hidden_dim=256)
        
        features = torch.randn(1, 256, 32, 32)
        masks = torch.zeros(1, 2, 128, 128)  # Empty masks
        
        # Should not crash
        queries = encoder(features, masks=masks)
        assert queries.shape == (1, 8, 256)  # 2 objects × 4 queries
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pmax_subsampling(self):
        """Test that large masks are subsampled to pmax."""
        encoder = ObjectEncoder(
            hidden_dim=256,
            pmax=100  # Small pmax for testing
        )
        
        features = torch.randn(1, 256, 64, 64)
        # Large mask (many pixels)
        masks = torch.ones(1, 1, 64, 64)
        
        queries = encoder(features, masks=masks)
        
        # Should not crash with large mask
        assert queries.shape == (1, 4, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_iterative_refinement(self):
        """Test that queries are refined through layers."""
        encoder = ObjectEncoder(
            hidden_dim=256,
            num_layers=3
        )
        
        features = torch.randn(2, 256, 32, 32)
        masks = torch.rand(2, 2, 128, 128) > 0.5
        
        # Hook to capture intermediate outputs
        intermediate_outputs = []
        
        def hook(module, input, output):
            intermediate_outputs.append(output.clone())
        
        # Register hooks on each layer
        for layer in encoder.layers:
            layer.register_forward_hook(hook)
        
        queries = encoder(features, masks=masks.float())
        
        # Should have 3 intermediate outputs (one per layer)
        assert len(intermediate_outputs) == 3
        
        # Queries should change through layers
        assert not torch.allclose(
            intermediate_outputs[0],
            intermediate_outputs[-1],
            atol=1e-5
        )
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_mutual_exclusivity(self):
        """Test that masks and points are mutually exclusive."""
        encoder = ObjectEncoder(hidden_dim=256)
        
        features = torch.randn(1, 256, 32, 32)
        masks = torch.rand(1, 2, 128, 128) > 0.5
        points = torch.rand(1, 2, 2)
        
        # Should raise assertion error
        with pytest.raises(AssertionError):
            encoder(features, masks=masks.float(), points=points)
    
    def test_mock_mode(self):
        """Test mock mode when PyTorch unavailable."""
        encoder = ObjectEncoder(hidden_dim=256)
        
        output = encoder(None, masks=True)
        
        # Should return mock list
        assert isinstance(output, list)


class TestObjectEncoderLayer:
    """Test cases for ObjectEncoderLayer."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_initialization(self):
        """Test layer initialization."""
        layer = ObjectEncoderLayer(
            hidden_dim=256,
            num_heads=8,
            dropout=0.1
        )
        
        assert layer.hidden_dim == 256
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_forward_pass(self):
        """Test forward pass through layer."""
        layer = ObjectEncoderLayer(hidden_dim=256)
        
        queries = torch.randn(2, 10, 256)
        features = torch.randn(2, 3, 100, 256)
        attn_mask = torch.ones(2, 3, 100)
        
        output = layer(queries, features, attn_mask)
        
        assert output.shape == (2, 10, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_attention_mask(self):
        """Test that attention mask is properly applied."""
        layer = ObjectEncoderLayer(hidden_dim=256)
        
        queries = torch.randn(1, 5, 256)
        features = torch.randn(1, 2, 50, 256)
        
        # Mask out second half of features
        attn_mask = torch.ones(1, 2, 50)
        attn_mask[:, :, 25:] = 0
        
        output = layer(queries, features, attn_mask)
        
        # Should not crash
        assert output.shape == (1, 5, 256)


class TestObjectQueryEncoder:
    """Test cases for complete ObjectQueryEncoder."""
    
    def test_initialization(self):
        """Test encoder initialization."""
        encoder = ObjectQueryEncoder(
            hidden_dim=256,
            num_bg_queries=16,
            queries_per_object=4
        )
        
        assert encoder.hidden_dim == 256
        assert encoder.num_bg_queries == 16
        assert encoder.queries_per_object == 4
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_vos_with_background(self):
        """Test VOS encoding with background queries."""
        encoder = ObjectQueryEncoder(
            hidden_dim=256,
            num_bg_queries=16,
            queries_per_object=4
        )
        
        features = torch.randn(2, 256, 32, 32)
        masks = torch.rand(2, 3, 128, 128) > 0.5  # 3 objects
        
        output = encoder(features, masks=masks.float(), batch_size=2)
        
        # 3 objects × 4 queries + 16 background = 28 total
        assert output['queries'].shape == (2, 28, 256)
        assert output['num_objects'] == 12
        assert output['num_background'] == 16
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pet_with_background(self):
        """Test PET encoding with background queries."""
        encoder = ObjectQueryEncoder(
            hidden_dim=256,
            num_bg_queries=16,
            queries_per_object=1
        )
        
        features = torch.randn(2, 256, 32, 32)
        points = torch.rand(2, 3, 2)  # 3 objects
        
        output = encoder(features, points=points, batch_size=2)
        
        # 3 objects × 1 query + 16 background = 19 total
        assert output['queries'].shape == (2, 19, 256)
        assert output['num_objects'] == 3
        assert output['num_background'] == 16
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_background_queries_learned(self):
        """Test that background queries are learned parameters."""
        encoder = ObjectQueryEncoder(
            hidden_dim=256,
            num_bg_queries=16
        )
        
        # Check that background queries are parameters
        assert isinstance(encoder.background_queries, torch.nn.Parameter)
        assert encoder.background_queries.shape == (16, 256)
        
        # Check they're not all zeros
        assert not torch.allclose(
            encoder.background_queries,
            torch.zeros_like(encoder.background_queries)
        )
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        encoder = ObjectQueryEncoder(hidden_dim=256)
        
        features = torch.randn(4, 256, 32, 32)
        masks = torch.rand(4, 2, 128, 128) > 0.5
        
        output = encoder(features, masks=masks.float(), batch_size=4)
        
        assert output['queries'].shape[0] == 4
    
    def test_mock_mode(self):
        """Test mock mode."""
        encoder = ObjectQueryEncoder(hidden_dim=256)
        
        output = encoder(None, masks=True, batch_size=1)
        
        assert 'queries' in output
        assert 'num_objects' in output
        assert 'num_background' in output


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_single_object(self):
        """Test with single object."""
        encoder = ObjectQueryEncoder(queries_per_object=4)
        
        features = torch.randn(1, 256, 32, 32)
        masks = torch.rand(1, 1, 128, 128) > 0.5  # 1 object
        
        output = encoder(features, masks=masks.float(), batch_size=1)
        
        # 1 object × 4 queries + 16 background = 20
        assert output['queries'].shape == (1, 20, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_many_objects(self):
        """Test with many objects."""
        encoder = ObjectQueryEncoder(queries_per_object=1)
        
        features = torch.randn(1, 256, 32, 32)
        masks = torch.rand(1, 10, 128, 128) > 0.5  # 10 objects
        
        output = encoder(features, masks=masks.float(), batch_size=1)
        
        # 10 objects × 1 query + 16 background = 26
        assert output['queries'].shape == (1, 26, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_different_hidden_dims(self):
        """Test with various hidden dimensions."""
        for dim in [128, 256, 512]:
            encoder = ObjectQueryEncoder(hidden_dim=dim)
            
            features = torch.randn(1, dim, 32, 32)
            masks = torch.rand(1, 2, 128, 128) > 0.5
            
            output = encoder(features, masks=masks.float(), batch_size=1)
            
            assert output['queries'].shape[2] == dim
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_point_coordinates_range(self):
        """Test that points outside [0, 1] are handled."""
        encoder = ObjectQueryEncoder(queries_per_object=1)
        
        features = torch.randn(1, 256, 32, 32)
        # Points slightly outside valid range
        points = torch.tensor([[[0.5, 0.5], [1.1, -0.1]]])
        
        # Should clip to valid range
        output = encoder(features, points=points, batch_size=1)
        
        assert output['queries'].shape == (1, 18, 256)


if __name__ == "__main__":
    pytest.main([__file__, '-v'])