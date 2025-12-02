"""
Tests for Semantic Query Encoder

Tests cover:
- Basic initialization
- Query generation
- Query splitting
- Multi-dataset handling
- Edge cases
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.model.query_encoders.semantic_queries import (
    SemanticQueryEncoder,
    MultiDatasetSemanticEncoder
)


class TestSemanticQueryEncoder:
    """Test cases for SemanticQueryEncoder."""
    
    def test_initialization(self):
        """Test basic encoder initialization."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256
        )
        
        assert encoder.num_classes == 40
        assert encoder.num_instances == 100
        assert encoder.hidden_dim == 256
        assert encoder.num_queries() == 141  # 40 + 100 + 1
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_query_generation(self):
        """Test query generation with different batch sizes."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256
        )
        
        # Batch size 1
        output = encoder(batch_size=1)
        assert output['queries'].shape == (1, 141, 256)
        assert output['query_types'].shape == (141,)
        assert output['num_semantic'] == 40
        assert output['num_instances'] == 100
        assert output['num_background'] == 1
        
        # Batch size 4
        output = encoder(batch_size=4)
        assert output['queries'].shape == (4, 141, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_query_types(self):
        """Test that query types are correctly assigned."""
        encoder = SemanticQueryEncoder(
            num_classes=10,
            num_instances=20,
            hidden_dim=128
        )
        
        output = encoder(batch_size=1)
        query_types = output['query_types']
        
        # First 10 should be semantic (0)
        assert (query_types[:10] == 0).all()
        
        # Next 20 should be instance (1)
        assert (query_types[10:30] == 1).all()
        
        # Last 1 should be background (2)
        assert (query_types[30:] == 2).all()
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_learned_embeddings(self):
        """Test learned embeddings for attention."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256,
            use_learned_embeddings=True
        )
        
        output = encoder(batch_size=2, return_embeddings=True)
        
        assert 'embeddings' in output
        assert output['embeddings'].shape == (2, 141, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_no_embeddings(self):
        """Test that embeddings are not returned when not requested."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256
        )
        
        output = encoder(batch_size=1, return_embeddings=False)
        assert 'embeddings' not in output
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_semantic_queries(self):
        """Test getting only semantic queries."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256
        )
        
        qsem = encoder.get_semantic_queries(batch_size=2)
        assert qsem.shape == (2, 40, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_instance_queries(self):
        """Test getting only instance queries."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256
        )
        
        qinst = encoder.get_instance_queries(batch_size=2)
        assert qinst.shape == (2, 100, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_background_query(self):
        """Test getting only background query."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256
        )
        
        qbg = encoder.get_background_query(batch_size=2)
        assert qbg.shape == (2, 1, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_split_output_queries(self):
        """Test splitting decoder output back into components."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256
        )
        
        # Simulate decoder output
        output_queries = torch.randn(2, 141, 256)
        
        qsem_out, qinst_out, qbg_out = encoder.split_output_queries(output_queries)
        
        assert qsem_out.shape == (2, 40, 256)
        assert qinst_out.shape == (2, 100, 256)
        assert qbg_out.shape == (2, 1, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        encoder = SemanticQueryEncoder(
            num_classes=10,
            num_instances=20,
            hidden_dim=128
        )
        
        # Check that weights are not all zeros
        assert not torch.allclose(
            encoder.semantic_queries,
            torch.zeros_like(encoder.semantic_queries)
        )
        assert not torch.allclose(
            encoder.instance_queries,
            torch.zeros_like(encoder.instance_queries)
        )
        assert not torch.allclose(
            encoder.background_query,
            torch.zeros_like(encoder.background_query)
        )
    
    def test_mock_mode(self):
        """Test that mock mode works when PyTorch is unavailable."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256
        )
        
        output = encoder(batch_size=2)
        
        # Should return mock data
        assert 'queries' in output
        assert 'query_types' in output
        assert output['num_semantic'] == 40
        assert output['num_instances'] == 100


class TestMultiDatasetSemanticEncoder:
    """Test cases for MultiDatasetSemanticEncoder."""
    
    def test_initialization(self):
        """Test multi-dataset encoder initialization."""
        configs = {
            'youtube_vis': 40,
            'ovis': 25,
            'coco': 80
        }
        
        encoder = MultiDatasetSemanticEncoder(
            dataset_configs=configs,
            num_instances=100,
            hidden_dim=256
        )
        
        assert len(encoder.dataset_configs) == 3
        assert encoder.num_instances == 100
        assert encoder.hidden_dim == 256
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_dataset_specific_queries(self):
        """Test that each dataset has correct number of semantic queries."""
        configs = {
            'youtube_vis': 40,
            'ovis': 25,
            'coco': 80
        }
        
        encoder = MultiDatasetSemanticEncoder(configs)
        
        # YouTube-VIS: 40 + 100 + 1 = 141 queries
        output = encoder('youtube_vis', batch_size=1)
        assert output['queries'].shape == (1, 141, 256)
        
        # OVIS: 25 + 100 + 1 = 126 queries
        output = encoder('ovis', batch_size=1)
        assert output['queries'].shape == (1, 126, 256)
        
        # COCO: 80 + 100 + 1 = 181 queries
        output = encoder('coco', batch_size=1)
        assert output['queries'].shape == (1, 181, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_shared_instance_queries(self):
        """Test that instance queries are shared across datasets."""
        configs = {
            'youtube_vis': 40,
            'ovis': 25
        }
        
        encoder = MultiDatasetSemanticEncoder(configs, num_instances=50)
        
        # Both should have 50 instance queries
        ytvis_output = encoder('youtube_vis', batch_size=1)
        ovis_output = encoder('ovis', batch_size=1)
        
        assert ytvis_output['num_instances'] == 50
        assert ovis_output['num_instances'] == 50
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_get_encoder(self):
        """Test getting individual dataset encoders."""
        configs = {
            'youtube_vis': 40,
            'ovis': 25
        }
        
        multi_encoder = MultiDatasetSemanticEncoder(configs)
        
        ytvis_encoder = multi_encoder.get_encoder('youtube_vis')
        assert ytvis_encoder is not None
        assert ytvis_encoder.num_classes == 40
        
        ovis_encoder = multi_encoder.get_encoder('ovis')
        assert ovis_encoder is not None
        assert ovis_encoder.num_classes == 25
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_invalid_dataset(self):
        """Test handling of invalid dataset name."""
        configs = {'youtube_vis': 40}
        encoder = MultiDatasetSemanticEncoder(configs)
        
        output = encoder('invalid_dataset', batch_size=1)
        assert output == {}


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_class(self):
        """Test encoder with single class."""
        encoder = SemanticQueryEncoder(
            num_classes=1,
            num_instances=10,
            hidden_dim=128
        )
        
        assert encoder.num_queries() == 12  # 1 + 10 + 1
    
    def test_single_instance(self):
        """Test encoder with single instance query."""
        encoder = SemanticQueryEncoder(
            num_classes=10,
            num_instances=1,
            hidden_dim=128
        )
        
        assert encoder.num_queries() == 12  # 10 + 1 + 1
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_large_batch(self):
        """Test with large batch size."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256
        )
        
        output = encoder(batch_size=32)
        assert output['queries'].shape == (32, 141, 256)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_different_hidden_dims(self):
        """Test with various hidden dimensions."""
        for dim in [128, 256, 512]:
            encoder = SemanticQueryEncoder(
                num_classes=40,
                num_instances=100,
                hidden_dim=dim
            )
            
            output = encoder(batch_size=1)
            assert output['queries'].shape == (1, 141, dim)
    
    def test_repr(self):
        """Test string representation."""
        encoder = SemanticQueryEncoder(
            num_classes=40,
            num_instances=100,
            hidden_dim=256
        )
        
        repr_str = repr(encoder)
        assert 'num_classes=40' in repr_str
        assert 'num_instances=100' in repr_str
        assert 'hidden_dim=256' in repr_str
        assert 'total_queries=141' in repr_str


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])