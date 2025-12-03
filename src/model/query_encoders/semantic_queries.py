"""
Semantic Query Encoder for VIS/VPS Tasks

This module implements the query encoding mechanism for Video Instance Segmentation (VIS)
and Video Panoptic Segmentation (VPS) tasks. It generates three types of queries:
- Qsem: Semantic class queries (one per class)
- Qinst: Instance queries (upper bound on number of instances)
- Qbg: Background query (catch-all for non-objects)

These queries are learned embeddings that serve as abstract representations of
segmentation targets, making the architecture task-agnostic.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
TORCH_AVAILABLE = True
TensorType = torch.Tensor


class SemanticQueryEncoder(nn.Module):
    """
    Encodes segmentation targets as learned query embeddings for VIS/VPS tasks.
    
    The encoder creates three types of queries:
    1. Semantic queries (Qsem): One query per semantic class
    2. Instance queries (Qinst): Fixed number of queries for instances
    3. Background query (Qbg): Single query for background/inactive instances
    
    These queries are refined by the Transformer decoder through self-attention
    and cross-attention with video features.
    
    Args:
        num_classes: Number of semantic classes in the dataset
        num_instances: Maximum number of instances to detect
        hidden_dim: Dimensionality of query embeddings
        use_learned_embeddings: If True, use learned embeddings. If False, random init
    """
    
    def __init__(
        self,
        num_classes: int,
        num_instances: int = 100,
        hidden_dim: int = 256,
        use_learned_embeddings: bool = True
    ):
        if TORCH_AVAILABLE:
            super().__init__()
        
        self.num_classes = num_classes
        self.num_instances = num_instances
        self.hidden_dim = hidden_dim
        self.use_learned_embeddings = use_learned_embeddings
        
        if not TORCH_AVAILABLE:
            return
        
        # Semantic queries: one per class (e.g., 'person', 'car', 'dog')
        # Shape: [num_classes, hidden_dim]
        self.semantic_queries = nn.Parameter(
            torch.randn(num_classes, hidden_dim)
        )
        
        # Instance queries: fixed number to detect multiple instances
        # Shape: [num_instances, hidden_dim]
        self.instance_queries = nn.Parameter(
            torch.randn(num_instances, hidden_dim)
        )
        
        # Background query: catch-all for non-object pixels and inactive instances
        # Shape: [1, hidden_dim]
        self.background_query = nn.Parameter(
            torch.randn(1, hidden_dim)
        )
        
        # Optional: Learned query embeddings for multi-head attention
        # These are used as "keys" in the attention mechanism
        if use_learned_embeddings:
            self.semantic_embed = nn.Embedding(num_classes, hidden_dim)
            self.instance_embed = nn.Embedding(num_instances, hidden_dim)
            self.background_embed = nn.Embedding(1, hidden_dim)
        else:
            self.semantic_embed = None
            self.instance_embed = None
            self.background_embed = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize query parameters with Xavier uniform initialization."""
        if not TORCH_AVAILABLE:
            return
        
        nn.init.xavier_uniform_(self.semantic_queries)
        nn.init.xavier_uniform_(self.instance_queries)
        nn.init.xavier_uniform_(self.background_query)
        
        if self.use_learned_embeddings:
            nn.init.xavier_uniform_(self.semantic_embed.weight)
            nn.init.xavier_uniform_(self.instance_embed.weight)
            nn.init.xavier_uniform_(self.background_embed.weight)
    
    def forward(
        self,
        batch_size: int = 1,
        return_embeddings: bool = False
    ) -> Dict[str, Union[TensorType, int, List]]:
        """
        Generate query set for VIS/VPS tasks.
        
        Args:
            batch_size: Number of samples in batch
            return_embeddings: If True, also return learned embeddings for attention
            
        Returns:
            Dictionary containing:
                - 'queries': Concatenated [Qsem, Qinst, Qbg] queries
                  Shape: [batch_size, num_classes + num_instances + 1, hidden_dim]
                - 'query_types': Tensor indicating query type (0=sem, 1=inst, 2=bg)
                  Shape: [num_classes + num_instances + 1]
                - 'embeddings' (optional): Learned embeddings for attention
        """
        if not TORCH_AVAILABLE:
            # Mock output for CPU testing
            total_queries = self.num_classes + self.num_instances + 1
            return {
                'queries': [[0.0] * self.hidden_dim] * total_queries,
                'query_types': [0] * self.num_classes + [1] * self.num_instances + [2],
                'num_semantic': self.num_classes,
                'num_instances': self.num_instances,
                'num_background': 1
            }
        
        # Expand queries to batch size
        # Shape: [batch_size, num_classes, hidden_dim]
        qsem = self.semantic_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Shape: [batch_size, num_instances, hidden_dim]
        qinst = self.instance_queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Shape: [batch_size, 1, hidden_dim]
        qbg = self.background_query.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate all queries
        # Shape: [batch_size, num_classes + num_instances + 1, hidden_dim]
        queries = torch.cat([qsem, qinst, qbg], dim=1)
        
        # Create query type indicators
        # 0 = semantic, 1 = instance, 2 = background
        query_types = torch.cat([
            torch.zeros(self.num_classes, dtype=torch.long),
            torch.ones(self.num_instances, dtype=torch.long),
            torch.full((1,), 2, dtype=torch.long)
        ]).to(queries.device)
        
        result = {
            'queries': queries,
            'query_types': query_types,
            'num_semantic': self.num_classes,
            'num_instances': self.num_instances,
            'num_background': 1
        }
        
        # Add learned embeddings if requested (for multi-head attention)
        if return_embeddings and self.use_learned_embeddings:
            sem_embed = self.semantic_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            inst_embed = self.instance_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            bg_embed = self.background_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
            
            embeddings = torch.cat([sem_embed, inst_embed, bg_embed], dim=1)
            result['embeddings'] = embeddings
        
        return result
    
    def get_semantic_queries(self, batch_size: int = 1) -> Union[TensorType, List]:
        """Get only semantic queries (for VPS semantic segmentation)."""
        if not TORCH_AVAILABLE:
            return [[0.0] * self.hidden_dim] * self.num_classes
        
        return self.semantic_queries.unsqueeze(0).expand(batch_size, -1, -1)
    
    def get_instance_queries(self, batch_size: int = 1) -> Union[TensorType, List]:
        """Get only instance queries (for VIS instance segmentation)."""
        if not TORCH_AVAILABLE:
            return [[0.0] * self.hidden_dim] * self.num_instances
        
        return self.instance_queries.unsqueeze(0).expand(batch_size, -1, -1)
    
    def get_background_query(self, batch_size: int = 1) -> Union[TensorType, List]:
        """Get only background query."""
        if not TORCH_AVAILABLE:
            return [[0.0] * self.hidden_dim]
        
        return self.background_query.unsqueeze(0).expand(batch_size, -1, -1)
    
    def split_output_queries(
        self,
        output_queries: TensorType
    ) -> Tuple[Optional[TensorType], Optional[TensorType], Optional[TensorType]]:
        """
        Split decoder output queries back into semantic, instance, and background.
        
        Args:
            output_queries: Decoder output queries
                Shape: [batch_size, num_classes + num_instances + 1, hidden_dim]
                
        Returns:
            Tuple of (Qsem', Qinst', Qbg') where ' denotes refined queries
        """
        if not TORCH_AVAILABLE:
            return None, None, None
        
        qsem_out = output_queries[:, :self.num_classes, :]
        qinst_out = output_queries[:, self.num_classes:self.num_classes + self.num_instances, :]
        qbg_out = output_queries[:, -1:, :]
        
        return qsem_out, qinst_out, qbg_out
    
    def num_queries(self) -> int:
        """Return total number of queries."""
        return self.num_classes + self.num_instances + 1
    
    def __repr__(self) -> str:
        return (
            f"SemanticQueryEncoder(\n"
            f"  num_classes={self.num_classes},\n"
            f"  num_instances={self.num_instances},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  total_queries={self.num_queries()}\n"
            f")"
        )


class MultiDatasetSemanticEncoder(nn.Module):
    """
    Handles multiple datasets with different class sets for joint training.
    
    Each dataset has its own semantic queries, but instance and background
    queries are shared across all datasets.
    
    Args:
        dataset_configs: Dict mapping dataset name to num_classes
        num_instances: Shared number of instance queries
        hidden_dim: Query embedding dimension
        
    Example:
        configs = {
            'youtube_vis': 40,  # 40 classes
            'ovis': 25,         # 25 classes
            'coco': 80          # 80 classes
        }
        encoder = MultiDatasetSemanticEncoder(configs)
    """
    
    def __init__(
        self,
        dataset_configs: Dict[str, int],
        num_instances: int = 100,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.dataset_configs = dataset_configs
        self.num_instances = num_instances
        self.hidden_dim = hidden_dim
        
        if not TORCH_AVAILABLE:
            return
        
        # Create separate semantic queries for each dataset
        self.semantic_encoders = nn.ModuleDict()
        for dataset_name, num_classes in dataset_configs.items():
            self.semantic_encoders[dataset_name] = SemanticQueryEncoder(
                num_classes=num_classes,
                num_instances=num_instances,
                hidden_dim=hidden_dim
            )
    
    def forward(
        self,
        dataset_name: str,
        batch_size: int = 1,
        return_embeddings: bool = False
    ) -> Dict[str, Union[TensorType, int, List]]:
        """Get queries for specific dataset."""
        if not TORCH_AVAILABLE or dataset_name not in self.semantic_encoders:
            return {}
        
        return self.semantic_encoders[dataset_name](batch_size, return_embeddings)
    
    def get_encoder(self, dataset_name: str) -> SemanticQueryEncoder:
        """Get encoder for specific dataset."""
        if not TORCH_AVAILABLE:
            return None
        return self.semantic_encoders.get(dataset_name)


if __name__ == "__main__":
    # Example usage
    print("=== Semantic Query Encoder Demo ===\n")
    
    if TORCH_AVAILABLE:
        # Single dataset encoder
        encoder = SemanticQueryEncoder(
            num_classes=40,  # YouTube-VIS has 40 classes
            num_instances=100,
            hidden_dim=256
        )
        
        print(f"Encoder: {encoder}\n")
        
        # Generate queries
        output = encoder(batch_size=2, return_embeddings=True)
        print(f"Query shape: {output['queries'].shape}")
        print(f"Query types: {output['query_types'].shape}")
        print(f"Embeddings shape: {output['embeddings'].shape}\n")
        
        # Multi-dataset example
        configs = {
            'youtube_vis': 40,
            'ovis': 25,
            'coco': 80
        }
        multi_encoder = MultiDatasetSemanticEncoder(configs)
        
        for dataset in configs:
            output = multi_encoder(dataset, batch_size=1)
            print(f"{dataset}: {output['queries'].shape}")
    else:
        print("PyTorch not available")
        print("Install PyTorch to see full demo:")
        print("  pip install torch torchvision")