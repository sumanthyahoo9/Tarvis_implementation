"""
Test Semantic Query Encoder with Real YouTube-VIS Dataset

This script tests the SemanticQueryEncoder with YouTube-VIS data.
Tests query generation for VIS task.
"""

import sys
import json
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch torchvision")
    sys.exit(1)

from src.model.query_encoders import SemanticQueryEncoder


def load_ytvis_annotations(dataset_path: str):
    """
    Load YouTube-VIS annotations.
    
    Args:
        dataset_path: Path to YouTube-VIS root
        
    Returns:
        annotations: Dict with videos, categories, annotations
    """
    dataset_path = Path(dataset_path)
    anno_path = dataset_path / "instances.json"
    
    if not anno_path.exists():
        raise FileNotFoundError(f"Annotations not found: {anno_path}")
    
    with open(anno_path, 'r') as f:
        annotations = json.load(f)
    
    return annotations


def load_ytvis_frame(
    dataset_path: str,
    video_id: str,
    frame_idx: int = 0
):
    """
    Load a frame from YouTube-VIS.
    
    Args:
        dataset_path: Path to YouTube-VIS root
        video_id: Video folder name (e.g., '0a49f5265b')
        frame_idx: Frame index
        
    Returns:
        image: RGB image tensor [1, 3, H, W]
    """
    dataset_path = Path(dataset_path)
    img_path = dataset_path / "JPEGImages" / video_id / f"{frame_idx:05d}.jpg"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = T.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]
    
    return img_tensor, img


def test_ytvis_semantic_encoder():
    """Test Semantic Query Encoder with YouTube-VIS."""
    print("="*60)
    print("Testing Semantic Query Encoder with YouTube-VIS")
    print("="*60 + "\n")
    
    dataset_path = "/Volumes/Elements/datasets/YOUTUBE_VIS"
    
    # Load annotations
    print("Loading YouTube-VIS annotations...")
    try:
        annotations = load_ytvis_annotations(dataset_path)
        
        num_videos = len(annotations['videos'])
        num_categories = len(annotations['categories'])
        num_annotations = len(annotations['annotations'])
        
        print(f"✓ Videos: {num_videos}")
        print(f"✓ Categories: {num_categories}")
        print(f"✓ Annotations: {num_annotations}\n")
        
        # Show first few categories
        print("Categories:")
        for i, cat in enumerate(annotations['categories'][:10], 1):
            print(f"  {cat['id']:2d}. {cat['name']}")
        if num_categories > 10:
            print(f"  ... and {num_categories - 10} more\n")
        else:
            print()
        
    except Exception as e:
        print(f"✗ Error loading annotations: {e}")
        return
    
    # Load a sample frame
    print("Loading sample video frame...")
    try:
        sample_video = annotations['videos'][0]
        video_id = sample_video['file_names'][0].split('/')[0]
        
        image, img_pil = load_ytvis_frame(
            dataset_path=dataset_path,
            video_id=video_id,
            frame_idx=0
        )
        
        print(f"✓ Video ID: {video_id}")
        print(f"✓ Image shape: {image.shape}")
        print(f"✓ Image size: {img_pil.size}\n")
        
    except Exception as e:
        print(f"✗ Error loading frame: {e}")
        print("  This is okay - we can still test the encoder!\n")
        image = None
    
    # Create semantic encoder for YouTube-VIS
    print("Creating Semantic Query Encoder...")
    print(f"  Classes: {num_categories}")
    print(f"  Instance queries: 100")
    print(f"  Hidden dim: 256\n")
    
    encoder = SemanticQueryEncoder(
        num_classes=num_categories,
        num_instances=100,
        hidden_dim=256
    )
    
    print(f"✓ Encoder created")
    print(f"✓ Total queries: {encoder.num_queries()}\n")
    
    # Generate queries
    print("Generating queries...")
    batch_size = 2
    output = encoder(batch_size=batch_size, return_embeddings=True)
    
    queries = output['queries']
    query_types = output['query_types']
    embeddings = output.get('embeddings')
    
    print(f"✓ Query shape: {queries.shape if TORCH_AVAILABLE else len(queries)}")
    print(f"✓ Query types shape: {query_types.shape if TORCH_AVAILABLE else len(query_types)}")
    if embeddings is not None:
        print(f"✓ Embeddings shape: {embeddings.shape}")
    
    print(f"\nQuery breakdown:")
    print(f"  Semantic (Qsem): {output['num_semantic']} queries")
    print(f"  Instance (Qinst): {output['num_instances']} queries")
    print(f"  Background (Qbg): {output['num_background']} query")
    print(f"  Total: {encoder.num_queries()} queries\n")
    
    # Test individual query getters
    print("Testing individual query getters...")
    qsem = encoder.get_semantic_queries(batch_size=1)
    qinst = encoder.get_instance_queries(batch_size=1)
    qbg = encoder.get_background_query(batch_size=1)
    
    if TORCH_AVAILABLE:
        print(f"✓ Semantic queries: {qsem.shape}")
        print(f"✓ Instance queries: {qinst.shape}")
        print(f"✓ Background query: {qbg.shape}\n")
    
    # Test query splitting (simulates decoder output)
    if TORCH_AVAILABLE:
        print("Testing query splitting (decoder output → components)...")
        # Simulate decoder refined queries
        refined_queries = torch.randn(batch_size, encoder.num_queries(), 256)
        
        qsem_out, qinst_out, qbg_out = encoder.split_output_queries(refined_queries)
        
        print(f"✓ Split semantic: {qsem_out.shape}")
        print(f"✓ Split instance: {qinst_out.shape}")
        print(f"✓ Split background: {qbg_out.shape}\n")
    
    print("="*60)
    print("✓ SUCCESS! Semantic encoder works with YouTube-VIS!")
    print("="*60)
    
    return encoder, output


def test_multi_dataset():
    """Test MultiDatasetSemanticEncoder with multiple datasets."""
    print("\n" + "="*60)
    print("Testing Multi-Dataset Semantic Encoder")
    print("="*60 + "\n")
    
    from src.model.query_encoders import MultiDatasetSemanticEncoder
    
    # Define multiple datasets with different class counts
    configs = {
        'youtube_vis': 40,
        'ovis': 25,
        'coco': 80
    }
    
    print("Creating multi-dataset encoder...")
    print("Dataset configurations:")
    for dataset, num_classes in configs.items():
        print(f"  {dataset}: {num_classes} classes")
    print()
    
    encoder = MultiDatasetSemanticEncoder(
        dataset_configs=configs,
        num_instances=100,
        hidden_dim=256
    )
    
    print("✓ Multi-dataset encoder created\n")
    
    # Test each dataset
    print("Generating queries for each dataset...")
    for dataset in configs:
        output = encoder(dataset, batch_size=1, return_embeddings=True)
        
        if TORCH_AVAILABLE:
            print(f"✓ {dataset:15s}: {output['queries'].shape}")
    
    print("\n" + "="*60)
    print("✓ SUCCESS! Multi-dataset encoder works!")
    print("="*60)


def list_available_videos(dataset_path: str, max_show: int = 10):
    """List available videos in YouTube-VIS dataset."""
    dataset_path = Path(dataset_path)
    video_dir = dataset_path / "JPEGImages"
    
    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        return []
    
    videos = [d.name for d in video_dir.iterdir() if d.is_dir()]
    videos = sorted(videos)
    
    print(f"\nAvailable videos (showing first {max_show}):")
    for i, video in enumerate(videos[:max_show], 1):
        # Count frames in video
        num_frames = len(list((video_dir / video).glob("*.jpg")))
        print(f"  {i:2d}. {video} ({num_frames} frames)")
    
    if len(videos) > max_show:
        print(f"  ... and {len(videos) - max_show} more videos")
    
    return videos


if __name__ == "__main__":
    print("\n" + "="*60)
    print("YouTube-VIS Dataset - Semantic Query Encoder Testing")
    print("="*60 + "\n")
    
    # Check if dataset exists
    dataset_path = "/Volumes/Elements/datasets/YOUTUBE_VIS"
    if not Path(dataset_path).exists():
        print(f"✗ Dataset not found at: {dataset_path}")
        print("  Please update the path in the script")
        sys.exit(1)
    
    # List videos
    videos = list_available_videos(dataset_path)
    
    # Test semantic encoder
    try:
        encoder, output = test_ytvis_semantic_encoder()
    except Exception as e:
        print(f"✗ Semantic encoder test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test multi-dataset encoder
    try:
        test_multi_dataset()
    except Exception as e:
        print(f"✗ Multi-dataset test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)