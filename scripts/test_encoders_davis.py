"""
Test Object Query Encoder with Real DAVIS Dataset

This script loads real DAVIS data and tests the ObjectQueryEncoder.
Tests both mask loading and query generation.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch torchvision")
    sys.exit(1)

from src.model.query_encoders import ObjectQueryEncoder


def load_davis_frame(
    dataset_path: str,
    video_name: str = "bear",
    frame_idx: int = 0,
    resolution: str = "480p"
):
    """
    Load a frame and its mask from DAVIS dataset.
    
    Args:
        dataset_path: Path to DAVIS root directory
        video_name: Video name (e.g., 'bear', 'blackswan')
        frame_idx: Frame index
        resolution: '480p' or '1080p'
        
    Returns:
        image: RGB image tensor [1, 3, H, W]
        mask: Binary mask tensor [1, N, H, W] where N = number of objects
    """
    dataset_path = Path(dataset_path)
    
    # Construct paths
    img_path = dataset_path / "JPEGImages" / resolution / video_name / f"{frame_idx:05d}.jpg"
    mask_path = dataset_path / "Annotations" / resolution / video_name / f"{frame_idx:05d}.png"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_tensor = T.ToTensor()(img).unsqueeze(0)  # [1, 3, H, W]
    
    # Load mask
    mask = Image.open(mask_path)
    mask_np = np.array(mask)
    
    # DAVIS masks: 0=background, 1,2,3...=object IDs
    unique_ids = np.unique(mask_np)
    unique_ids = unique_ids[unique_ids > 0]  # Remove background
    
    # Create binary mask for each object
    masks = []
    for obj_id in unique_ids:
        obj_mask = (mask_np == obj_id).astype(np.float32)
        masks.append(obj_mask)
    
    if len(masks) == 0:
        print(f"Warning: No objects found in {mask_path}")
        # Create dummy mask
        masks = [np.zeros_like(mask_np, dtype=np.float32)]
    
    # Stack masks [N, H, W]
    masks = np.stack(masks, axis=0)
    mask_tensor = torch.from_numpy(masks).unsqueeze(0)  # [1, N, H, W]
    
    return img_tensor, mask_tensor, img, mask_np


def extract_image_features(image: torch.Tensor, feature_dim: int = 256):
    """
    Extract simple features from image for testing.
    
    In real TarViS, this would be done by the backbone (ResNet/Swin).
    For testing, we'll use a simple conv layer.
    
    Args:
        image: [B, 3, H, W]
        feature_dim: Output feature dimension
        
    Returns:
        features: [B, feature_dim, H//4, W//4]
    """
    # Simple feature extractor (replace with real backbone later)
    feature_extractor = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU()
    )
    
    with torch.no_grad():
        features = feature_extractor(image)
    
    return features


def test_davis_vos():
    """Test VOS (Video Object Segmentation) with DAVIS."""
    print("="*60)
    print("Testing Object Query Encoder with DAVIS Dataset (VOS)")
    print("="*60 + "\n")
    
    # Dataset path
    dataset_path = "/Volumes/Elements/datasets/DAVIS"
    
    # Load DAVIS frame
    print("Loading DAVIS frame...")
    try:
        image, masks, img_pil, mask_np = load_davis_frame(
            dataset_path=dataset_path,
            video_name="bear",
            frame_idx=0,
            resolution="480p"
        )
        
        print(f"✓ Image shape: {image.shape}")
        print(f"✓ Mask shape: {masks.shape}")
        print(f"✓ Number of objects: {masks.shape[1]}")
        print(f"✓ Image size: {img_pil.size}")
        print(f"✓ Unique mask values: {np.unique(mask_np)}\n")
        
    except FileNotFoundError as e:
        print(f"✗ Error loading DAVIS data: {e}")
        print("  Make sure the path is correct and data exists")
        return
    
    # Extract features (mock backbone)
    print("Extracting image features (mock backbone)...")
    features = extract_image_features(image, feature_dim=256)
    print(f"✓ Feature shape: {features.shape}\n")
    
    # Create encoder (VOS uses 4 queries per object)
    print("Creating Object Query Encoder (VOS mode: 4 queries/object)...")
    encoder = ObjectQueryEncoder(
        hidden_dim=256,
        num_bg_queries=16,
        queries_per_object=4,  # VOS
        encoder_layers=3
    )
    print(f"✓ Encoder created\n")
    
    # Encode objects
    print("Encoding objects into queries...")
    with torch.no_grad():
        output = encoder(features, masks=masks, batch_size=1)
    
    queries = output['queries']
    num_objects = output['num_objects']
    num_bg = output['num_background']
    
    print(f"✓ Query shape: {queries.shape}")
    print(f"✓ Object queries: {num_objects} ({masks.shape[1]} objects × 4 queries)")
    print(f"✓ Background queries: {num_bg}")
    print(f"✓ Total queries: {queries.shape[1]}\n")
    
    # Verify
    expected_total = masks.shape[1] * 4 + 16
    assert queries.shape[1] == expected_total, f"Expected {expected_total} queries, got {queries.shape[1]}"
    
    print("="*60)
    print("✓ SUCCESS! Object encoder works with real DAVIS data!")
    print("="*60)
    
    return queries


def test_davis_pet():
    """Test PET (Point Exemplar-guided Tracking) with DAVIS."""
    print("\n" + "="*60)
    print("Testing Object Query Encoder with DAVIS Dataset (PET)")
    print("="*60 + "\n")
    
    dataset_path = "/Volumes/Elements/datasets/DAVIS"
    
    # Load DAVIS frame
    print("Loading DAVIS frame...")
    image, masks, _, _ = load_davis_frame(
        dataset_path=dataset_path,
        video_name="bear",
        frame_idx=0,
        resolution="480p"
    )
    
    # Simulate points by taking mask centroids
    print("Computing object centroids as point annotations...")
    points = []
    for i in range(masks.shape[1]):
        mask = masks[0, i].numpy()
        if mask.sum() > 0:
            y_coords, x_coords = np.where(mask > 0)
            centroid_y = y_coords.mean() / mask.shape[0]  # Normalize to [0, 1]
            centroid_x = x_coords.mean() / mask.shape[1]
            points.append([centroid_x, centroid_y])
        else:
            points.append([0.5, 0.5])  # Default center
    
    points = torch.tensor(points).unsqueeze(0).float()  # [1, N, 2]
    print(f"✓ Points shape: {points.shape}")
    print(f"✓ Point coordinates (x, y normalized):\n{points[0]}\n")
    
    # Extract features
    features = extract_image_features(image, feature_dim=256)
    
    # Create encoder (PET uses 1 query per object)
    print("Creating Object Query Encoder (PET mode: 1 query/object)...")
    encoder = ObjectQueryEncoder(
        hidden_dim=256,
        num_bg_queries=16,
        queries_per_object=1,  # PET
        encoder_layers=3
    )
    
    # Encode objects from points
    print("Encoding objects from point coordinates...")
    with torch.no_grad():
        output = encoder(features, points=points, batch_size=1)
    
    queries = output['queries']
    num_objects = output['num_objects']
    num_bg = output['num_background']
    
    print(f"✓ Query shape: {queries.shape}")
    print(f"✓ Object queries: {num_objects} ({points.shape[1]} objects × 1 query)")
    print(f"✓ Background queries: {num_bg}")
    print(f"✓ Total queries: {queries.shape[1]}\n")
    
    # Verify
    expected_total = points.shape[1] * 1 + 16
    assert queries.shape[1] == expected_total
    
    print("="*60)
    print("✓ SUCCESS! Object encoder works with point tracking (PET)!")
    print("="*60)
    
    return queries


def list_available_videos(dataset_path: str, resolution: str = "480p"):
    """List all available videos in DAVIS dataset."""
    dataset_path = Path(dataset_path)
    video_dir = dataset_path / "JPEGImages" / resolution
    
    if not video_dir.exists():
        print(f"Directory not found: {video_dir}")
        return []
    
    videos = [d.name for d in video_dir.iterdir() if d.is_dir()]
    return sorted(videos)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("DAVIS Dataset - Object Query Encoder Testing")
    print("="*60 + "\n")
    
    # Check if dataset exists
    dataset_path = "/Volumes/Elements/datasets/DAVIS"
    if not Path(dataset_path).exists():
        print(f"✗ Dataset not found at: {dataset_path}")
        print("  Please update the path in the script")
        sys.exit(1)
    
    # List available videos
    print("Available videos in DAVIS:")
    videos = list_available_videos(dataset_path)
    for i, video in enumerate(videos[:10], 1):  # Show first 10
        print(f"  {i}. {video}")
    if len(videos) > 10:
        print(f"  ... and {len(videos) - 10} more")
    print()
    
    # Test VOS (mask-based)
    try:
        queries_vos = test_davis_vos()
    except Exception as e:
        print(f"✗ VOS test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test PET (point-based)
    try:
        queries_pet = test_davis_pet()
    except Exception as e:
        print(f"✗ PET test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)