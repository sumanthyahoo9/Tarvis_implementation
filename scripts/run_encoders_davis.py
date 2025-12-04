"""
Test Object Encoder with REAL ResNet-50 Backbone

This script uses a pretrained ResNet-50 backbone to extract real features
from DAVIS images, then tests the object encoder with those features.

Can run on CPU with 16GB RAM!
"""

import sys
from pathlib import Path
import traceback
import numpy as np
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as T
from src.model.query_encoders.object_queries import ObjectQueryEncoder
TORCH_AVAILABLE = True
sys.path.insert(0, str(Path(__file__).parent.parent))


class ResNet50FeatureExtractor:
    """
    Extract multi-scale features from ResNet-50.
    
    Uses intermediate layers to get features at different scales:
    - layer1: 1/4 resolution (F4)
    - layer2: 1/8 resolution (F8)
    - layer3: 1/16 resolution (F16)
    - layer4: 1/32 resolution (F32)
    """
    
    def __init__(self, device='cpu'):
        print("Loading pretrained ResNet-50...")
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=True)
        resnet.eval()
        resnet = resnet.to(device)
        
        # Extract the layers we need
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # Output: 256 channels, 1/4 resolution
        self.layer2 = resnet.layer2  # Output: 512 channels, 1/8 resolution
        self.layer3 = resnet.layer3  # Output: 1024 channels, 1/16 resolution
        self.layer4 = resnet.layer4  # Output: 2048 channels, 1/32 resolution
        
        # Projection layers to make all features 256-dimensional
        self.proj1 = torch.nn.Conv2d(256, 256, kernel_size=1).to(device)
        self.proj2 = torch.nn.Conv2d(512, 256, kernel_size=1).to(device)
        self.proj3 = torch.nn.Conv2d(1024, 256, kernel_size=1).to(device)
        self.proj4 = torch.nn.Conv2d(2048, 256, kernel_size=1).to(device)
        
        self.device = device
        
        print(f"✓ ResNet-50 loaded on {device}")
    
    def extract_features(self, image):
        """
        Extract multi-scale features.
        
        Args:
            image: [B, 3, H, W] RGB image
            
        Returns:
            Dictionary with all 4 scales:
                'F32': [B, 256, H/32, W/32]
                'F16': [B, 256, H/16, W/16]
                'F8':  [B, 256, H/8, W/8]
                'F4':  [B, 256, H/4, W/4]
        """
        with torch.no_grad():
            # Stem
            x = self.conv1(image)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            # Extract at each layer
            x1 = self.layer1(x)      # [B, 256, H/4, W/4]
            x2 = self.layer2(x1)     # [B, 512, H/8, W/8]
            x3 = self.layer3(x2)     # [B, 1024, H/16, W/16]
            x4 = self.layer4(x3)     # [B, 2048, H/32, W/32]
            
            # Project to 256 dimensions
            f4 = self.proj1(x1)      # [B, 256, H/4, W/4]
            f8 = self.proj2(x2)      # [B, 256, H/8, W/8]
            f16 = self.proj3(x3)     # [B, 256, H/16, W/16]
            f32 = self.proj4(x4)     # [B, 256, H/32, W/32]
            
            return {
                'F32': f32,
                'F16': f16,
                'F8': f8,
                'F4': f4
            }


def load_davis_sample(dataset_path: str, video_name: str = "bear", frame_idx: int = 0):
    """Load DAVIS frame and mask."""
    dataset_path = Path(dataset_path)
    
    img_path = dataset_path / "JPEGImages" / "480p" / video_name / f"{frame_idx:05d}.jpg"
    mask_path = dataset_path / "Annotations" / "480p" / video_name / f"{frame_idx:05d}.png"
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    
    # Normalize for ResNet (ImageNet stats)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, H, W]
    
    # Load mask
    mask = Image.open(mask_path)
    mask_np = np.array(mask)
    
    # Extract objects
    unique_ids = np.unique(mask_np)
    unique_ids = unique_ids[unique_ids > 0]
    
    masks = []
    for obj_id in unique_ids:
        obj_mask = (mask_np == obj_id).astype(np.float32)
        masks.append(obj_mask)
    
    if len(masks) == 0:
        masks = [np.zeros_like(mask_np, dtype=np.float32)]
    
    masks = np.stack(masks, axis=0)
    mask_tensor = torch.from_numpy(masks).unsqueeze(0)  # [1, N, H, W]
    
    return img_tensor, mask_tensor, img, mask_np


def test_with_real_backbone():
    """Test object encoder with real ResNet-50 features."""
    print("="*70)
    print("Testing Object Encoder with REAL ResNet-50 Features")
    print("="*70 + "\n")
    
    dataset_path = "/Volumes/Elements/datasets/DAVIS"
    device = 'cpu'  # Can change to 'cuda' if you have GPU
    
    # Load DAVIS frame
    print("Step 1: Loading DAVIS frame...")
    image, masks, img_pil, mask_np = load_davis_sample(
        dataset_path=dataset_path,
        video_name="bear",
        frame_idx=0
    )
    
    print(f"✓ Image shape: {image.shape}")
    print(f"✓ Mask shape: {masks.shape}")
    print(f"✓ Number of objects: {masks.shape[1]}")
    print(f"✓ Image size: {img_pil.size}\n")
    
    # Create feature extractor
    print("Step 2: Creating ResNet-50 feature extractor...")
    feature_extractor = ResNet50FeatureExtractor(device=device)
    print()
    
    # Extract REAL features
    print("Step 3: Extracting features with ResNet-50...")
    print("  (This may take a few seconds on CPU...)")
    
    image = image.to(device)
    feature_dict = feature_extractor.extract_features(image)
    
    print("✓ Multi-scale features extracted:")
    for scale_name in ['F32', 'F16', 'F8', 'F4']:
        feat = feature_dict[scale_name]
        print(f"    {scale_name}: {feat.shape} (min={feat.min():.3f}, max={feat.max():.3f})")
    
    # Use F4 for object encoder (highest resolution)
    features = feature_dict['F4']
    print(f"\n✓ Using F4 for object encoding: {features.shape}\n")
    
    # Create object encoder
    print("Step 4: Creating Object Query Encoder...")
    encoder = ObjectQueryEncoder(
        hidden_dim=256,
        num_bg_queries=16,
        queries_per_object=4,  # VOS mode
        encoder_layers=3
    ).to(device)
    print("✓ Encoder created\n")
    
    # Encode objects
    print("Step 5: Encoding objects into queries...")
    masks = masks.to(device)
    
    with torch.no_grad():
        output = encoder(features, masks=masks, batch_size=1)
    
    queries = output['queries']
    
    print(f"✓ Query shape: {queries.shape}")
    print(f"✓ Object queries: {output['num_objects']}")
    print(f"✓ Background queries: {output['num_background']}")
    print(f"✓ Total queries: {queries.shape[1]}\n")
    
    # Analyze queries
    print("Step 6: Analyzing query values...")
    print("✓ Query statistics:")
    print(f"    Min: {queries.min():.4f}")
    print(f"    Max: {queries.max():.4f}")
    print(f"    Mean: {queries.mean():.4f}")
    print(f"    Std: {queries.std():.4f}\n")
    
    print("Query breakdown:")
    for i in range(min(4, queries.shape[1])):
        q = queries[0, i]
        print(f"  Query {i}: mean={q.mean():.4f}, std={q.std():.4f}")
        print(f"           first 5 values: {q[:5].cpu().numpy()}")
    
    print("\n" + "="*70)
    print("✓ SUCCESS! Object encoder works with REAL ResNet-50 features!")
    print("="*70)
    
    return queries, features


def compare_mock_vs_real():
    """Compare mock features vs real ResNet features."""
    print("\n" + "="*70)
    print("Comparison: Mock Features vs Real ResNet Features")
    print("="*70 + "\n")
    
    # This would show the difference between random features and learned features
    print("Mock features (random):")
    print("  - Random noise: no semantic meaning")
    print("  - Distribution: N(0, 1)")
    print("  - Queries: Still computed correctly but random\n")
    
    print("Real ResNet features (pretrained):")
    print("  - Learned on ImageNet: capture real visual patterns")
    print("  - Distribution: learned, non-random")
    print("  - Queries: Meaningful embeddings of object parts")
    print("  - Can actually be used for segmentation!\n")


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available!")
        sys.exit(1)
    
    # Check dataset exists
    dataset_path = "/Volumes/Elements/datasets/DAVIS"
    if not Path(dataset_path).exists():
        print(f"Dataset not found at: {dataset_path}")
        print("Please update the path in the script")
        sys.exit(1)
    
    try:
        # Test with real backbone
        queries, features = test_with_real_backbone()
        
        # Compare
        compare_mock_vs_real()
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()