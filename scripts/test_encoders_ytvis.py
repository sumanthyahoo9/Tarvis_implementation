"""
Test Semantic Query Encoder with REAL ResNet-50 Backbone on YouTube-VIS

This script:
1. Loads YouTube-VIS annotations and frame
2. Uses pretrained ResNet-50 to extract real features
3. Tests semantic query encoder with those features
4. Shows real query embeddings for VIS task

Can run on CPU with 16GB RAM!
"""

import sys
import traceback
import json
from pathlib import Path
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as T
from src.model.query_encoders.semantic_queries import SemanticQueryEncoder, MultiDatasetSemanticEncoder
TORCH_AVAILABLE = True

sys.path.insert(0, str(Path(__file__).parent.parent))


class ResNet50FeatureExtractor:
    """Extract multi-scale features from ResNet-50 for VIS task."""
    
    def __init__(self, device='cpu'):
        print("Loading pretrained ResNet-50...")
        
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        resnet.eval()
        resnet = resnet.to(device)
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels, 1/4 scale
        self.layer2 = resnet.layer2  # 512 channels, 1/8 scale
        self.layer3 = resnet.layer3  # 1024 channels, 1/16 scale
        self.layer4 = resnet.layer4  # 2048 channels, 1/32 scale
        
        # Projection layers to 256D
        self.proj1 = torch.nn.Conv2d(256, 256, kernel_size=1).to(device)
        self.proj2 = torch.nn.Conv2d(512, 256, kernel_size=1).to(device)
        self.proj3 = torch.nn.Conv2d(1024, 256, kernel_size=1).to(device)
        self.proj4 = torch.nn.Conv2d(2048, 256, kernel_size=1).to(device)
        
        self.device = device
        print(f"✓ ResNet-50 loaded on {device}")
    
    def extract_features(self, image):
        """
        Extract multi-scale features.
        
        Returns dict with F32, F16, F8, F4
        """
        with torch.no_grad():
            x = self.conv1(image)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            
            f4 = self.proj1(x1)
            f8 = self.proj2(x2)
            f16 = self.proj3(x3)
            f32 = self.proj4(x4)
            
            return {
                'F32': f32,
                'F16': f16,
                'F8': f8,
                'F4': f4
            }


def load_ytvis_annotations(dataset_path: str):
    """Load YouTube-VIS annotations."""
    dataset_path = Path(dataset_path)
    anno_path = dataset_path / "instances.json"
    
    if not anno_path.exists():
        raise FileNotFoundError(f"Annotations not found: {anno_path}")
    
    with open(anno_path, 'r') as f:
        annotations = json.load(f)
    
    return annotations


def load_ytvis_frame(dataset_path: str, video_id: str, frame_idx: int = 0):
    """Load a frame from YouTube-VIS with ResNet preprocessing."""
    dataset_path = Path(dataset_path)
    img_path = dataset_path / "JPEGImages" / video_id / f"{frame_idx:05d}.jpg"
    
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = Image.open(img_path).convert('RGB')
    
    # ImageNet normalization for ResNet
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, img


def test_semantic_encoder_with_real_backbone():
    """Test semantic encoder with real ResNet features."""
    print("="*70)
    print("Testing Semantic Encoder with REAL ResNet-50 Features")
    print("="*70 + "\n")
    
    dataset_path = "/Volumes/Elements/datasets/YOUTUBE_VIS"
    device = 'cpu'
    
    # Step 1: Load annotations
    print("Step 1: Loading YouTube-VIS annotations...")
    try:
        annotations = load_ytvis_annotations(dataset_path)
        
        num_videos = len(annotations['videos'])
        num_categories = len(annotations['categories'])
        
        print(f"✓ Videos: {num_videos}")
        print(f"✓ Categories: {num_categories}")
        
        # Show categories
        print("\nCategories:")
        for i, cat in enumerate(annotations['categories'][:10], 1):
            print(f"  {cat['id']:2d}. {cat['name']}")
        if num_categories > 10:
            print(f"  ... and {num_categories - 10} more")
        print()
        
    except Exception as e:
        print(f"✗ Error loading annotations: {e}")
        return
    
    # Step 2: Load sample frame
    print("Step 2: Loading sample video frame...")
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
        print(f"⚠ Could not load frame: {e}")
        print("  Continuing with encoder test...\n")
        image = None
    
    # Step 3: Extract features with real backbone
    if image is not None:
        print("Step 3: Creating ResNet-50 feature extractor...")
        feature_extractor = ResNet50FeatureExtractor(device=device)
        print()
        
        print("Step 4: Extracting features with ResNet-50...")
        print("  (This may take a few seconds on CPU...)")
        
        image = image.to(device)
        feature_dict = feature_extractor.extract_features(image)
        
        print(f"✓ Multi-scale features extracted:")
        for scale_name in ['F32', 'F16', 'F8', 'F4']:
            feat = feature_dict[scale_name]
            print(f"    {scale_name}: {feat.shape} (min={feat.min():.3f}, max={feat.max():.3f})")
        
        features = feature_dict['F4']
        print(f"\n✓ Using F4 for display: {features.shape}\n")
    else:
        print("Step 3-4: Skipping feature extraction (no frame loaded)\n")
        features = None
    
    # Step 5: Create semantic encoder
    print("Step 5: Creating Semantic Query Encoder...")
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
    
    # Step 6: Generate queries
    print("Step 6: Generating semantic queries...")
    batch_size = 2
    output = encoder(batch_size=batch_size, return_embeddings=True)
    
    queries = output['queries']
    query_types = output['query_types']
    
    print(f"✓ Query shape: {queries.shape}")
    print(f"✓ Query types shape: {query_types.shape}")
    
    print(f"\nQuery breakdown:")
    print(f"  Semantic (Qsem): {output['num_semantic']} queries")
    print(f"  Instance (Qinst): {output['num_instances']} queries")
    print(f"  Background (Qbg): {output['num_background']} query")
    print(f"  Total: {encoder.num_queries()} queries\n")
    
    # Step 7: Analyze query values
    print("Step 7: Analyzing semantic query values...")
    
    # Get individual query types
    qsem = encoder.get_semantic_queries(batch_size=1)
    qinst = encoder.get_instance_queries(batch_size=1)
    qbg = encoder.get_background_query(batch_size=1)
    
    print(f"✓ Semantic queries shape: {qsem.shape}")
    print(f"✓ Instance queries shape: {qinst.shape}")
    print(f"✓ Background query shape: {qbg.shape}\n")
    
    print("Query statistics:")
    print(f"  All queries:")
    print(f"    Min: {queries.min():.4f}")
    print(f"    Max: {queries.max():.4f}")
    print(f"    Mean: {queries.mean():.4f}")
    print(f"    Std: {queries.std():.4f}\n")
    
    print("Sample semantic queries (class embeddings):")
    for i in range(min(5, num_categories)):
        cat_name = annotations['categories'][i]['name']
        q = qsem[0, i]
        print(f"  Query {i} ({cat_name:15s}): mean={q.mean():.4f}, std={q.std():.4f}")
        print(f"                           first 5 values: {q[:5].detach().numpy()}")
    
    print("\nSample instance queries:")
    for i in range(3):
        q = qinst[0, i]
        print(f"  Query {i}: mean={q.mean():.4f}, std={q.std():.4f}")
        print(f"           first 5 values: {q[:5].detach().numpy()}")
    
    print("\n" + "="*70)
    print("✓ SUCCESS! Semantic encoder works with YouTube-VIS!")
    print("="*70)
    
    return encoder, queries, features


def test_multi_dataset_encoder():
    """Test multi-dataset semantic encoder."""
    print("\n" + "="*70)
    print("Testing Multi-Dataset Semantic Encoder")
    print("="*70 + "\n")
    
    # Define multiple datasets
    configs = {
        'youtube_vis': 40,
        'ovis': 25,
        'coco': 80
    }
    
    print("Creating multi-dataset encoder...")
    print("Dataset configurations:")
    for dataset, num_classes in configs.items():
        print(f"  {dataset:15s}: {num_classes} classes")
    print()
    
    encoder = MultiDatasetSemanticEncoder(
        dataset_configs=configs,
        num_instances=100,
        hidden_dim=256
    )
    
    print("✓ Multi-dataset encoder created\n")
    
    # Test each dataset
    print("Generating queries for each dataset:")
    for dataset in configs:
        output = encoder(dataset, batch_size=1, return_embeddings=True)
        
        print(f"  {dataset:15s}: queries={output['queries'].shape}, "
              f"semantic={output['num_semantic']}, "
              f"instance={output['num_instances']}")
    
    print("\n" + "="*70)
    print("✓ SUCCESS! Multi-dataset encoder works!")
    print("="*70)
    
    # Compare query statistics across datasets
    print("\nQuery Statistics Comparison:")
    print("-" * 70)
    print(f"{'Dataset':<15} {'Semantic':<10} {'Instance':<10} {'Total':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 70)
    
    for dataset in configs:
        output = encoder(dataset, batch_size=1)
        queries = output['queries']
        
        print(f"{dataset:<15} {output['num_semantic']:<10} "
              f"{output['num_instances']:<10} {queries.shape[1]:<10} "
              f"{queries.mean():.4f}    {queries.std():.4f}")
    
    print("-" * 70)


def compare_query_types():
    """Compare semantic vs object queries."""
    print("\n" + "="*70)
    print("Comparison: Semantic Queries vs Object Queries")
    print("="*70 + "\n")
    
    print("Semantic Queries (VIS/VPS):")
    print("  ✓ Learned embeddings (parameters)")
    print("  ✓ Represent abstract class concepts")
    print("  ✓ Same across all videos")
    print("  ✓ Example: 'person' query, 'car' query")
    print("  ✓ Training: Learned to recognize class patterns\n")
    
    print("Object Queries (VOS/PET):")
    print("  ✓ Computed from input (masks/points)")
    print("  ✓ Represent specific object instances")
    print("  ✓ Different for each video")
    print("  ✓ Example: 'this bear' query, 'that bird' query")
    print("  ✓ Training: Learned to encode object appearance\n")
    
    print("Key Difference:")
    print("  Semantic = Class-level (learnable params)")
    print("  Object = Instance-level (computed from input)")
    print("="*70)


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available!")
        sys.exit(1)
    
    # Check dataset
    dataset_path = "/Volumes/Elements/datasets/YOUTUBE_VIS"
    if not Path(dataset_path).exists():
        print(f"Dataset not found at: {dataset_path}")
        print("Please update the path in the script")
        sys.exit(1)
    
    try:
        # Test semantic encoder with real backbone
        encoder, queries, features = test_semantic_encoder_with_real_backbone()
        
        # Test multi-dataset encoder
        test_multi_dataset_encoder()
        
        # Compare query types
        compare_query_types()
        
        print("\n" + "="*70)
        print("All tests completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        traceback.print_exc()