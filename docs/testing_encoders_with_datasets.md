# Testing Query Encoders with Real Datasets

## Quick Start

You now have scripts to test the query encoders with your actual DAVIS and YouTube-VIS datasets!

### Prerequisites

```bash
# Make sure PyTorch is installed
pip install torch torchvision
pip install pillow numpy
```

### Test Object Encoder with DAVIS (VOS/PET)

```bash
cd /path/to/tarvis-implementation
python scripts/test_encoders_davis.py
```

**This will:**
- Load real DAVIS frames and masks
- Test VOS mode (4 queries per object from masks)
- Test PET mode (1 query per object from points)
- Verify query shapes are correct

**Expected Output:**
```
DAVIS Dataset - Object Query Encoder Testing
============================================================

Available videos in DAVIS:
  1. bear
  2. blackswan
  ... and more

Testing Object Query Encoder with DAVIS Dataset (VOS)
============================================================

Loading DAVIS frame...
✓ Image shape: torch.Size([1, 3, 480, 854])
✓ Mask shape: torch.Size([1, 1, 480, 854])
✓ Number of objects: 1
✓ Query shape: torch.Size([1, 20, 256])
✓ Object queries: 4 (1 objects × 4 queries)
✓ Background queries: 16
✓ Total queries: 20

✓ SUCCESS! Object encoder works with real DAVIS data!
```

### Test Semantic Encoder with YouTube-VIS (VIS)

```bash
python scripts/test_encoders_ytvis.py
```

**This will:**
- Load YouTube-VIS annotations (instances.json)
- Parse categories (40 classes)
- Test semantic query generation
- Test multi-dataset encoder

**Expected Output:**
```
YouTube-VIS Dataset - Semantic Query Encoder Testing
============================================================

Loading YouTube-VIS annotations...
✓ Videos: 2985
✓ Categories: 40
✓ Annotations: 8171

Categories:
   1. person
   2. giant_panda
   ... and 38 more

✓ Query shape: torch.Size([2, 141, 256])

Query breakdown:
  Semantic (Qsem): 40 queries
  Instance (Qinst): 100 queries
  Background (Qbg): 1 query
  Total: 141 queries

✓ SUCCESS! Semantic encoder works with YouTube-VIS!
```

## Dataset Paths

The scripts are configured for your dataset locations:

```python
# DAVIS
dataset_path = "/Volumes/Elements/datasets/DAVIS"

# YouTube-VIS
dataset_path = "/Volumes/Elements/datasets/YOUTUBE_VIS"
```

**To change paths:** Edit the `dataset_path` variable in each script.

## What Gets Tested

### DAVIS (Object Encoder)

**VOS Mode (Video Object Segmentation):**
1. Loads frame: `bear/00000.jpg`
2. Loads mask: `bear/00000.png`
3. Extracts features (mock backbone)
4. Encodes objects → 4 queries per object
5. Adds 16 background queries
6. Verifies shapes: `[batch, num_obj×4 + 16, 256]`

**PET Mode (Point Exemplar Tracking):**
1. Computes mask centroids as points
2. Encodes objects from points → 1 query per object
3. Adds 16 background queries
4. Verifies shapes: `[batch, num_obj×1 + 16, 256]`

### YouTube-VIS (Semantic Encoder)

1. Loads `instances.json` annotations
2. Parses 40 object categories
3. Creates semantic queries (40 classes)
4. Creates instance queries (100 instances)
5. Creates background query (1 query)
6. Tests multi-dataset encoder with different class counts

## Troubleshooting

### "Dataset not found"
Update the `dataset_path` in the scripts to match your actual location.

### "PyTorch not available"
```bash
pip install torch torchvision
```

### "Image/Mask not found"
The scripts default to:
- DAVIS: `bear` video, frame 0
- YouTube-VIS: First video in annotations

Change `video_name` or `video_id` in the script if these don't exist.

### "Module not found"
Make sure you're running from the project root:
```bash
cd /path/to/tarvis-implementation
python scripts/test_encoders_davis.py
```

## Understanding the Output

### Query Shapes

**DAVIS (Object Encoder):**
```
VOS: [batch, N_obj×4 + 16, 256]
PET: [batch, N_obj×1 + 16, 256]

where:
- N_obj = number of objects in frame
- 4/1 = queries per object
- 16 = background queries
- 256 = hidden dimension
```

**YouTube-VIS (Semantic Encoder):**
```
[batch, C + I + 1, 256]

where:
- C = number of classes (40 for YouTube-VIS)
- I = instance queries (100)
- 1 = background query
- 256 = hidden dimension
```

## Next Steps

Once these tests pass, you've verified:
✅ Object encoder works with real masks (DAVIS)
✅ Object encoder works with points (PET simulation)
✅ Semantic encoder works with real class counts (YouTube-VIS)

**Next:** Build the Temporal Neck to process video features!

## Modifying Tests

### Test Different Videos

```python
# DAVIS
load_davis_frame(
    dataset_path=dataset_path,
    video_name="blackswan",  # Change this
    frame_idx=10,            # Different frame
    resolution="480p"
)

# YouTube-VIS
load_ytvis_frame(
    dataset_path=dataset_path,
    video_id="0a59e18a8",   # Different video
    frame_idx=5              # Different frame
)
```

### Test with Real Backbone

Replace the mock feature extractor with ResNet or Swin:

```python
import torchvision.models as models

# Use ResNet-50
backbone = models.resnet50(pretrained=True)
backbone.eval()

# Extract features
with torch.no_grad():
    features = backbone.conv1(image)
    features = backbone.bn1(features)
    features = backbone.relu(features)
    features = backbone.layer1(features)
    features = backbone.layer2(features)
```

## File Structure

```
scripts/
├── test_encoders_davis.py   # Test object encoder with DAVIS
└── test_encoders_ytvis.py   # Test semantic encoder with YouTube-VIS

src/model/query_encoders/
├── semantic_queries.py       # Semantic encoder implementation
└── object_queries.py         # Object encoder implementation
```