# Query Encoders - Implementation Guide

## Overview

Query encoders are THE core innovation of TarViS. They transform task-specific segmentation targets into abstract query embeddings that the Transformer decoder can process uniformly.

## Two Encoder Types

### 1. Semantic Query Encoder (VIS/VPS)
**Used by:** YouTube-VIS, OVIS, KITTI-STEP, Cityscapes-VPS, VIPSeg, COCO, ADE20k

**Generates three query types:**
- **Qsem** (Semantic queries): One learned embedding per class (e.g., 'person', 'car')
- **Qinst** (Instance queries): Fixed number of learned embeddings for instances
- **Qbg** (Background query): Single learned embedding for non-objects

**Key insight:** These are pure learned parameters (like word embeddings in NLP). The model learns what "person-ness" or "car-ness" means through training.

**Example:**
```python
encoder = SemanticQueryEncoder(
    num_classes=40,      # YouTube-VIS has 40 classes
    num_instances=100,   # Can detect up to 100 instances
    hidden_dim=256
)

output = encoder(batch_size=2)
# output['queries']: [2, 141, 256]  # 40 + 100 + 1 = 141 total queries
```

### 2. Object Query Encoder (VOS/PET)
**Used by:** DAVIS, BURST

**Generates two query types:**
- **Qobj** (Object queries): Dynamically encoded from masks/points
- **Qbg** (Background queries): Multiple learned embeddings for non-targets

**Key insight:** Unlike semantic queries, these are COMPUTED from the input (mask or point) through iterative refinement with attention.

**Iterative Refinement Process:**
1. **Initialize:** Pool features inside mask (VOS) or extract at point (PET)
2. **Refine:** Apply 3 layers of:
   - Self-attention: Queries attend to each other
   - Cross-attention: Queries attend to (masked) image features
   - FFN: Non-linear transformation
3. **Output:** Refined query embeddings

**Example:**
```python
# VOS
encoder_vos = ObjectQueryEncoder(queries_per_object=4)
output = encoder_vos(features, masks=masks, batch_size=2)
# output['queries']: [2, 28, 256]  # 3 objects × 4 + 16 bg = 28

# PET  
encoder_pet = ObjectQueryEncoder(queries_per_object=1)
output = encoder_pet(features, points=points, batch_size=2)
# output['queries']: [2, 19, 256]  # 3 objects × 1 + 16 bg = 19
```

## Why This Design?

**Problem:** Different tasks define targets differently:
- VIS: "Find all people in video"
- VOS: "Track THIS specific object" (given mask)
- PET: "Track object at THIS point"

**Solution:** Abstract targets as queries, decouple task definition from architecture!

**The magic:** The Transformer decoder doesn't know if queries came from:
- Learned embeddings (VIS/VPS)
- Mask encoding (VOS)
- Point encoding (PET)

It just refines queries → predicts masks. Architecture is task-agnostic!

## Key Differences from Prior Work

**Compared to Mask2Former (VIS):**
- Mask2Former: Instance queries → FC layer → C+1 classes (baked-in classification)
- TarViS: Semantic queries + Instance queries → inner product → classes (flexible)

**Compared to STM (VOS):**
- STM: Pixel-to-pixel correspondence (preserves fine detail but task-specific)
- TarViS: Object-to-query encoding (loses some detail but task-agnostic)

**Compared to PET baseline:**
- Baseline: Point → pseudo-mask → VOS method (two-stage)
- TarViS: Point → query directly (end-to-end)

## Implementation Details

### Semantic Query Encoder
**File:** `src/model/query_encoders/semantic_queries.py`

**Key components:**
```python
# Learned parameters
self.semantic_queries = nn.Parameter(torch.randn(C, D))  # C classes
self.instance_queries = nn.Parameter(torch.randn(I, D))  # I instances
self.background_query = nn.Parameter(torch.randn(1, D))  # 1 background

# Optional: Key embeddings for attention
self.semantic_embed = nn.Embedding(C, D)  # Used in attention K/V
```

**Forward pass:** Just expand to batch size and concatenate!

### Object Query Encoder
**File:** `src/model/query_encoders/object_queries.py`

**Key components:**
```python
# Object encoder (iterative refinement)
self.object_encoder = ObjectEncoder(
    num_layers=3,      # 3 refinement layers
    num_heads=8,       # Multi-head attention
    pmax=1024          # Max feature points per object
)

# Background queries (learned)
self.background_queries = nn.Parameter(torch.randn(B, D))  # B bg queries
```

**Forward pass:**
1. Initialize queries from mask/point
2. Refine through attention layers
3. Concatenate with background queries

### Memory Optimization: pmax

**Problem:** VOS masks can be huge (128×128 = 16k pixels)
**Solution:** Subsample to pmax=1024 points per object

This is TarViS's key optimization over HODOR:
- HODOR: Soft-masked attention over entire H×W image
- TarViS: Hard-masked attention over ≤pmax points

**Result:** 16× memory savings with minimal accuracy loss!

## Multi-Dataset Training

For joint training on datasets with different class counts:

```python
configs = {
    'youtube_vis': 40,
    'ovis': 25,
    'coco': 80
}

encoder = MultiDatasetSemanticEncoder(configs)

# Use dataset-specific semantic queries
output = encoder('youtube_vis', batch_size=2)  # 40 semantic queries
output = encoder('coco', batch_size=2)         # 80 semantic queries

# But instance/background queries are SHARED across datasets!
```

## Testing Your Implementation

**With your datasets:**

```python
# Test with DAVIS (VOS)
from your_data_loader import load_davis_sample

features, mask = load_davis_sample()  # First frame
encoder = ObjectQueryEncoder(queries_per_object=4)
queries = encoder(features, masks=mask)

# Test with YouTube-VIS (VIS)
from your_data_loader import load_ytvis_sample

encoder = SemanticQueryEncoder(num_classes=40)
queries = encoder(batch_size=1)
```
## Quick Reference

**Semantic Queries:**
- Learned parameters
- Task: VIS/VPS
- Output: [B, C+I+1, D]

**Object Queries:**
- Computed from input
- Task: VOS/PET
- Output: [B, O×qo+B, D]

**Common params:**
- D (hidden_dim): 256
- I (num_instances): 100
- B (num_bg_queries): 16
- qo (queries_per_object): 4 for VOS, 1 for PET