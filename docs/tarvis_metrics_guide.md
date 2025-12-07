# TarViS Evaluation Metrics - Complete Guide

## Overview

TarViS uses different metrics for each task:
- **VOS (DAVIS):** J&F Score (Jaccard + F-measure)
- **VIS (YouTube-VIS):** mAP (mean Average Precision)
- **VPS (VIPSeg):** STQ (Segmentation & Tracking Quality)

---

## 1. J&F Score (VOS)

### **What it measures:** Quality of object masks in videos

### **Components:**

#### **J (Jaccard Index / IoU)**
Measures spatial overlap between predicted and ground truth masks.

```
J = Intersection / Union = |A ∩ B| / |A ∪ B|
```

- **J = 1.0:** Perfect overlap
- **J = 0.5:** Half overlap
- **J = 0.0:** No overlap

#### **F (F-measure)**
Measures boundary accuracy (contour precision).

```
F = 2 * (Precision * Recall) / (Precision + Recall)
```

- Compares predicted boundaries vs ground truth boundaries
- **F = 1.0:** Perfect boundary match
- **F = 0.0:** No boundary overlap

#### **J&F**
Final score is the mean: `J&F = (J + F) / 2`

### **Usage Example:**

```python
from src.utils.evaluation_metrics import compute_j_and_f

# Predictions: [T, O, H, W] - T frames, O objects
pred_masks = np.random.rand(10, 2, 480, 854) > 0.5
gt_masks = np.random.rand(10, 2, 480, 854) > 0.5

scores = compute_j_and_f(pred_masks, gt_masks)
print(f"J (IoU): {scores['J']:.4f}")
print(f"F (Boundary): {scores['F']:.4f}")
print(f"J&F: {scores['J&F']:.4f}")
```

### **DAVIS Benchmark Results:**

| Method | J&F | J | F |
|--------|-----|---|---|
| TarViS (paper) | 86.5 | 83.3 | 89.7 |
| Good baseline | 70+ | 67+ | 73+ |
| State-of-art | 85+ | 82+ | 88+ |

---

## 2. mAP (VIS)

### **What it measures:** Instance detection + segmentation accuracy

### **How it works:**

1. **Match predictions to ground truth** using IoU
2. **Rank predictions** by confidence score
3. **Compute Precision-Recall curve** for each class
4. **Average Precision (AP)** = Area under PR curve
5. **mAP** = Mean AP across all classes

### **IoU Thresholds:**
- **AP@0.5:** IoU ≥ 0.5 required for match
- **AP@0.75:** IoU ≥ 0.75 (stricter)
- **mAP:** Average across thresholds [0.5:0.05:0.95]

### **Usage Example:**

```python
from src.utils.evaluation_metrics import compute_map

predictions = [{
    'masks': np.random.rand(5, 10, 480, 854) > 0.5,  # 5 instances, 10 frames
    'classes': np.array([0, 1, 2, 3, 4]),
    'scores': np.array([0.95, 0.90, 0.85, 0.80, 0.75])
}]

ground_truths = [{
    'masks': np.random.rand(3, 10, 480, 854) > 0.5,  # 3 GT instances
    'classes': np.array([0, 1, 2])
}]

scores = compute_map(
    pred_masks_list=[p['masks'] for p in predictions],
    pred_classes_list=[p['classes'] for p in predictions],
    pred_scores_list=[p['scores'] for p in predictions],
    gt_masks_list=[g['masks'] for g in ground_truths],
    gt_classes_list=[g['classes'] for g in ground_truths],
    num_classes=40
)

print(f"mAP: {scores['mAP']:.4f}")
print(f"AP@0.5: {scores['AP@0.50']:.4f}")
print(f"AP@0.75: {scores['AP@0.75']:.4f}")
```

### **YouTube-VIS Benchmark Results:**

| Method | mAP | AP@50 | AP@75 |
|--------|-----|-------|-------|
| TarViS (paper) | 59.8 | 82.1 | 66.7 |
| Good baseline | 40+ | 60+ | 45+ |
| State-of-art | 55+ | 80+ | 60+ |

---

## 3. STQ (VPS)

### **What it measures:** Panoptic segmentation + tracking quality

### **Components:**

#### **SQ (Segmentation Quality)**
Average IoU of matched instances.

```
SQ = (1/|TP|) * Σ IoU(pred, gt)
```

#### **AQ (Association Quality)**
Tracking consistency across frames (IDF1-style).

```
AQ = Correct_Associations / (Total_GT + Total_Pred)
```

#### **STQ**
Geometric mean of SQ and AQ:

```
STQ = √(SQ × AQ)
```

### **Usage Example:**

```python
from src.utils.evaluation_metrics import compute_stq

# Predictions: [T, N, H, W] - T frames, N instances
pred_masks = np.random.rand(10, 5, 480, 854) > 0.5
pred_classes = np.random.randint(0, 124, (10, 5))

gt_masks = np.random.rand(10, 3, 480, 854) > 0.5
gt_classes = np.random.randint(0, 124, (10, 3))

scores = compute_stq(pred_masks, pred_classes, gt_masks, gt_classes)
print(f"STQ: {scores['STQ']:.4f}")
print(f"SQ (Segmentation): {scores['SQ']:.4f}")
print(f"AQ (Association): {scores['AQ']:.4f}")
```

### **VIPSeg Benchmark Results:**

| Method | STQ | AQ | SQ |
|--------|-----|----|----|
| TarViS (paper) | 55.5 | 52.8 | 58.4 |
| Good baseline | 40+ | 37+ | 43+ |
| State-of-art | 53+ | 50+ | 56+ |

---

## Unified Metrics Interface

### **EvaluationMetrics Class**

Simplifies evaluation across all tasks:

```python
from src.utils.evaluation_metrics import EvaluationMetrics

metrics = EvaluationMetrics()

# VOS
j_f_scores = metrics.evaluate_vos(pred_masks, gt_masks)

# VIS
map_scores = metrics.evaluate_vis(predictions, ground_truths, num_classes=40)

# VPS
stq_scores = metrics.evaluate_vps(pred_masks, pred_classes, gt_masks, gt_classes)
```

---

## Implementation Details

### **IoU Computation**

```python
def compute_iou(pred_mask, gt_mask):
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return intersection / union if union > 0 else 1.0
```

**Edge cases:**
- Both empty → IoU = 1.0 (perfect match)
- Union = 0 → IoU = 1.0
- No overlap → IoU = 0.0

### **F-measure Computation**

```python
def compute_f_measure(pred_mask, gt_mask):
    # Extract boundaries
    pred_boundary = extract_boundary(pred_mask)
    gt_boundary = extract_boundary(gt_mask)
    
    # Distance transforms
    pred_dists = distance_transform(pred_boundary)
    gt_dists = distance_transform(gt_boundary)
    
    # Precision: pred boundary close to gt
    precision = (pred_dists < threshold).sum() / pred_boundary.sum()
    
    # Recall: gt boundary close to pred
    recall = (gt_dists < threshold).sum() / gt_boundary.sum()
    
    # F-measure
    return 2 * precision * recall / (precision + recall)
```

**Boundary threshold:** 0.8% of image diagonal (default)

### **mAP Computation**

```python
def compute_map(predictions, ground_truths):
    # For each IoU threshold
    for iou_thresh in [0.5, 0.55, ..., 0.95]:
        # For each class
        for cls in classes:
            # Match predictions to GTs
            matches = match_by_iou(preds, gts, iou_thresh)
            
            # Compute Precision-Recall
            precisions, recalls = compute_pr_curve(matches)
            
            # Compute AP (area under PR curve)
            ap = integrate_pr_curve(precisions, recalls)
        
        # Average across classes
        mean_ap = np.mean(class_aps)
    
    # Average across IoU thresholds
    return np.mean(threshold_maps)
```

**Matching strategy:** Greedy (highest IoU first)

### **STQ Computation**

```python
def compute_stq(pred_masks, pred_classes, gt_masks, gt_classes):
    # Match instances in each frame
    for frame in frames:
        tp, fp, fn = match_instances(pred, gt, iou_thresh=0.5)
        
        # Accumulate IoU (for SQ)
        total_iou += sum([iou for _, _, iou in tp])
        
        # Track associations (for AQ)
        track_associations(tp)
    
    # Segmentation Quality
    SQ = total_iou / num_matches
    
    # Association Quality
    AQ = correct_associations / (total_gt + total_pred)
    
    # STQ
    return sqrt(SQ * AQ)
```

---

## Testing

### **Run all metric tests:**

```bash
pytest tests/test_evaluation_metrics.py -v
```

### **Test coverage:**

- ✅ IoU computation (4 tests)
- ✅ F-measure computation (2 tests)
- ✅ J&F score (3 tests)
- ✅ Average Precision (2 tests)
- ✅ Instance matching (2 tests)
- ✅ mAP computation (2 tests)
- ✅ STQ computation (2 tests)
- ✅ Unified interface (3 tests)
- ✅ Edge cases (3 tests)

**Total:** 23 comprehensive tests

---

## Performance Considerations

### **Memory Usage:**

- **VOS:** Minimal (frame-by-frame)
- **VIS:** Moderate (needs full video masks)
- **VPS:** High (tracks + associations)

### **Computation Time:**

For 100-frame video @ 480×854:

- **J&F:** ~2 seconds
- **mAP:** ~5 seconds (depends on #instances)
- **STQ:** ~3 seconds

### **Optimization Tips:**

1. **Batch IoU computation:** Vectorize across frames
2. **Sparse masks:** Use run-length encoding
3. **Caching:** Precompute distance transforms
4. **Parallel:** Multi-process across videos

---

## Common Issues

### **Issue 1: Low F-measure despite high J**

**Cause:** Boundary is rough/jagged

**Solution:** Apply morphological smoothing to masks

### **Issue 2: mAP = 0 despite visible predictions**

**Cause:** Class mismatch or IoU threshold too strict

**Debug:**
```python
# Check matches at different thresholds
for thresh in [0.3, 0.5, 0.7]:
    tp, fp, fn = match_instances(pred, gt, iou_threshold=thresh)
    print(f"IoU={thresh}: TP={len(tp)}, FP={len(fp)}, FN={len(fn)}")
```

### **Issue 3: STQ much lower than SQ**

**Cause:** Poor temporal consistency (objects switching IDs)

**Solution:** Improve tracking with temporal smoothing

---

## Comparison to Other Frameworks

### **vs. COCO metrics:**

| Metric | COCO | TarViS |
|--------|------|--------|
| Detection | AP@[0.5:0.95] | mAP@[0.5:0.95] |
| Segmentation | Mask AP | J&F |
| Tracking | - | STQ |

**Key difference:** TarViS includes temporal consistency (tracking)

### **vs. MOT metrics:**

| Metric | MOT | TarViS |
|--------|-----|--------|
| Detection | MOTA | - |
| Tracking | IDF1 | AQ (part of STQ) |
| Segmentation | - | SQ (part of STQ) |

**Key difference:** TarViS combines segmentation + tracking

---

## Benchmark Datasets

### **DAVIS 2017 (VOS)**
- 150 videos
- 376 objects
- Dense annotations
- **Metric:** J&F

### **YouTube-VIS 2021 (VIS)**
- 2,985 train videos
- 40 classes
- Sparse annotations
- **Metric:** mAP

### **VIPSeg (VPS)**
- 3,536 videos
- 124 categories (58 things + 66 stuff)
- Panoptic annotations
- **Metric:** STQ

## Resources

**Papers:**
- DAVIS: https://arxiv.org/abs/1704.00675
- YouTube-VIS: https://arxiv.org/abs/1905.04804
- VIPSeg: https://arxiv.org/abs/2205.05769
- TarViS: https://arxiv.org/abs/2301.04962

**Code:**
- DAVIS evaluation: https://github.com/davisvideochallenge/davis-2017
- YouTube-VIS evaluation: https://github.com/youtubevos/cocoapi
- Official TarViS: https://github.com/Ali2500/TarViS

**Benchmarks:**
- DAVIS: https://davischallenge.org/
- YouTube-VIS: https://youtube-vos.org/
- VIPSeg: https://github.com/VIPSeg-Dataset/VIPSeg-Dataset