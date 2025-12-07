"""
src/utils/evaluation_metrics.py

Evaluation metrics for TarViS video segmentation tasks.

Implements:
- J&F Score (DAVIS VOS): Jaccard (IoU) & F-measure
- mAP (YouTube-VIS): Mean Average Precision
- STQ (VIPSeg VPS): Segmentation & Tracking Quality
- IoU: Intersection over Union (base metric)
"""

from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
TORCH_AVAILABLE = True


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU/Jaccard Index).
    
    Args:
        pred_mask: Predicted binary mask [H, W]
        gt_mask: Ground truth binary mask [H, W]
    
    Returns:
        IoU score in [0, 1]
    """
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection) / float(union)


def compute_f_measure(pred_mask: np.ndarray, gt_mask: np.ndarray, bound_th: float = 0.008) -> float:
    """
    Compute F-measure (contour accuracy).
    
    Measures how well predicted boundaries match ground truth boundaries.
    
    Args:
        pred_mask: Predicted binary mask [H, W]
        gt_mask: Ground truth binary mask [H, W]
        bound_th: Boundary thickness threshold (fraction of image diagonal)
    
    Returns:
        F-measure score in [0, 1]
    """
    try:
        from scipy.ndimage import distance_transform_edt
        from skimage.morphology import binary_dilation, disk
    except ImportError:
        # Fallback: return IoU if scipy/skimage not available
        return compute_iou(pred_mask, gt_mask)
    
    # Get boundaries
    pred_boundary = pred_mask ^ binary_dilation(pred_mask, disk(1))
    gt_boundary = gt_mask ^ binary_dilation(gt_mask, disk(1))
    
    # Early exit if no boundaries
    if gt_boundary.sum() == 0:
        return 1.0 if pred_boundary.sum() == 0 else 0.0
    if pred_boundary.sum() == 0:
        return 0.0
    
    # Distance transforms
    pred_dists = distance_transform_edt(~pred_boundary)
    gt_dists = distance_transform_edt(~gt_boundary)
    
    # Threshold (as fraction of image diagonal)
    H, W = pred_mask.shape
    threshold = bound_th * np.sqrt(H**2 + W**2)
    
    # Precision: fraction of pred boundary close to gt boundary
    pred_close = pred_dists[pred_boundary] < threshold
    precision = pred_close.sum() / pred_boundary.sum() if pred_boundary.sum() > 0 else 0.0
    
    # Recall: fraction of gt boundary close to pred boundary
    gt_close = gt_dists[gt_boundary] < threshold
    recall = gt_close.sum() / gt_boundary.sum() if gt_boundary.sum() > 0 else 0.0
    
    # F-measure
    if precision + recall == 0:
        return 0.0
    
    f_measure = 2 * precision * recall / (precision + recall)
    
    return float(f_measure)


def compute_j_and_f(pred_masks: np.ndarray, gt_masks: np.ndarray) -> Dict[str, float]:
    """
    Compute J&F score for VOS (DAVIS).
    
    Args:
        pred_masks: Predicted masks [T, H, W] or [T, O, H, W]
        gt_masks: Ground truth masks [T, H, W] or [T, O, H, W]
    
    Returns:
        Dictionary with:
            - 'J': Mean Jaccard (IoU)
            - 'F': Mean F-measure
            - 'J&F': Mean of J and F
    """
    # Handle single object case
    if pred_masks.ndim == 3:
        pred_masks = pred_masks[:, np.newaxis, :, :]
    if gt_masks.ndim == 3:
        gt_masks = gt_masks[:, np.newaxis, :, :]
    
    T, O, H, W = pred_masks.shape
    
    j_scores = []
    f_scores = []
    
    for t in range(T):
        for o in range(O):
            pred = pred_masks[t, o] > 0.5
            gt = gt_masks[t, o] > 0.5
            
            # Skip if both empty
            if not gt.any() and not pred.any():
                continue
            
            j_scores.append(compute_iou(pred, gt))
            f_scores.append(compute_f_measure(pred, gt))
    
    # Compute means
    j_mean = np.mean(j_scores) if j_scores else 0.0
    f_mean = np.mean(f_scores) if f_scores else 0.0
    j_and_f = (j_mean + f_mean) / 2.0
    
    return {
        'J': j_mean,
        'F': f_mean,
        'J&F': j_and_f
    }


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision from precision-recall curve.
    
    Uses 101-point interpolation (COCO-style).
    
    Args:
        recalls: Recall values [N]
        precisions: Precision values [N]
    
    Returns:
        Average Precision
    """
    # Add sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])
    
    # Ensure precision is non-increasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Compute AP using 101-point interpolation
    recall_thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    
    for r_thresh in recall_thresholds:
        # Find precision at this recall
        idx = np.where(recalls >= r_thresh)[0]
        if len(idx) > 0:
            ap += precisions[idx[0]]
    
    ap /= 101.0
    
    return ap


def match_instances(
    pred_masks: np.ndarray,
    pred_classes: np.ndarray,
    gt_masks: np.ndarray,
    gt_classes: np.ndarray,
    iou_threshold: float = 0.5
) -> Tuple[List, List, List]:
    """
    Match predicted instances to ground truth using IoU.
    
    Args:
        pred_masks: Predicted masks [N, H, W]
        pred_classes: Predicted class IDs [N]
        gt_masks: Ground truth masks [M, H, W]
        gt_classes: Ground truth class IDs [M]
        iou_threshold: IoU threshold for matching
    
    Returns:
        Tuple of (true_positives, false_positives, false_negatives)
    """
    N = len(pred_masks)
    M = len(gt_masks)
    
    # Compute IoU matrix [N, M]
    iou_matrix = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            if pred_classes[i] == gt_classes[j]:
                iou_matrix[i, j] = compute_iou(pred_masks[i], gt_masks[j])
    
    # Greedy matching: assign predictions to GTs
    matched_gt = set()
    true_positives = []
    false_positives = []
    
    # Sort predictions by confidence (assume sorted already)
    for i in range(N):
        # Find best matching GT
        best_iou = 0.0
        best_j = -1
        
        for j in range(M):
            if j not in matched_gt and iou_matrix[i, j] > best_iou:
                best_iou = iou_matrix[i, j]
                best_j = j
        
        if best_iou >= iou_threshold:
            # True positive
            true_positives.append((i, best_j, best_iou))
            matched_gt.add(best_j)
        else:
            # False positive
            false_positives.append(i)
    
    # Unmatched GTs are false negatives
    false_negatives = [j for j in range(M) if j not in matched_gt]
    
    return true_positives, false_positives, false_negatives


def compute_map(
    pred_masks_list: List[np.ndarray],
    pred_classes_list: List[np.ndarray],
    pred_scores_list: List[np.ndarray],
    gt_masks_list: List[np.ndarray],
    gt_classes_list: List[np.ndarray],
    num_classes: int,
    iou_thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
) -> Dict[str, float]:
    """
    Compute mean Average Precision (mAP) for VIS.
    
    Args:
        pred_masks_list: List of predicted masks per video [N, T, H, W]
        pred_classes_list: List of predicted classes per video [N]
        pred_scores_list: List of confidence scores per video [N]
        gt_masks_list: List of GT masks per video [M, T, H, W]
        gt_classes_list: List of GT classes per video [M]
        num_classes: Number of classes
        iou_thresholds: IoU thresholds for AP computation
    
    Returns:
        Dictionary with mAP scores at different IoU thresholds
    """
    # Aggregate all predictions and GTs across videos
    all_predictions = []
    all_ground_truths = []
    
    for video_idx in range(len(pred_masks_list)):
        pred_masks = pred_masks_list[video_idx]  # [N, T, H, W]
        pred_classes = pred_classes_list[video_idx]
        pred_scores = pred_scores_list[video_idx]
        
        gt_masks = gt_masks_list[video_idx]  # [M, T, H, W]
        gt_classes = gt_classes_list[video_idx]
        
        # For VIS, we evaluate on the union of masks across time
        for i, (masks, cls, score) in enumerate(zip(pred_masks, pred_classes, pred_scores)):
            # Union mask across time
            union_mask = np.any(masks > 0.5, axis=0)  # [H, W]
            all_predictions.append({
                'mask': union_mask,
                'class': cls,
                'score': score,
                'video_id': video_idx
            })
        
        for j, (masks, cls) in enumerate(zip(gt_masks, gt_classes)):
            union_mask = np.any(masks > 0.5, axis=0)
            all_ground_truths.append({
                'mask': union_mask,
                'class': cls,
                'video_id': video_idx
            })
    
    # Sort predictions by score (descending)
    all_predictions.sort(key=lambda x: x['score'], reverse=True)
    
    # Compute AP for each class and IoU threshold
    results = {}
    
    for iou_thresh in iou_thresholds:
        class_aps = []
        
        for cls in range(num_classes):
            # Filter by class
            cls_preds = [p for p in all_predictions if p['class'] == cls]
            cls_gts = [g for g in all_ground_truths if g['class'] == cls]
            
            if len(cls_gts) == 0:
                continue  # No GT for this class
            
            # Track which GTs are matched
            matched = defaultdict(set)  # video_id -> set of matched GT indices
            
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            
            for i, pred in enumerate(cls_preds):
                video_id = pred['video_id']
                pred_mask = pred['mask']
                
                # Find GTs in same video
                video_gts = [(j, g) for j, g in enumerate(cls_gts) if g['video_id'] == video_id]
                
                # Match to best GT
                best_iou = 0.0
                best_gt_idx = -1
                
                for j, gt in video_gts:
                    if j in matched[video_id]:
                        continue
                    
                    iou = compute_iou(pred_mask, gt['mask'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = j
                
                if best_iou >= iou_thresh:
                    tp[i] = 1
                    matched[video_id].add(best_gt_idx)
                else:
                    fp[i] = 1
            
            # Compute precision-recall
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recalls = tp_cumsum / len(cls_gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            # Compute AP
            ap = compute_ap(recalls, precisions)
            class_aps.append(ap)
        
        # Mean AP across classes
        if class_aps:
            results[f'AP@{iou_thresh:.2f}'] = np.mean(class_aps)
    
    # Overall mAP (mean across IoU thresholds)
    if results:
        results['mAP'] = np.mean(list(results.values()))
    else:
        results['mAP'] = 0.0
    
    return results


def compute_stq(
    pred_masks: np.ndarray,
    pred_classes: np.ndarray,
    gt_masks: np.ndarray,
    gt_classes: np.ndarray
) -> Dict[str, float]:
    """
    Compute Segmentation and Tracking Quality (STQ) for VPS.
    
    STQ = sqrt(AQ * SQ)
    - AQ: Association Quality (tracking accuracy)
    - SQ: Segmentation Quality (mask quality)
    
    Args:
        pred_masks: Predicted masks [T, N, H, W]
        pred_classes: Predicted semantic classes [T, N]
        gt_masks: Ground truth masks [T, M, H, W]
        gt_classes: Ground truth semantic classes [T, M]
    
    Returns:
        Dictionary with STQ, AQ, SQ scores
    """
    T = len(pred_masks)
    
    # Track correspondences across time
    total_iou = 0.0
    total_matches = 0
    total_gt = 0
    total_pred = 0
    
    # Track instance associations
    pred_tracks = defaultdict(set)  # pred_id -> set of (time, gt_id)
    gt_tracks = defaultdict(set)    # gt_id -> set of (time, pred_id)
    
    for t in range(T):
        pred_m = pred_masks[t]  # [N, H, W]
        pred_c = pred_classes[t]  # [N]
        gt_m = gt_masks[t]  # [M, H, W]
        gt_c = gt_classes[t]  # [M]
        
        N = len(pred_m)
        M = len(gt_m)
        
        # Match instances in this frame
        tp, fp, fn = match_instances(pred_m, pred_c, gt_m, gt_c, iou_threshold=0.5)
        
        # Update segmentation quality
        for pred_idx, gt_idx, iou in tp:
            total_iou += iou
            total_matches += 1
            
            # Track associations
            pred_tracks[pred_idx].add((t, gt_idx))
            gt_tracks[gt_idx].add((t, pred_idx))
        
        total_gt += M
        total_pred += N
    
    # Segmentation Quality
    sq = total_iou / total_matches if total_matches > 0 else 0.0
    
    # Association Quality (IDF1-style)
    # Count correctly associated detections
    correct_associations = 0
    
    for pred_id, associations in pred_tracks.items():
        # Find most common GT
        gt_counts = defaultdict(int)
        for t, gt_id in associations:
            gt_counts[gt_id] += 1
        
        if gt_counts:
            correct_associations += max(gt_counts.values())
    
    aq = correct_associations / (total_gt + total_pred) if (total_gt + total_pred) > 0 else 0.0
    
    # STQ
    stq = np.sqrt(sq * aq)
    
    return {
        'STQ': stq,
        'AQ': aq,
        'SQ': sq
    }


class EvaluationMetrics:
    """
    Unified evaluation metrics for all TarViS tasks.
    
    Usage:
        metrics = EvaluationMetrics()
        
        # VOS evaluation
        j_f_scores = metrics.evaluate_vos(pred_masks, gt_masks)
        
        # VIS evaluation
        map_scores = metrics.evaluate_vis(predictions, ground_truths)
        
        # VPS evaluation
        stq_scores = metrics.evaluate_vps(predictions, ground_truths)
    """
    
    def evaluate_vos(
        self,
        pred_masks: np.ndarray,
        gt_masks: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate VOS (Video Object Segmentation).
        
        Args:
            pred_masks: Predicted masks [T, O, H, W]
            gt_masks: Ground truth masks [T, O, H, W]
        
        Returns:
            J&F scores
        """
        return compute_j_and_f(pred_masks, gt_masks)
    
    def evaluate_vis(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        num_classes: int
    ) -> Dict[str, float]:
        """
        Evaluate VIS (Video Instance Segmentation).
        
        Args:
            predictions: List of prediction dicts per video with:
                - 'masks': [N, T, H, W]
                - 'classes': [N]
                - 'scores': [N]
            ground_truths: List of GT dicts per video with:
                - 'masks': [M, T, H, W]
                - 'classes': [M]
            num_classes: Number of classes
        
        Returns:
            mAP scores
        """
        pred_masks_list = [p['masks'] for p in predictions]
        pred_classes_list = [p['classes'] for p in predictions]
        pred_scores_list = [p['scores'] for p in predictions]
        
        gt_masks_list = [g['masks'] for g in ground_truths]
        gt_classes_list = [g['classes'] for g in ground_truths]
        
        return compute_map(
            pred_masks_list,
            pred_classes_list,
            pred_scores_list,
            gt_masks_list,
            gt_classes_list,
            num_classes
        )
    
    def evaluate_vps(
        self,
        pred_masks: np.ndarray,
        pred_classes: np.ndarray,
        gt_masks: np.ndarray,
        gt_classes: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate VPS (Video Panoptic Segmentation).
        
        Args:
            pred_masks: Predicted masks [T, N, H, W]
            pred_classes: Predicted classes [T, N]
            gt_masks: Ground truth masks [T, M, H, W]
            gt_classes: Ground truth classes [T, M]
        
        Returns:
            STQ scores
        """
        return compute_stq(pred_masks, pred_classes, gt_masks, gt_classes)


# Example usage
if __name__ == "__main__":
    print("=== TarViS Evaluation Metrics Demo ===\n")
    
    # VOS Example (DAVIS)
    print("1. VOS Evaluation (J&F):")
    pred_vos = np.random.rand(10, 2, 480, 854) > 0.5  # 10 frames, 2 objects
    gt_vos = np.random.rand(10, 2, 480, 854) > 0.5
    
    metrics = EvaluationMetrics()
    j_f_scores = metrics.evaluate_vos(pred_vos, gt_vos)
    print(f"   J (IoU): {j_f_scores['J']:.4f}")
    print(f"   F (F-measure): {j_f_scores['F']:.4f}")
    print(f"   J&F: {j_f_scores['J&F']:.4f}\n")
    
    # VIS Example (YouTube-VIS)
    print("2. VIS Evaluation (mAP):")
    predictions = [{
        'masks': np.random.rand(5, 10, 480, 854) > 0.5,  # 5 instances, 10 frames
        'classes': np.random.randint(0, 40, 5),
        'scores': np.random.rand(5)
    }]
    ground_truths = [{
        'masks': np.random.rand(3, 10, 480, 854) > 0.5,  # 3 GT instances
        'classes': np.random.randint(0, 40, 3)
    }]
    
    map_scores = metrics.evaluate_vis(predictions, ground_truths, num_classes=40)
    print(f"   mAP: {map_scores['mAP']:.4f}")
    print(f"   AP@0.50: {map_scores.get('AP@0.50', 0):.4f}\n")
    
    # VPS Example (VIPSeg)
    print("3. VPS Evaluation (STQ):")
    pred_vps = np.random.rand(10, 5, 480, 854) > 0.5  # 10 frames, 5 instances
    pred_classes_vps = np.random.randint(0, 124, (10, 5))
    gt_vps = np.random.rand(10, 3, 480, 854) > 0.5
    gt_classes_vps = np.random.randint(0, 124, (10, 3))
    
    stq_scores = metrics.evaluate_vps(pred_vps, pred_classes_vps, gt_vps, gt_classes_vps)
    print(f"   STQ: {stq_scores['STQ']:.4f}")
    print(f"   AQ (Association): {stq_scores['AQ']:.4f}")
    print(f"   SQ (Segmentation): {stq_scores['SQ']:.4f}")