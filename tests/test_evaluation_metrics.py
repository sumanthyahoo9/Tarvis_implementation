"""
tests/test_evaluation_metrics.py

Unit tests for TarViS evaluation metrics.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
from src.utils.evaluation_metrics import (
    compute_iou,
    compute_f_measure,
    compute_j_and_f,
    compute_ap,
    match_instances,
    compute_map,
    compute_stq,
    EvaluationMetrics
)

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIoU:
    """Test IoU computation."""
    
    def test_perfect_match(self):
        """Test IoU with perfect overlap."""
        mask1 = np.ones((100, 100), dtype=bool)
        mask2 = np.ones((100, 100), dtype=bool)
        
        iou = compute_iou(mask1, mask2)
        assert iou == 1.0
    
    def test_no_overlap(self):
        """Test IoU with no overlap."""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[:50, :50] = True
        
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[50:, 50:] = True
        
        iou = compute_iou(mask1, mask2)
        assert iou == 0.0
    
    def test_half_overlap(self):
        """Test IoU with 50% overlap."""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[:, :50] = True
        
        mask2 = np.zeros((100, 100), dtype=bool)
        mask2[:, 25:75] = True
        
        iou = compute_iou(mask1, mask2)
        # Intersection: 25 cols, Union: 75 cols
        expected = 25 / 75
        assert abs(iou - expected) < 0.01
    
    def test_empty_masks(self):
        """Test IoU with both masks empty."""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask2 = np.zeros((100, 100), dtype=bool)
        
        iou = compute_iou(mask1, mask2)
        assert iou == 1.0  # Convention: both empty = perfect match


class TestFMeasure:
    """Test F-measure computation."""
    
    def test_perfect_match(self):
        """Test F-measure with perfect overlap."""
        mask1 = np.ones((100, 100), dtype=bool)
        mask2 = np.ones((100, 100), dtype=bool)
        
        f = compute_f_measure(mask1, mask2)
        assert f >= 0.9  # Should be very high
    
    def test_no_overlap(self):
        """Test F-measure with no overlap."""
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[25:75, 25:75] = True
        
        mask2 = np.zeros((100, 100), dtype=bool)
        
        f = compute_f_measure(mask1, mask2)
        assert f == 0.0


class TestJAndF:
    """Test J&F score computation."""
    
    def test_single_object(self):
        """Test J&F for single object across time."""
        T, H, W = 10, 100, 100
        
        # Perfect prediction
        pred = np.ones((T, H, W))
        gt = np.ones((T, H, W))
        
        scores = compute_j_and_f(pred, gt)
        
        assert 'J' in scores
        assert 'F' in scores
        assert 'J&F' in scores
        assert scores['J'] == 1.0
        assert scores['J&F'] >= 0.9
    
    def test_multiple_objects(self):
        """Test J&F for multiple objects."""
        T, O, H, W = 10, 3, 100, 100
        
        pred = np.random.rand(T, O, H, W) > 0.5
        gt = pred.copy()  # Perfect prediction
        
        scores = compute_j_and_f(pred, gt)
        
        assert scores['J'] == 1.0
        assert scores['J&F'] >= 0.9
    
    def test_partial_overlap(self):
        """Test J&F with partial overlap."""
        T, H, W = 5, 100, 100
        
        pred = np.zeros((T, H, W))
        pred[:, :, :60] = 1
        
        gt = np.zeros((T, H, W))
        gt[:, :, 40:] = 1
        
        scores = compute_j_and_f(pred, gt)
        
        # Should have some overlap but not perfect
        assert 0.0 < scores['J'] < 1.0
        assert 0.0 < scores['J&F'] < 1.0


class TestAP:
    """Test Average Precision computation."""
    
    def test_perfect_ranking(self):
        """Test AP with perfect ranking."""
        # All true positives, ordered by confidence
        recalls = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
        precisions = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        ap = compute_ap(recalls, precisions)
        assert ap == 1.0
    
    def test_mixed_ranking(self):
        """Test AP with mixed ranking."""
        recalls = np.array([0.25, 0.5, 0.5, 0.75, 1.0])
        precisions = np.array([1.0, 1.0, 0.67, 0.75, 0.8])
        
        ap = compute_ap(recalls, precisions)
        assert 0.0 < ap < 1.0


class TestMatchInstances:
    """Test instance matching."""
    
    def test_perfect_matches(self):
        """Test matching with perfect overlap."""
        N, M = 3, 3
        H, W = 100, 100
        
        # Create identical predictions and GTs
        pred_masks = np.zeros((N, H, W))
        gt_masks = np.zeros((M, H, W))
        
        for i in range(N):
            pred_masks[i, i*30:(i+1)*30, :] = 1
            gt_masks[i, i*30:(i+1)*30, :] = 1
        
        pred_classes = np.array([0, 1, 2])
        gt_classes = np.array([0, 1, 2])
        
        tp, fp, fn = match_instances(
            pred_masks, pred_classes,
            gt_masks, gt_classes,
            iou_threshold=0.5
        )
        
        assert len(tp) == 3  # All matched
        assert len(fp) == 0
        assert len(fn) == 0
    
    def test_no_matches(self):
        """Test matching with no overlap."""
        pred_masks = np.zeros((2, 100, 100))
        pred_masks[0, :50, :] = 1
        pred_masks[1, 50:, :] = 1
        
        gt_masks = np.zeros((2, 100, 100))
        gt_masks[0, :, :50] = 1
        gt_masks[1, :, 50:] = 1
        
        pred_classes = np.array([0, 0])
        gt_classes = np.array([0, 0])
        
        tp, fp, fn = match_instances(
            pred_masks, pred_classes,
            gt_masks, gt_classes,
            iou_threshold=0.5
        )
        
        assert len(tp) == 0
        assert len(fp) == 2  # All predictions unmatched
        assert len(fn) == 2  # All GTs unmatched


class TestMAP:
    """Test mAP computation."""
    
    def test_single_video_perfect(self):
        """Test mAP with perfect predictions."""
        # Single video, perfect predictions
        pred_masks = [np.ones((2, 5, 100, 100))]  # 2 instances, 5 frames
        pred_classes = [np.array([0, 1])]
        pred_scores = [np.array([0.9, 0.8])]
        
        gt_masks = [np.ones((2, 5, 100, 100))]
        gt_classes = [np.array([0, 1])]
        
        scores = compute_map(
            pred_masks, pred_classes, pred_scores,
            gt_masks, gt_classes,
            num_classes=2,
            iou_thresholds=[0.5]
        )
        
        assert 'mAP' in scores
        assert scores['mAP'] > 0.8  # Should be high
    
    def test_multiple_videos(self):
        """Test mAP with multiple videos."""
        # Two videos
        pred_masks = [
            np.random.rand(2, 5, 100, 100) > 0.5,
            np.random.rand(3, 5, 100, 100) > 0.5
        ]
        pred_classes = [
            np.array([0, 1]),
            np.array([0, 1, 2])
        ]
        pred_scores = [
            np.array([0.9, 0.8]),
            np.array([0.85, 0.75, 0.7])
        ]
        
        gt_masks = [
            np.random.rand(2, 5, 100, 100) > 0.5,
            np.random.rand(2, 5, 100, 100) > 0.5
        ]
        gt_classes = [
            np.array([0, 1]),
            np.array([1, 2])
        ]
        
        scores = compute_map(
            pred_masks, pred_classes, pred_scores,
            gt_masks, gt_classes,
            num_classes=3,
            iou_thresholds=[0.5]
        )
        
        assert 'mAP' in scores
        assert 0.0 <= scores['mAP'] <= 1.0


class TestSTQ:
    """Test STQ computation."""
    
    def test_perfect_tracking(self):
        """Test STQ with perfect tracking."""
        T, N, H, W = 10, 3, 100, 100
        
        # Perfect predictions
        pred_masks = np.random.rand(T, N, H, W) > 0.5
        pred_classes = np.random.randint(0, 10, (T, N))
        
        gt_masks = pred_masks.copy()
        gt_classes = pred_classes.copy()
        
        scores = compute_stq(pred_masks, pred_classes, gt_masks, gt_classes)
        
        assert 'STQ' in scores
        assert 'AQ' in scores
        assert 'SQ' in scores
        assert scores['SQ'] == 1.0  # Perfect segmentation
        assert scores['STQ'] > 0.5  # Should be high
    
    def test_imperfect_tracking(self):
        """Test STQ with imperfect tracking."""
        T, H, W = 5, 100, 100
        
        # Predictions and GTs with some overlap
        pred_masks = np.random.rand(T, 3, H, W) > 0.5
        pred_classes = np.zeros((T, 3), dtype=int)
        
        gt_masks = np.random.rand(T, 2, H, W) > 0.5
        gt_classes = np.zeros((T, 2), dtype=int)
        
        scores = compute_stq(pred_masks, pred_classes, gt_masks, gt_classes)
        
        assert 0.0 <= scores['STQ'] <= 1.0
        assert 0.0 <= scores['AQ'] <= 1.0
        assert 0.0 <= scores['SQ'] <= 1.0


class TestEvaluationMetrics:
    """Test unified metrics class."""
    
    def test_evaluate_vos(self):
        """Test VOS evaluation."""
        metrics = EvaluationMetrics()
        
        pred = np.ones((10, 2, 100, 100))
        gt = np.ones((10, 2, 100, 100))
        
        scores = metrics.evaluate_vos(pred, gt)
        
        assert 'J' in scores
        assert 'F' in scores
        assert 'J&F' in scores
    
    def test_evaluate_vis(self):
        """Test VIS evaluation."""
        metrics = EvaluationMetrics()
        
        predictions = [{
            'masks': np.random.rand(3, 5, 100, 100) > 0.5,
            'classes': np.array([0, 1, 2]),
            'scores': np.array([0.9, 0.8, 0.7])
        }]
        
        ground_truths = [{
            'masks': np.random.rand(2, 5, 100, 100) > 0.5,
            'classes': np.array([0, 1])
        }]
        
        scores = metrics.evaluate_vis(predictions, ground_truths, num_classes=3)
        
        assert 'mAP' in scores
    
    def test_evaluate_vps(self):
        """Test VPS evaluation."""
        metrics = EvaluationMetrics()
        
        pred_masks = np.random.rand(5, 3, 100, 100) > 0.5
        pred_classes = np.zeros((5, 3), dtype=int)
        gt_masks = np.random.rand(5, 2, 100, 100) > 0.5
        gt_classes = np.zeros((5, 2), dtype=int)
        
        scores = metrics.evaluate_vps(pred_masks, pred_classes, gt_masks, gt_classes)
        
        assert 'STQ' in scores
        assert 'AQ' in scores
        assert 'SQ' in scores


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_predictions(self):
        """Test with no predictions."""
        pred_masks = np.zeros((5, 0, 100, 100))
        gt_masks = np.ones((5, 2, 100, 100))
        
        scores = compute_j_and_f(pred_masks, gt_masks)
        assert scores['J'] == 0.0
    
    def test_empty_ground_truth(self):
        """Test with no ground truth."""
        pred_masks = np.ones((5, 2, 100, 100))
        gt_masks = np.zeros((5, 0, 100, 100))
        
        scores = compute_j_and_f(pred_masks, gt_masks)
        assert scores['J'] == 0.0
    
    def test_single_frame(self):
        """Test with single frame."""
        pred = np.ones((1, 100, 100))
        gt = np.ones((1, 100, 100))
        
        scores = compute_j_and_f(pred, gt)
        assert scores['J'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])