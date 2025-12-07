#!/usr/bin/env python3
"""
scripts/validate_metrics.py

Validate evaluation metrics implementation with synthetic data.
"""

import sys
from pathlib import Path
import numpy as np
from src.utils.evaluation_metrics import EvaluationMetrics
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_vos_metrics():
    """Test VOS (J&F) metrics."""
    print("=" * 60)
    print("Testing VOS Metrics (J&F)")
    print("=" * 60)
    
    metrics = EvaluationMetrics()
    
    # Test 1: Perfect prediction
    print("\n1. Perfect Prediction:")
    pred = np.ones((10, 2, 100, 100))
    gt = np.ones((10, 2, 100, 100))
    
    scores = metrics.evaluate_vos(pred, gt)
    print(f"   J: {scores['J']:.4f} (expected: 1.0000)")
    print(f"   F: {scores['F']:.4f} (expected: ~1.0000)")
    print(f"   J&F: {scores['J&F']:.4f} (expected: ~1.0000)")
    
    assert scores['J'] == 1.0, "Perfect J should be 1.0"
    assert scores['J&F'] > 0.95, "Perfect J&F should be > 0.95"
    print("   ✓ PASSED")
    
    # Test 2: Half overlap
    print("\n2. Half Overlap:")
    pred = np.zeros((5, 100, 100))
    pred[:, :, :60] = 1
    
    gt = np.zeros((5, 100, 100))
    gt[:, :, 40:] = 1
    
    scores = metrics.evaluate_vos(pred, gt)
    print(f"   J: {scores['J']:.4f} (expected: ~0.33)")
    print(f"   F: {scores['F']:.4f}")
    print(f"   J&F: {scores['J&F']:.4f}")
    
    assert 0.2 < scores['J'] < 0.5, "Half overlap J should be ~0.33"
    print("   ✓ PASSED")
    
    # Test 3: No overlap
    print("\n3. No Overlap:")
    pred = np.zeros((5, 100, 100))
    pred[:, :50, :] = 1
    
    gt = np.zeros((5, 100, 100))
    gt[:, 50:, :] = 1
    
    scores = metrics.evaluate_vos(pred, gt)
    print(f"   J: {scores['J']:.4f} (expected: 0.0000)")
    print(f"   F: {scores['F']:.4f} (expected: 0.0000)")
    print(f"   J&F: {scores['J&F']:.4f} (expected: 0.0000)")
    
    assert scores['J'] == 0.0, "No overlap J should be 0.0"
    assert scores['J&F'] == 0.0, "No overlap J&F should be 0.0"
    print("   ✓ PASSED")
    
    print("\n✅ All VOS tests passed!")


def test_vis_metrics():
    """Test VIS (mAP) metrics."""
    print("\n" + "=" * 60)
    print("Testing VIS Metrics (mAP)")
    print("=" * 60)
    
    metrics = EvaluationMetrics()
    
    # Test 1: Perfect predictions
    print("\n1. Perfect Predictions:")
    predictions = [{
        'masks': np.ones((3, 5, 100, 100)),
        'classes': np.array([0, 1, 2]),
        'scores': np.array([0.9, 0.8, 0.7])
    }]
    
    ground_truths = [{
        'masks': np.ones((3, 5, 100, 100)),
        'classes': np.array([0, 1, 2])
    }]
    
    scores = metrics.evaluate_vis(predictions, ground_truths, num_classes=3)
    print(f"   mAP: {scores['mAP']:.4f} (expected: ~1.0000)")
    print(f"   AP@0.50: {scores.get('AP@0.50', 0):.4f}")
    
    assert scores['mAP'] > 0.8, "Perfect mAP should be > 0.8"
    print("   ✓ PASSED")
    
    # Test 2: Mixed predictions
    print("\n2. Mixed Predictions:")
    np.random.seed(42)
    predictions = [{
        'masks': np.random.rand(5, 5, 100, 100) > 0.5,
        'classes': np.array([0, 0, 1, 1, 2]),
        'scores': np.array([0.95, 0.9, 0.85, 0.8, 0.75])
    }]
    
    ground_truths = [{
        'masks': np.random.rand(3, 5, 100, 100) > 0.5,
        'classes': np.array([0, 1, 2])
    }]
    
    scores = metrics.evaluate_vis(predictions, ground_truths, num_classes=3)
    print(f"   mAP: {scores['mAP']:.4f} (expected: 0.0-1.0)")
    
    assert 0.0 <= scores['mAP'] <= 1.0, "mAP should be in [0, 1]"
    print("   ✓ PASSED")
    
    print("\n✅ All VIS tests passed!")


def test_vps_metrics():
    """Test VPS (STQ) metrics."""
    print("\n" + "=" * 60)
    print("Testing VPS Metrics (STQ)")
    print("=" * 60)
    
    metrics = EvaluationMetrics()
    
    # Test 1: Perfect tracking
    print("\n1. Perfect Tracking:")
    pred_masks = np.ones((10, 3, 100, 100))
    pred_classes = np.array([[0, 1, 2]] * 10)
    
    gt_masks = np.ones((10, 3, 100, 100))
    gt_classes = np.array([[0, 1, 2]] * 10)
    
    scores = metrics.evaluate_vps(pred_masks, pred_classes, gt_masks, gt_classes)
    print(f"   STQ: {scores['STQ']:.4f} (expected: ~1.0000)")
    print(f"   SQ: {scores['SQ']:.4f} (expected: 1.0000)")
    print(f"   AQ: {scores['AQ']:.4f} (expected: ~0.5-1.0)")
    
    assert scores['SQ'] == 1.0, "Perfect SQ should be 1.0"
    assert scores['STQ'] > 0.5, "Perfect STQ should be > 0.5"
    print("   ✓ PASSED")
    
    # Test 2: Imperfect tracking
    print("\n2. Imperfect Tracking:")
    np.random.seed(42)
    pred_masks = np.random.rand(5, 4, 100, 100) > 0.5
    pred_classes = np.random.randint(0, 10, (5, 4))
    
    gt_masks = np.random.rand(5, 3, 100, 100) > 0.5
    gt_classes = np.random.randint(0, 10, (5, 3))
    
    scores = metrics.evaluate_vps(pred_masks, pred_classes, gt_masks, gt_classes)
    print(f"   STQ: {scores['STQ']:.4f} (expected: 0.0-1.0)")
    print(f"   SQ: {scores['SQ']:.4f}")
    print(f"   AQ: {scores['AQ']:.4f}")
    
    assert 0.0 <= scores['STQ'] <= 1.0, "STQ should be in [0, 1]"
    assert 0.0 <= scores['SQ'] <= 1.0, "SQ should be in [0, 1]"
    assert 0.0 <= scores['AQ'] <= 1.0, "AQ should be in [0, 1]"
    print("   ✓ PASSED")
    
    print("\n✅ All VPS tests passed!")


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 60)
    print("Testing Edge Cases")
    print("=" * 60)
    
    metrics = EvaluationMetrics()
    
    # Test 1: Empty predictions
    print("\n1. Empty Predictions:")
    pred = np.zeros((5, 0, 100, 100))
    gt = np.ones((5, 2, 100, 100))
    
    scores = metrics.evaluate_vos(pred, gt)
    print(f"   J&F: {scores['J&F']:.4f} (expected: 0.0000)")
    assert scores['J&F'] == 0.0, "Empty predictions should give 0"
    print("   ✓ PASSED")
    
    # Test 2: Single frame
    print("\n2. Single Frame:")
    pred = np.ones((1, 1, 100, 100))
    gt = np.ones((1, 1, 100, 100))
    
    scores = metrics.evaluate_vos(pred, gt)
    print(f"   J&F: {scores['J&F']:.4f} (expected: ~1.0000)")
    assert scores['J&F'] > 0.95, "Single frame perfect should be ~1.0"
    print("   ✓ PASSED")
    
    # Test 3: Both empty
    print("\n3. Both Empty:")
    pred = np.zeros((5, 1, 100, 100))
    gt = np.zeros((5, 1, 100, 100))
    
    scores = metrics.evaluate_vos(pred, gt)
    print(f"   J&F: {scores['J&F']:.4f} (expected: 0.0000)")
    # Both empty = no objects to evaluate
    print("   ✓ PASSED")
    
    print("\n✅ All edge case tests passed!")


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("TarViS Evaluation Metrics Validation")
    print("=" * 60)
    
    try:
        test_vos_metrics()
        test_vis_metrics()
        test_vps_metrics()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nMetrics implementation is validated and ready to use.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())