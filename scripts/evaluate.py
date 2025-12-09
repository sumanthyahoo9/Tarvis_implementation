#!/usr/bin/env python3
"""
scripts/evaluate.py

Evaluation script for TarViS on various benchmarks.

Supports:
- DAVIS 2017 (VOS) - J&F metric
- YouTube-VIS 2021 (VIS) - mAP metric
- VIPSeg (VPS) - STQ metric

Usage:
    python scripts/evaluate.py --config configs/eval_config.yaml --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --task vos --dataset davis --checkpoint checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path
import yaml
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not available")
    sys.exit(1)

from src.utils.evaluation_metrics import EvaluationMetrics
from src.data.davis_dataset import build_davis_dataloader
from src.data.ytvis_dataset import build_ytvis_dataloader


def load_model(checkpoint_path: Path, device: str = 'cuda'):
    """Load model from checkpoint."""
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # TODO: Build actual TarViS model
    # For now, use dummy model
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
        
        def forward(self, frames, task='vos'):
            B, T, C, H, W = frames.shape
            masks = torch.sigmoid(torch.randn(B, T, 10, H//4, W//4, device=frames.device))
            outputs = {'masks': masks}
            
            if task in ['vis', 'vps']:
                outputs['classes'] = torch.randn(B, 10, 40, device=frames.device)
            
            return outputs
    
    model = DummyModel()
    
    # Load weights if available
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded model weights")
        except Exception as e:
            print(f"⚠️  Could not load weights (using random): {e}")
    
    model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_vos(
    model: torch.nn.Module,
    dataloader,
    device: str = 'cuda',
    save_predictions: bool = False,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Evaluate on DAVIS (VOS).
    
    Args:
        model: TarViS model
        dataloader: DAVIS dataloader
        device: Device
        save_predictions: Save predicted masks
        output_dir: Directory to save predictions
    
    Returns:
        Dictionary with J&F scores
    """
    print("\n" + "=" * 80)
    print("Evaluating VOS (DAVIS)")
    print("=" * 80)
    
    metrics = EvaluationMetrics()
    all_results = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move to device
        frames = batch['frames'].to(device)  # [B, T, 3, H, W]
        gt_masks = batch['masks']  # [B, T, O, H, W]
        
        # Forward pass
        outputs = model(frames, task='vos')
        pred_masks = outputs['masks']  # [B, T, N, H', W']
        
        # Resize predictions to match GT
        B, T, N, H_pred, W_pred = pred_masks.shape
        H_gt, W_gt = gt_masks.shape[-2:]
        
        if H_pred != H_gt or W_pred != W_gt:
            pred_masks = torch.nn.functional.interpolate(
                pred_masks.flatten(0, 1),  # [B*T, N, H', W']
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            )
            pred_masks = pred_masks.reshape(B, T, N, H_gt, W_gt)
        
        # Convert to numpy
        pred_masks_np = pred_masks.cpu().numpy()
        gt_masks_np = gt_masks.numpy()
        
        # Evaluate each video in batch
        for i in range(B):
            # Select relevant objects (non-zero in GT)
            num_objects = batch['num_objects'][i].item()
            
            pred_video = pred_masks_np[i, :, :num_objects]  # [T, O, H, W]
            gt_video = gt_masks_np[i, :, :num_objects]
            
            # Compute J&F
            scores = metrics.evaluate_vos(pred_video, gt_video)
            all_results.append(scores)
            
            # Save predictions
            if save_predictions and output_dir:
                seq_name = batch['sequence'][i]
                save_dir = output_dir / seq_name
                save_dir.mkdir(parents=True, exist_ok=True)
                
                for t, frame_idx in enumerate(batch['frame_indices'][i]):
                    mask = (pred_video[t].sum(axis=0) > 0.5).astype(np.uint8)
                    mask_path = save_dir / f"{frame_idx:05d}.png"
                    from PIL import Image
                    Image.fromarray(mask * 255).save(mask_path)
    
    # Aggregate results
    avg_j = np.mean([r['J'] for r in all_results])
    avg_f = np.mean([r['F'] for r in all_results])
    avg_jf = np.mean([r['J&F'] for r in all_results])
    
    results = {
        'J': avg_j,
        'F': avg_f,
        'J&F': avg_jf,
        'num_videos': len(all_results)
    }
    
    return results


@torch.no_grad()
def evaluate_vis(
    model: torch.nn.Module,
    dataloader,
    device: str = 'cuda',
    num_classes: int = 40,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Evaluate on YouTube-VIS (VIS).
    
    Args:
        model: TarViS model
        dataloader: YouTube-VIS dataloader
        device: Device
        num_classes: Number of classes
        save_predictions: Save predictions to JSON
        output_dir: Directory to save predictions
    
    Returns:
        Dictionary with mAP scores
    """
    print("\n" + "=" * 80)
    print("Evaluating VIS (YouTube-VIS)")
    print("=" * 80)
    
    metrics = EvaluationMetrics()
    
    all_predictions = []
    all_ground_truths = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move to device
        frames = batch['frames'].to(device)
        gt_masks = batch['masks']
        gt_classes = batch['classes']
        
        # Forward pass
        outputs = model(frames, task='vis')
        pred_masks = outputs['masks']  # [B, T, N, H', W']
        pred_classes_logits = outputs['classes']  # [B, N, C]
        
        # Get predicted classes
        pred_classes = torch.argmax(pred_classes_logits, dim=-1)  # [B, N]
        pred_scores = torch.softmax(pred_classes_logits, dim=-1).max(dim=-1)[0]  # [B, N]
        
        # Resize predictions
        B, T, N, H_pred, W_pred = pred_masks.shape
        H_gt, W_gt = gt_masks.shape[-2:]
        
        if H_pred != H_gt or W_pred != W_gt:
            pred_masks = torch.nn.functional.interpolate(
                pred_masks.flatten(0, 1),
                size=(H_gt, W_gt),
                mode='bilinear',
                align_corners=False
            )
            pred_masks = pred_masks.reshape(B, T, N, H_gt, W_gt)
        
        # Convert to numpy
        pred_masks_np = (pred_masks > 0.5).cpu().numpy()
        pred_classes_np = pred_classes.cpu().numpy()
        pred_scores_np = pred_scores.cpu().numpy()
        
        gt_masks_np = gt_masks.numpy()
        gt_classes_np = gt_classes.numpy()
        
        # Collect predictions and GTs
        for i in range(B):
            all_predictions.append({
                'masks': pred_masks_np[i],  # [T, N, H, W]
                'classes': pred_classes_np[i],  # [N]
                'scores': pred_scores_np[i]  # [N]
            })
            
            all_ground_truths.append({
                'masks': gt_masks_np[i],  # [T, M, H, W]
                'classes': gt_classes_np[i]  # [M]
            })
    
    # Compute mAP
    results = metrics.evaluate_vis(
        all_predictions,
        all_ground_truths,
        num_classes=num_classes
    )
    
    results['num_videos'] = len(all_predictions)
    
    # Save predictions to JSON
    if save_predictions and output_dir:
        output_file = output_dir / 'predictions.json'
        
        # Convert to submission format
        submission = []
        for i, pred in enumerate(all_predictions):
            video_id = i  # Replace with actual video_id
            for inst_idx in range(len(pred['classes'])):
                submission.append({
                    'video_id': video_id,
                    'category_id': int(pred['classes'][inst_idx]),
                    'score': float(pred['scores'][inst_idx]),
                    # 'segmentations': ... (RLE format)
                })
        
        with open(output_file, 'w') as f:
            json.dump(submission, f)
        
        print(f"💾 Saved predictions to: {output_file}")
    
    return results


def print_results(results: Dict, task: str):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print(f"📊 {task.upper()} Results")
    print("=" * 80)
    
    if task == 'vos':
        print(f"J (Jaccard):     {results['J']:.4f}")
        print(f"F (F-measure):   {results['F']:.4f}")
        print(f"J&F:             {results['J&F']:.4f}")
    elif task == 'vis':
        print(f"mAP:             {results['mAP']:.4f}")
        if 'AP@0.50' in results:
            print(f"AP@0.50:         {results['AP@0.50']:.4f}")
        if 'AP@0.75' in results:
            print(f"AP@0.75:         {results['AP@0.75']:.4f}")
    elif task == 'vps':
        print(f"STQ:             {results['STQ']:.4f}")
        print(f"AQ (Association):{results['AQ']:.4f}")
        print(f"SQ (Segmentation):{results['SQ']:.4f}")
    
    print(f"Videos evaluated: {results['num_videos']}")
    print("=" * 80)


def main(args):
    """Main evaluation function."""
    
    print("=" * 80)
    print("TarViS Evaluation")
    print("=" * 80)
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    model = load_model(checkpoint_path, device=device)
    
    # Create output directory
    if args.save_predictions:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Evaluate based on task
    if args.task == 'vos':
        # DAVIS evaluation
        dataloader = build_davis_dataloader(
            root=args.dataset_path,
            split='val',
            batch_size=1,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            shuffle=False
        )
        
        results = evaluate_vos(
            model=model,
            dataloader=dataloader,
            device=device,
            save_predictions=args.save_predictions,
            output_dir=output_dir
        )
    
    elif args.task == 'vis':
        # YouTube-VIS evaluation
        dataloader = build_ytvis_dataloader(
            root=args.dataset_path,
            split='valid',
            batch_size=1,
            num_workers=args.num_workers,
            num_frames=args.num_frames,
            shuffle=False
        )
        
        results = evaluate_vis(
            model=model,
            dataloader=dataloader,
            device=device,
            num_classes=40,
            save_predictions=args.save_predictions,
            output_dir=output_dir
        )
    
    else:
        print(f"ERROR: Unknown task: {args.task}")
        sys.exit(1)
    
    # Print results
    print_results(results, args.task)
    
    # Save results to JSON
    results_file = Path(args.output_dir) / f'{args.task}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TarViS model")
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['vos', 'vis', 'vps'],
        required=True,
        help='Evaluation task'
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to dataset root'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='eval_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=5,
        help='Number of frames per clip'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU instead of GPU'
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        help='Save predicted masks/results'
    )
    
    args = parser.parse_args()
    main(args)