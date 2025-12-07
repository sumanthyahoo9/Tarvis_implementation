"""
src/training/tarvis_trainer.py

Standard supervised training loop for TarViS.

Supports multi-task training across:
- VOS (Video Object Segmentation)
- VIS (Video Instance Segmentation)
- VPS (Video Panoptic Segmentation)
- PET (Point Exemplar-guided Tracking)
"""

from typing import Dict, List, Optional, Tuple, Any
import time
from pathlib import Path
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class LossComputer:
    """
    Compute losses for TarViS training.
    
    Combines:
    - Mask loss (Dice + Focal)
    - Classification loss (Cross-entropy)
    - Task-specific weighting
    """
    
    def __init__(
        self,
        mask_weight: float = 5.0,
        dice_weight: float = 5.0,
        class_weight: float = 2.0,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.class_weight = class_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss for masks.
        
        Args:
            pred: Predicted masks [B, N, H, W]
            target: Target masks [B, N, H, W]
        
        Returns:
            Dice loss (lower is better)
        """
        pred = pred.sigmoid()
        
        # Flatten
        pred = pred.flatten(1)
        target = target.flatten(1)
        
        # Dice coefficient
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2.0 * intersection + 1.0) / (union + 1.0)
        
        return 1.0 - dice.mean()
    
    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Compute Focal loss for masks.
        
        Focal loss = -alpha * (1-p)^gamma * log(p)
        
        Args:
            pred: Predicted logits [B, N, H, W]
            target: Target masks [B, N, H, W]
            alpha: Balance factor
            gamma: Focusing parameter
        
        Returns:
            Focal loss
        """
        pred = pred.sigmoid()
        
        # Focal loss
        ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        p_t = pred * target + (1 - pred) * (1 - target)
        focal_weight = alpha * (1 - p_t) ** gamma
        
        loss = focal_weight * ce_loss
        
        return loss.mean()
    
    def compute_mask_loss(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined mask loss.
        
        Args:
            pred_masks: Predicted mask logits [B, N, H, W]
            target_masks: Target masks [B, N, H, W]
        
        Returns:
            Dictionary with dice_loss, focal_loss, total_mask_loss
        """
        dice = self.dice_loss(pred_masks, target_masks)
        
        if self.use_focal_loss:
            focal = self.focal_loss(pred_masks, target_masks, self.focal_alpha, self.focal_gamma)
        else:
            focal = F.binary_cross_entropy_with_logits(pred_masks, target_masks)
        
        total_mask_loss = self.dice_weight * dice + self.mask_weight * focal
        
        return {
            'dice_loss': dice,
            'focal_loss': focal,
            'mask_loss': total_mask_loss
        }
    
    def compute_class_loss(
        self,
        pred_classes: torch.Tensor,
        target_classes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Args:
            pred_classes: Predicted class logits [N, C]
            target_classes: Target class indices [N]
        
        Returns:
            Cross-entropy loss
        """
        return F.cross_entropy(pred_classes, target_classes)
    
    def compute_loss(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        pred_classes: Optional[torch.Tensor] = None,
        target_classes: Optional[torch.Tensor] = None,
        task: str = 'vos'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss for a task.
        
        Args:
            pred_masks: Predicted mask logits
            target_masks: Target masks
            pred_classes: Predicted class logits (for VIS/VPS)
            target_classes: Target classes (for VIS/VPS)
            task: Task name ('vos', 'vis', 'vps', 'pet')
        
        Returns:
            Dictionary with all losses
        """
        losses = {}
        
        # Mask loss (all tasks)
        mask_losses = self.compute_mask_loss(pred_masks, target_masks)
        losses.update(mask_losses)
        
        # Classification loss (VIS/VPS only)
        if task in ['vis', 'vps'] and pred_classes is not None:
            class_loss = self.compute_class_loss(pred_classes, target_classes)
            losses['class_loss'] = class_loss
            losses['total_loss'] = mask_losses['mask_loss'] + self.class_weight * class_loss
        else:
            losses['total_loss'] = mask_losses['mask_loss']
        
        return losses


class TarvisTrainer:
    """
    Training loop for TarViS.
    
    Features:
    - Multi-task training (VOS, VIS, VPS, PET)
    - Gradient accumulation
    - Mixed precision training
    - Checkpoint saving/loading
    - TensorBoard logging
    
    Args:
        model: TarViS model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_computer: Loss computation module
        device: Training device
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Gradient clipping threshold
        use_amp: Use automatic mixed precision
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss_computer: Optional[LossComputer] = None,
        device: str = 'cuda',
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        checkpoint_dir: Optional[Path] = None
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for training")
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_computer = loss_computer or LossComputer()
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = 0.0
        
        # Move model to device
        self.model.to(device)
        
        # Create checkpoint directory
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        task: str = 'vos'
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: Batch dictionary with:
                - 'frames': Input frames [B, T, 3, H, W]
                - 'masks': Target masks [B, T, N, H, W]
                - 'classes': Target classes [B, N] (for VIS/VPS)
            task: Task name
        
        Returns:
            Dictionary with loss values
        """
        # Move batch to device
        frames = batch['frames'].to(self.device)
        target_masks = batch['masks'].to(self.device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # Model forward
            outputs = self.model(frames, task=task)
            
            pred_masks = outputs['masks']  # [B, T, N, H, W]
            
            # Resize predictions to match targets
            if pred_masks.shape[-2:] != target_masks.shape[-2:]:
                B, T, N = pred_masks.shape[:3]
                pred_masks = pred_masks.flatten(0, 1)  # [B*T, N, H, W]
                pred_masks = F.interpolate(
                    pred_masks,
                    size=target_masks.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
                pred_masks = pred_masks.reshape(B, T, N, *target_masks.shape[-2:])
            
            # Compute loss
            if task in ['vis', 'vps']:
                pred_classes = outputs['classes']  # [B, N, C]
                target_classes = batch['classes'].to(self.device)
                
                losses = self.loss_computer.compute_loss(
                    pred_masks.flatten(0, 1),  # [B*T, N, H, W]
                    target_masks.flatten(0, 1),
                    pred_classes.flatten(0, 1),
                    target_classes.flatten(0, 1),
                    task=task
                )
            else:
                losses = self.loss_computer.compute_loss(
                    pred_masks.flatten(0, 1),
                    target_masks.flatten(0, 1),
                    task=task
                )
            
            # Scale loss for gradient accumulation
            loss = losses['total_loss'] / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Extract loss values
        loss_dict = {k: v.item() for k, v in losses.items()}
        
        return loss_dict
    
    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.use_amp:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )
            
            # Optimizer step
            self.optimizer.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
    
    def train_epoch(
        self,
        dataloaders: Dict[str, DataLoader],
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch across multiple tasks.
        
        Args:
            dataloaders: Dictionary of dataloaders per task
            epoch: Current epoch number
        
        Returns:
            Dictionary with average losses per task
        """
        self.model.train()
        self.epoch = epoch
        
        epoch_losses = defaultdict(lambda: defaultdict(float))
        epoch_counts = defaultdict(int)
        
        # Iterate through all dataloaders in round-robin fashion
        iterators = {task: iter(loader) for task, loader in dataloaders.items()}
        active_tasks = list(iterators.keys())
        
        pbar_desc = f"Epoch {epoch}"
        step_in_epoch = 0
        
        while active_tasks:
            for task in list(active_tasks):
                try:
                    # Get next batch
                    batch = next(iterators[task])
                    
                    # Training step
                    loss_dict = self.train_step(batch, task=task)
                    
                    # Accumulate losses
                    for k, v in loss_dict.items():
                        epoch_losses[task][k] += v
                    epoch_counts[task] += 1
                    
                    # Optimizer step (every N accumulation steps)
                    step_in_epoch += 1
                    if step_in_epoch % self.gradient_accumulation_steps == 0:
                        self.optimizer_step()
                        self.global_step += 1
                    
                    # Log progress
                    if self.global_step % 10 == 0:
                        avg_loss = epoch_losses[task]['total_loss'] / epoch_counts[task]
                        print(f"\r{pbar_desc} | Step {self.global_step} | "
                              f"{task.upper()}: loss={avg_loss:.4f}", end='')
                
                except StopIteration:
                    # This dataloader is exhausted
                    active_tasks.remove(task)
        
        print()  # New line after progress
        
        # Compute average losses
        avg_losses = {}
        for task in dataloaders.keys():
            if epoch_counts[task] > 0:
                avg_losses[task] = {
                    k: v / epoch_counts[task]
                    for k, v in epoch_losses[task].items()
                }
        
        return avg_losses
    
    def validate(
        self,
        dataloaders: Dict[str, DataLoader],
        metrics_fn: Optional[callable] = None
    ) -> Dict[str, float]:
        """
        Validate on multiple tasks.
        
        Args:
            dataloaders: Validation dataloaders per task
            metrics_fn: Function to compute metrics (if None, use loss)
        
        Returns:
            Dictionary with validation metrics per task
        """
        self.model.eval()
        
        val_losses = defaultdict(lambda: defaultdict(float))
        val_counts = defaultdict(int)
        
        with torch.no_grad():
            for task, loader in dataloaders.items():
                for batch in loader:
                    # Move to device
                    frames = batch['frames'].to(self.device)
                    target_masks = batch['masks'].to(self.device)
                    
                    # Forward pass
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        outputs = self.model(frames, task=task)
                        pred_masks = outputs['masks']
                        
                        # Resize if needed
                        if pred_masks.shape[-2:] != target_masks.shape[-2:]:
                            B, T, N = pred_masks.shape[:3]
                            pred_masks = pred_masks.flatten(0, 1)
                            pred_masks = F.interpolate(
                                pred_masks,
                                size=target_masks.shape[-2:],
                                mode='bilinear',
                                align_corners=False
                            )
                            pred_masks = pred_masks.reshape(B, T, N, *target_masks.shape[-2:])
                        
                        # Compute loss
                        if task in ['vis', 'vps']:
                            pred_classes = outputs['classes']
                            target_classes = batch['classes'].to(self.device)
                            losses = self.loss_computer.compute_loss(
                                pred_masks.flatten(0, 1),
                                target_masks.flatten(0, 1),
                                pred_classes.flatten(0, 1),
                                target_classes.flatten(0, 1),
                                task=task
                            )
                        else:
                            losses = self.loss_computer.compute_loss(
                                pred_masks.flatten(0, 1),
                                target_masks.flatten(0, 1),
                                task=task
                            )
                    
                    # Accumulate
                    for k, v in losses.items():
                        val_losses[task][k] += v.item()
                    val_counts[task] += 1
        
        # Average losses
        avg_losses = {}
        for task in dataloaders.keys():
            if val_counts[task] > 0:
                avg_losses[task] = {
                    k: v / val_counts[task]
                    for k, v in val_losses[task].items()
                }
        
        return avg_losses
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Validation metrics
            is_best: Whether this is the best model so far
        """
        if not self.checkpoint_dir:
            return
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best checkpoint (metric: {self.best_metric:.4f})")
        
        # Save periodic
        if epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Loaded checkpoint from epoch {self.epoch}")
    
    def train(
        self,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Dict[str, DataLoader],
        num_epochs: int,
        eval_every: int = 1,
        save_every: int = 1
    ):
        """
        Full training loop.
        
        Args:
            train_loaders: Training dataloaders per task
            val_loaders: Validation dataloaders per task
            num_epochs: Number of epochs to train
            eval_every: Evaluate every N epochs
            save_every: Save checkpoint every N epochs
        """
        print("=" * 60)
        print("Starting TarViS Training")
        print("=" * 60)
        print(f"Tasks: {list(train_loaders.keys())}")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Gradient Accumulation: {self.gradient_accumulation_steps}")
        print("=" * 60)
        
        for epoch in range(self.epoch + 1, num_epochs + 1):
            start_time = time.time()
            
            # Train epoch
            train_losses = self.train_epoch(train_loaders, epoch)
            
            # Print training losses
            print(f"\n📊 Epoch {epoch} Training Losses:")
            for task, losses in train_losses.items():
                loss_str = ", ".join([f"{k}={v:.4f}" for k, v in losses.items()])
                print(f"  {task.upper()}: {loss_str}")
            
            # Validation
            if epoch % eval_every == 0:
                print(f"\n🔍 Validating...")
                val_losses = self.validate(val_loaders)
                
                print(f"📊 Epoch {epoch} Validation Losses:")
                for task, losses in val_losses.items():
                    loss_str = ", ".join([f"{k}={v:.4f}" for k, v in losses.items()])
                    print(f"  {task.upper()}: {loss_str}")
                
                # Check if best
                avg_val_loss = sum(
                    losses['total_loss'] 
                    for losses in val_losses.values()
                ) / len(val_losses)
                
                is_best = avg_val_loss < self.best_metric or self.best_metric == 0.0
                if is_best:
                    self.best_metric = avg_val_loss
            else:
                is_best = False
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch, train_losses, is_best)
            
            epoch_time = time.time() - start_time
            print(f"\n⏱️  Epoch {epoch} completed in {epoch_time:.2f}s\n")
        
        print("=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)


# Example usage
if __name__ == "__main__":
    if TORCH_AVAILABLE:
        print("=== TarViS Trainer Demo ===\n")
        
        # Create dummy model
        class DummyModel(nn.Module):
            def forward(self, frames, task='vos'):
                B, T, _, H, W = frames.shape
                masks = torch.randn(B, T, 3, H//4, W//4)
                outputs = {'masks': masks}
                if task in ['vis', 'vps']:
                    outputs['classes'] = torch.randn(B, 3, 40)
                return outputs
        
        model = DummyModel()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        
        trainer = TarvisTrainer(
            model=model,
            optimizer=optimizer,
            device='cpu',  # Use CPU for demo
            use_amp=False
        )
        
        print("✅ Trainer initialized successfully!")
        print(f"   Device: {trainer.device}")
        print(f"   Global step: {trainer.global_step}")
        print(f"   Epoch: {trainer.epoch}")
    else:
        print("PyTorch not available - cannot demo trainer")