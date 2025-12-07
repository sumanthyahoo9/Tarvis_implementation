#!/usr/bin/env python3
"""
scripts/train.py

Main training script for TarViS.

Usage:
    python scripts/train.py --config configs/training_config.yaml
    python scripts/train.py --config configs/training_config.yaml --resume checkpoints/latest.pt
    python scripts/train.py --config configs/training_config.yaml --debug
"""

import argparse
import sys
from pathlib import Path
import random
import numpy as np
import yaml
from src.training.tarvis_trainer import TarvisTrainer, LossComputer

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not available. Please install PyTorch.")
    sys.exit(1)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config: dict) -> nn.Module:
    """
    Build TarViS model from config.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        TarViS model
    """
    # TODO: Import and build actual TarViS model
    # For now, return dummy model
    
    class DummyTarVisModel(nn.Module):
        """Placeholder TarViS model."""
        
        def __init__(self, config):
            super().__init__()
            self.config = config
            hidden_dim = config['model']['hidden_dim']
            
            # Dummy layers
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1)
            )
            
            self.decoder = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.ReLU()
            )
            
            self.mask_head = nn.Conv2d(hidden_dim, 100, 1)  # Max 100 instances
            self.class_head = nn.Linear(hidden_dim, 124)  # Max 124 classes
        
        def forward(self, frames, task='vos'):
            B, T, C, H, W = frames.shape
            
            # Process frames
            features = []
            for t in range(T):
                feat = self.backbone(frames[:, t])  # [B, D, H', W']
                features.append(feat)
            
            # Use first frame features
            feat = features[0]  # [B, D, H', W']
            
            # Decode
            decoded = self.decoder(feat)
            
            # Predict masks
            mask_logits = self.mask_head(decoded)  # [B, N, H', W']
            
            # Expand to all frames (simplified)
            N = mask_logits.shape[1]
            mask_logits = mask_logits.unsqueeze(1).expand(B, T, N, *mask_logits.shape[2:])
            
            outputs = {'masks': mask_logits}
            
            # Add classes for VIS/VPS
            if task in ['vis', 'vps']:
                # Pool features
                pooled = decoded.mean(dim=[2, 3])  # [B, D]
                class_logits = self.class_head(pooled)  # [B, C]
                class_logits = class_logits.unsqueeze(1).expand(B, N, -1)
                outputs['classes'] = class_logits
            
            return outputs
    
    print("🏗️  Building TarViS model...")
    model = DummyTarVisModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


def build_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """Build optimizer from config."""
    opt_config = config['optimizer']
    
    # Parameter groups
    param_groups = []
    
    # Backbone with lower LR
    backbone_params = []
    other_params = []
    no_decay = set(opt_config.get('no_weight_decay', []))
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': opt_config['lr'] * opt_config.get('backbone_lr_multiplier', 0.1)
        })
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': opt_config['lr']
        })
    
    # Create optimizer
    if opt_config['name'] == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas']
        )
    elif opt_config['name'] == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=opt_config['lr'],
            weight_decay=opt_config['weight_decay'],
            betas=opt_config['betas']
        )
    elif opt_config['name'] == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=opt_config['lr'],
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_config['name']}")
    
    print(f"📊 Optimizer: {opt_config['name']}")
    print(f"   Learning rate: {opt_config['lr']}")
    print(f"   Weight decay: {opt_config['weight_decay']}")
    
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """Build learning rate scheduler from config."""
    sched_config = config['scheduler']
    num_epochs = config['training']['num_epochs']
    
    if sched_config['name'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - sched_config.get('warmup_epochs', 0),
            eta_min=sched_config.get('min_lr', 1e-7)
        )
    elif sched_config['name'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sched_config['step_size'],
            gamma=sched_config['gamma']
        )
    elif sched_config['name'] == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    else:
        raise ValueError(f"Unknown scheduler: {sched_config['name']}")
    
    print(f"📈 Scheduler: {sched_config['name']}")
    
    return scheduler


def build_dataloaders(config: dict, split: str = 'train') -> dict:
    """
    Build dataloaders for all datasets.
    
    Args:
        config: Configuration dictionary
        split: 'train' or 'val'
    
    Returns:
        Dictionary of dataloaders per task
    """
    # TODO: Import and build actual datasets
    # For now, return dummy dataloaders
    
    print(f"📦 Building {split} dataloaders...")
    
    datasets_config = config['data'][f'{split}_datasets']
    dataloaders = {}
    
    class DummyVideoDataset:
        """Dummy dataset for testing."""
        
        def __init__(self, task, num_samples=100):
            self.task = task
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Return dummy batch
            frames = torch.randn(5, 3, 480, 854)
            masks = torch.randint(0, 2, (5, 3, 120, 214)).float()
            
            batch = {
                'frames': frames,
                'masks': masks
            }
            
            if self.task in ['vis', 'vps']:
                num_classes = config['tasks'][self.task]['num_classes']
                batch['classes'] = torch.randint(0, num_classes, (3,))
            
            return batch
    
    for ds_config in datasets_config:
        task = ds_config['task']
        
        # Create dataset
        dataset = DummyVideoDataset(task=task, num_samples=100 if split == 'train' else 20)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=config['data']['batch_size'],
            shuffle=(split == 'train'),
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        dataloaders[task] = dataloader
        print(f"   {task.upper()}: {len(dataset)} samples")
    
    return dataloaders


def main(args):
    """Main training function."""
    
    # Load configuration
    print("=" * 80)
    print("TarViS Training")
    print("=" * 80)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    print(f"📄 Loaded config from: {config_path}")
    
    # Debug mode
    if args.debug or config['debug'].get('enabled', False):
        print("\n⚠️  DEBUG MODE ENABLED")
        config['training']['num_epochs'] = 2
        config['data']['num_workers'] = 0
        print()
    
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    print(f"🎲 Random seed: {seed}")
    
    # Build model
    model = build_model(config)
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Build scheduler
    scheduler = build_scheduler(optimizer, config)
    
    # Build loss computer
    loss_config = config['loss']
    loss_computer = LossComputer(
        mask_weight=loss_config['mask_weight'],
        dice_weight=loss_config['dice_weight'],
        class_weight=loss_config['class_weight'],
        use_focal_loss=loss_config['use_focal_loss'],
        focal_alpha=loss_config['focal_alpha'],
        focal_gamma=loss_config['focal_gamma']
    )
    
    # Build dataloaders
    train_loaders = build_dataloaders(config, split='train')
    val_loaders = build_dataloaders(config, split='val')
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_save_path = checkpoint_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"💾 Saved config to: {config_save_path}")
    
    # Build trainer
    device = config['hardware']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device = 'cpu'
    
    trainer = TarvisTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_computer=loss_computer,
        device=device,
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        use_amp=config['training']['use_amp'],
        checkpoint_dir=checkpoint_dir
    )
    
    # Resume from checkpoint
    if args.resume or config['training'].get('resume_from'):
        resume_path = Path(args.resume or config['training']['resume_from'])
        if resume_path.exists():
            print(f"📂 Resuming from: {resume_path}")
            trainer.load_checkpoint(resume_path)
        else:
            print(f"⚠️  Checkpoint not found: {resume_path}")
    
    # TensorBoard
    tensorboard_dir = Path(config['training']['tensorboard_dir'])
    writer = SummaryWriter(log_dir=tensorboard_dir)
    print(f"📊 TensorBoard logs: {tensorboard_dir}")
    
    # Training info
    print("\n" + "=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(f"Tasks: {list(train_loaders.keys())}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Gradient accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"Effective batch size: {config['data']['batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"Device: {device}")
    print(f"Mixed precision: {config['training']['use_amp']}")
    print("=" * 80)
    
    # Start training
    try:
        trainer.train(
            train_loaders=train_loaders,
            val_loaders=val_loaders,
            num_epochs=config['training']['num_epochs'],
            eval_every=config['training']['eval_every'],
            save_every=config['training']['save_every']
        )
        
        print("\n" + "=" * 80)
        print("✅ Training completed successfully!")
        print("=" * 80)
        print(f"Best checkpoint: {checkpoint_dir / 'checkpoint_best.pt'}")
        print(f"Latest checkpoint: {checkpoint_dir / 'checkpoint_latest.pt'}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print(f"Saving checkpoint to: {checkpoint_dir / 'checkpoint_interrupted.pt'}")
        trainer.save_checkpoint(
            epoch=trainer.epoch,
            metrics={'interrupted': True},
            is_best=False
        )
    except Exception as e:
        print(f"\n\n❌ Training failed with error:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TarViS model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_config.yaml',
        help='Path to training config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (2 epochs, no multiprocessing)'
    )
    
    args = parser.parse_args()
    main(args)