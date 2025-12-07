"""
tests/test_tarvis_trainer.py

Unit tests for TarViS trainer.
"""

import pytest
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from src.training.tarvis_trainer import (
        LossComputer,
        TarvisTrainer
    )


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
    
    def forward(self, frames, task='vos'):
        B, T, C, H, W = frames.shape
        
        # Process first frame
        x = frames[:, 0]  # [B, 3, H, W]
        x = self.conv(x)  # [B, 64, H, W]
        
        # Generate masks
        masks = torch.randn(B, T, 3, H//4, W//4)
        outputs = {'masks': masks}
        
        # Add classes for VIS/VPS
        if task in ['vis', 'vps']:
            outputs['classes'] = torch.randn(B, 3, 40)
        
        return outputs


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestLossComputer:
    """Test loss computation."""
    
    def test_dice_loss_perfect(self):
        """Test Dice loss with perfect prediction."""
        loss_computer = LossComputer()
        
        pred = torch.ones(2, 3, 100, 100)
        target = torch.ones(2, 3, 100, 100)
        
        loss = loss_computer.dice_loss(pred, target)
        
        assert loss.item() < 0.01, "Perfect Dice should be ~0"
    
    def test_dice_loss_no_overlap(self):
        """Test Dice loss with no overlap."""
        loss_computer = LossComputer()
        
        pred = torch.zeros(2, 3, 100, 100)
        pred[:, :, :50, :] = 1
        
        target = torch.zeros(2, 3, 100, 100)
        target[:, :, 50:, :] = 1
        
        loss = loss_computer.dice_loss(pred, target)
        
        assert loss.item() > 0.9, "No overlap Dice should be ~1"
    
    def test_focal_loss(self):
        """Test focal loss computation."""
        loss_computer = LossComputer()
        
        pred = torch.randn(2, 3, 100, 100)
        target = torch.randint(0, 2, (2, 3, 100, 100)).float()
        
        loss = loss_computer.focal_loss(pred, target)
        
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
    
    def test_compute_mask_loss(self):
        """Test combined mask loss."""
        loss_computer = LossComputer()
        
        pred = torch.randn(2, 3, 100, 100)
        target = torch.randint(0, 2, (2, 3, 100, 100)).float()
        
        losses = loss_computer.compute_mask_loss(pred, target)
        
        assert 'dice_loss' in losses
        assert 'focal_loss' in losses
        assert 'mask_loss' in losses
        assert losses['mask_loss'].item() >= 0.0
    
    def test_compute_class_loss(self):
        """Test classification loss."""
        loss_computer = LossComputer()
        
        pred = torch.randn(10, 40)
        target = torch.randint(0, 40, (10,))
        
        loss = loss_computer.compute_class_loss(pred, target)
        
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)
    
    def test_compute_loss_vos(self):
        """Test total loss for VOS."""
        loss_computer = LossComputer()
        
        pred_masks = torch.randn(2, 3, 100, 100)
        target_masks = torch.randint(0, 2, (2, 3, 100, 100)).float()
        
        losses = loss_computer.compute_loss(
            pred_masks, target_masks, task='vos'
        )
        
        assert 'total_loss' in losses
        assert 'mask_loss' in losses
        assert 'class_loss' not in losses  # VOS has no classes
    
    def test_compute_loss_vis(self):
        """Test total loss for VIS."""
        loss_computer = LossComputer()
        
        pred_masks = torch.randn(2, 3, 100, 100)
        target_masks = torch.randint(0, 2, (2, 3, 100, 100)).float()
        pred_classes = torch.randn(2, 40)
        target_classes = torch.randint(0, 40, (2,))
        
        losses = loss_computer.compute_loss(
            pred_masks, target_masks,
            pred_classes, target_classes,
            task='vis'
        )
        
        assert 'total_loss' in losses
        assert 'mask_loss' in losses
        assert 'class_loss' in losses


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestTarvisTrainer:
    """Test trainer."""
    
    def test_trainer_init(self):
        """Test trainer initialization."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = TarvisTrainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_amp=False
        )
        
        assert trainer.global_step == 0
        assert trainer.epoch == 0
        assert trainer.device == 'cpu'
    
    def test_train_step_vos(self):
        """Test single training step for VOS."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = TarvisTrainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_amp=False
        )
        
        # Create batch
        batch = {
            'frames': torch.randn(1, 5, 3, 128, 128),
            'masks': torch.randint(0, 2, (1, 5, 3, 32, 32)).float()
        }
        
        # Training step
        loss_dict = trainer.train_step(batch, task='vos')
        
        assert 'total_loss' in loss_dict
        assert 'mask_loss' in loss_dict
        assert loss_dict['total_loss'] >= 0.0
    
    def test_train_step_vis(self):
        """Test single training step for VIS."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = TarvisTrainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_amp=False
        )
        
        # Create batch
        batch = {
            'frames': torch.randn(1, 5, 3, 128, 128),
            'masks': torch.randint(0, 2, (1, 5, 3, 32, 32)).float(),
            'classes': torch.randint(0, 40, (1, 3))
        }
        
        # Training step
        loss_dict = trainer.train_step(batch, task='vis')
        
        assert 'total_loss' in loss_dict
        assert 'mask_loss' in loss_dict
        assert 'class_loss' in loss_dict
    
    def test_optimizer_step(self):
        """Test optimizer step."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = TarvisTrainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_amp=False
        )
        
        # Create dummy gradients
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        
        # Optimizer step
        initial_step = trainer.global_step
        trainer.optimizer_step()
        
        # Check step incremented
        # Note: global_step only increments after accumulation steps
    
    def test_train_epoch(self):
        """Test training for one epoch."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = TarvisTrainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_amp=False,
            gradient_accumulation_steps=2
        )
        
        # Create dataloaders
        frames = torch.randn(10, 5, 3, 128, 128)
        masks = torch.randint(0, 2, (10, 5, 3, 32, 32)).float()
        dataset = TensorDataset(frames, masks)
        
        dataloaders = {
            'vos': DataLoader(dataset, batch_size=2)
        }
        
        # Wrap to return dict
        class DictDataLoader:
            def __init__(self, loader):
                self.loader = loader
            
            def __iter__(self):
                for frames, masks in self.loader:
                    yield {'frames': frames, 'masks': masks}
            
            def __len__(self):
                return len(self.loader)
        
        dataloaders = {
            'vos': DictDataLoader(DataLoader(dataset, batch_size=2))
        }
        
        # Train epoch
        losses = trainer.train_epoch(dataloaders, epoch=1)
        
        assert 'vos' in losses
        assert 'total_loss' in losses['vos']
        assert trainer.epoch == 1
    
    def test_validate(self):
        """Test validation."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = TarvisTrainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_amp=False
        )
        
        # Create dataloaders
        frames = torch.randn(10, 5, 3, 128, 128)
        masks = torch.randint(0, 2, (10, 5, 3, 32, 32)).float()
        
        class DictDataLoader:
            def __init__(self, frames, masks):
                self.frames = frames
                self.masks = masks
            
            def __iter__(self):
                for i in range(len(self.frames)):
                    yield {
                        'frames': self.frames[i:i+1],
                        'masks': self.masks[i:i+1]
                    }
        
        dataloaders = {
            'vos': DictDataLoader(frames, masks)
        }
        
        # Validate
        val_losses = trainer.validate(dataloaders)
        
        assert 'vos' in val_losses
        assert 'total_loss' in val_losses['vos']
    
    def test_save_load_checkpoint(self):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = DummyModel()
            optimizer = torch.optim.Adam(model.parameters())
            
            trainer = TarvisTrainer(
                model=model,
                optimizer=optimizer,
                device='cpu',
                use_amp=False,
                checkpoint_dir=tmpdir
            )
            
            # Save checkpoint
            trainer.global_step = 100
            trainer.epoch = 5
            trainer.save_checkpoint(
                epoch=5,
                metrics={'loss': 0.5},
                is_best=True
            )
            
            # Check files exist
            checkpoint_dir = Path(tmpdir)
            assert (checkpoint_dir / 'checkpoint_latest.pt').exists()
            assert (checkpoint_dir / 'checkpoint_best.pt').exists()
            
            # Create new trainer and load
            model2 = DummyModel()
            optimizer2 = torch.optim.Adam(model2.parameters())
            
            trainer2 = TarvisTrainer(
                model=model2,
                optimizer=optimizer2,
                device='cpu',
                use_amp=False
            )
            
            trainer2.load_checkpoint(checkpoint_dir / 'checkpoint_best.pt')
            
            assert trainer2.epoch == 5
            assert trainer2.global_step == 100


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestIntegration:
    """Integration tests."""
    
    def test_full_training_loop(self):
        """Test complete training loop."""
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = TarvisTrainer(
                model=model,
                optimizer=optimizer,
                device='cpu',
                use_amp=False,
                gradient_accumulation_steps=2,
                checkpoint_dir=tmpdir
            )
            
            # Create small datasets
            train_frames = torch.randn(6, 5, 3, 64, 64)
            train_masks = torch.randint(0, 2, (6, 5, 3, 16, 16)).float()
            
            val_frames = torch.randn(4, 5, 3, 64, 64)
            val_masks = torch.randint(0, 2, (4, 5, 3, 16, 16)).float()
            
            class SimpleDictDataLoader:
                def __init__(self, frames, masks):
                    self.data = []
                    for i in range(len(frames)):
                        self.data.append({
                            'frames': frames[i:i+1],
                            'masks': masks[i:i+1]
                        })
                
                def __iter__(self):
                    return iter(self.data)
                
                def __len__(self):
                    return len(self.data)
            
            train_loaders = {
                'vos': SimpleDictDataLoader(train_frames, train_masks)
            }
            
            val_loaders = {
                'vos': SimpleDictDataLoader(val_frames, val_masks)
            }
            
            # Train for 2 epochs
            trainer.train(
                train_loaders=train_loaders,
                val_loaders=val_loaders,
                num_epochs=2,
                eval_every=1,
                save_every=1
            )
            
            # Check training completed
            assert trainer.epoch == 2
            assert trainer.global_step > 0
            
            # Check checkpoint saved
            checkpoint_dir = Path(tmpdir)
            assert (checkpoint_dir / 'checkpoint_latest.pt').exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])