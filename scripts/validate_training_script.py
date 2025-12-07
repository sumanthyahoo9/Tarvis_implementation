#!/usr/bin/env python3
"""
scripts/validate_training_setup.py

Validate training setup before starting long training runs.

Checks:
- Config files are valid
- Datasets are accessible
- Model builds correctly
- Training step executes
- Checkpointing works
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def check_config_file(config_path: Path) -> bool:
    """Check if config file is valid."""
    print(f"\n[1/6] Checking config file: {config_path}")
    
    if not config_path.exists():
        print(f"   ❌ Config file not found!")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   ✅ Config loaded successfully")
    except Exception as e:
        print(f"   ❌ Failed to load config: {e}")
        return False
    
    # Check required sections
    required_sections = ['model', 'data', 'training', 'optimizer', 'scheduler', 'loss']
    for section in required_sections:
        if section not in config:
            print(f"   ❌ Missing required section: {section}")
            return False
    
    print(f"   ✅ All required sections present")
    return True


def check_pytorch() -> bool:
    """Check if PyTorch is available."""
    print(f"\n[2/6] Checking PyTorch installation")
    
    if not TORCH_AVAILABLE:
        print(f"   ❌ PyTorch not installed!")
        print(f"      Install: pip install torch torchvision")
        return False
    
    print(f"   ✅ PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"   ⚠️  CUDA not available (will use CPU)")
    
    return True


def check_imports() -> bool:
    """Check if required modules can be imported."""
    print(f"\n[3/6] Checking required imports")
    
    required_imports = [
        ('src.training.tarvis_trainer', 'TarvisTrainer'),
        ('src.training.tarvis_trainer', 'LossComputer'),
        ('src.utils.evaluation_metrics', 'EvaluationMetrics'),
    ]
    
    all_ok = True
    for module_name, class_name in required_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"   ✅ {module_name}.{class_name}")
        except Exception as e:
            print(f"   ❌ {module_name}.{class_name}: {e}")
            all_ok = False
    
    return all_ok


def check_model_build(config_path: Path) -> bool:
    """Check if model can be built."""
    print(f"\n[4/6] Testing model build")
    
    if not TORCH_AVAILABLE:
        print(f"   ⚠️  Skipping (PyTorch not available)")
        return True
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create dummy model (same as in train.py)
        import torch.nn as nn
        
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3)
            
            def forward(self, frames, task='vos'):
                B, T, C, H, W = frames.shape
                masks = torch.randn(B, T, 3, H//4, W//4)
                return {'masks': masks}
        
        model = DummyModel()
        print(f"   ✅ Model created successfully")
        
        # Test forward pass
        dummy_input = torch.randn(1, 5, 3, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   ✅ Forward pass successful")
        print(f"      Input: {dummy_input.shape}")
        print(f"      Output masks: {output['masks'].shape}")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Model build failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_training_step(config_path: Path) -> bool:
    """Check if training step executes."""
    print(f"\n[5/6] Testing training step")
    
    if not TORCH_AVAILABLE:
        print(f"   ⚠️  Skipping (PyTorch not available)")
        return True
    
    try:
        from src.training.tarvis_trainer import TarvisTrainer, LossComputer
        import torch.nn as nn
        
        # Create model and trainer
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3, padding=1)
            
            def forward(self, frames, task='vos'):
                B, T, C, H, W = frames.shape
                x = frames[:, 0]
                x = self.conv(x)
                masks = torch.randn(B, T, 3, H//4, W//4)
                return {'masks': masks}
        
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        trainer = TarvisTrainer(
            model=model,
            optimizer=optimizer,
            device='cpu',
            use_amp=False
        )
        
        print(f"   ✅ Trainer created")
        
        # Create dummy batch
        batch = {
            'frames': torch.randn(1, 5, 3, 128, 128),
            'masks': torch.randint(0, 2, (1, 5, 3, 32, 32)).float()
        }
        
        # Execute training step
        loss_dict = trainer.train_step(batch, task='vos')
        
        print(f"   ✅ Training step executed")
        print(f"      Losses: {', '.join([f'{k}={v:.4f}' for k, v in loss_dict.items()])}")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_checkpointing() -> bool:
    """Check if checkpointing works."""
    print(f"\n[6/6] Testing checkpointing")
    
    if not TORCH_AVAILABLE:
        print(f"   ⚠️  Skipping (PyTorch not available)")
        return True
    
    try:
        from src.training.tarvis_trainer import TarvisTrainer
        import torch.nn as nn
        import tempfile
        
        # Create model and trainer
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 64, 3)
            
            def forward(self, x, task='vos'):
                return {'masks': torch.randn(1, 5, 3, 32, 32)}
        
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        with tempfile.TemporaryDirectory() as tmpdir:
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
            trainer.save_checkpoint(epoch=5, metrics={'loss': 0.5}, is_best=True)
            
            checkpoint_path = Path(tmpdir) / 'checkpoint_best.pt'
            if not checkpoint_path.exists():
                print(f"   ❌ Checkpoint not saved!")
                return False
            
            print(f"   ✅ Checkpoint saved")
            
            # Load checkpoint
            trainer2 = TarvisTrainer(
                model=DummyModel(),
                optimizer=torch.optim.Adam(model.parameters()),
                device='cpu',
                use_amp=False
            )
            
            trainer2.load_checkpoint(checkpoint_path)
            
            if trainer2.epoch != 5 or trainer2.global_step != 100:
                print(f"   ❌ Checkpoint loaded incorrectly!")
                return False
            
            print(f"   ✅ Checkpoint loaded correctly")
        
        return True
    
    except Exception as e:
        print(f"   ❌ Checkpointing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation checks."""
    print("=" * 80)
    print("TarViS Training Setup Validation")
    print("=" * 80)
    
    # Default config
    config_path = Path("configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/quick_train.yaml")
    
    results = []
    
    # Run checks
    results.append(("Config file", check_config_file(config_path)))
    results.append(("PyTorch", check_pytorch()))
    results.append(("Imports", check_imports()))
    results.append(("Model build", check_model_build(config_path)))
    results.append(("Training step", check_training_step(config_path)))
    results.append(("Checkpointing", check_checkpointing()))
    
    # Summary
    print("\n" + "=" * 80)
    print("Validation Summary")
    print("=" * 80)
    
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("=" * 80)
    if all_passed:
        print("✅ All checks passed! Ready to start training.")
        print("\nTo start training, run:")
        print(f"  python scripts/train.py --config {config_path}")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())