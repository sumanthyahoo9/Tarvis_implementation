# TarViS Training Guide

## Quick Start

### 1. Prepare Data

Download datasets:
```bash
# DAVIS 2017 (VOS)
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip

# YouTube-VIS 2021 (VIS)
# Download from: https://youtube-vos.org/dataset/vis/

# VIPSeg (VPS)
# Download from: https://github.com/VIPSeg-Dataset/VIPSeg-Dataset
```

Update dataset paths in `configs/training_config.yaml`:
```yaml
data:
  train_datasets:
    - name: "davis"
      root: "/path/to/DAVIS"  # ← Update this
```

### 2. Quick Test Training

Test with small config (fast):
```bash
python scripts/train.py --config configs/quick_train.yaml --debug
```

This runs for 2 epochs with minimal settings to verify everything works.

### 3. Full Training

Once verified, run full training:
```bash
python scripts/train.py --config configs/training_config.yaml
```

### 4. Resume Training

If training is interrupted:
```bash
python scripts/train.py \
    --config configs/training_config.yaml \
    --resume checkpoints/tarvis/checkpoint_latest.pt
```

---

## Training Configurations

### Available Configs

**1. `training_config.yaml`** - Full training
- All tasks (VOS, VIS, VPS)
- Full model (256 hidden dim, 6 decoder layers)
- 50 epochs
- Multi-task learning

**2. `quick_train.yaml`** - Quick testing
- Single task (VOS)
- Small model (128 hidden dim, 3 decoder layers)
- 10 epochs
- Fast iteration

### Creating Custom Config

Copy and modify:
```bash
cp configs/training_config.yaml configs/my_config.yaml
# Edit my_config.yaml
python scripts/train.py --config configs/my_config.yaml
```

---

## Configuration Options

### Model Settings

```yaml
model:
  backbone: "resnet50"      # resnet18/34/50/101
  hidden_dim: 256           # Feature dimension
  num_decoder_layers: 6     # Transformer layers
  num_heads: 8              # Attention heads
  dim_feedforward: 2048     # FFN dimension
```

**Memory impact:**
- `hidden_dim`: Linear with memory
- `num_decoder_layers`: Linear with memory
- `backbone`: ResNet-50 uses 2x memory vs ResNet-18

### Data Settings

```yaml
data:
  batch_size: 2             # Per-GPU batch size
  num_workers: 4            # Data loading threads
  num_frames: 5             # Frames per video clip
  frame_stride: 2           # Skip frames
```

**Batch size guidelines:**
- **16GB GPU:** batch_size=1-2
- **24GB GPU:** batch_size=2-4
- **32GB GPU:** batch_size=4-8

Use gradient accumulation for larger effective batch:
```yaml
training:
  gradient_accumulation_steps: 4  # Effective batch = 2 * 4 = 8
```

### Training Settings

```yaml
training:
  num_epochs: 50
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0        # Gradient clipping
  use_amp: true             # Mixed precision (fp16)
  eval_every: 1             # Validate every N epochs
  save_every: 5             # Save every N epochs
```

### Optimizer Settings

```yaml
optimizer:
  name: "adamw"             # adamw/adam/sgd
  lr: 1.0e-4                # Learning rate
  weight_decay: 1.0e-4
  backbone_lr_multiplier: 0.1  # Lower LR for backbone
```

**Learning rate guidelines:**
- Start with `1e-4`
- If loss oscillates: reduce to `5e-5`
- If training slow: increase to `2e-4`
- Backbone should use 10x lower LR

### Loss Settings

```yaml
loss:
  mask_weight: 5.0          # Binary cross-entropy weight
  dice_weight: 5.0          # Dice loss weight
  class_weight: 2.0         # Classification loss weight
  use_focal_loss: true      # Handle class imbalance
```

---

## Multi-Task Training

TarViS trains on multiple tasks simultaneously:

```yaml
data:
  train_datasets:
    - name: "davis"
      task: "vos"           # Video Object Segmentation
      weight: 1.0
    
    - name: "youtube_vis_2021"
      task: "vis"           # Video Instance Segmentation
      weight: 1.0
    
    - name: "vipseg"
      task: "vps"           # Video Panoptic Segmentation
      weight: 1.0
```

**Task weights:**
- `weight: 1.0` - Equal sampling
- `weight: 2.0` - Sample twice as often
- Use weights to balance dataset sizes

**Round-robin sampling:**
Training alternates between tasks each batch.

---

## Monitoring Training

### TensorBoard

View training progress:
```bash
tensorboard --logdir runs/tarvis
```

Open browser to `http://localhost:6006`

**What to monitor:**
- **Loss curves:** Should decrease steadily
- **Learning rate:** Should decay gradually
- **Gradient norms:** Should stay below max_grad_norm

### Console Output

Training prints every 10 steps:
```
Epoch 1 | Step 100 | VOS: loss=0.5234
```

Validation results after each epoch:
```
📊 Epoch 1 Validation Losses:
  VOS: total_loss=0.4123, dice_loss=0.2156, focal_loss=0.1967
  VIS: total_loss=0.5234, mask_loss=0.3123, class_loss=0.2111
```

---

## Checkpoints

### Automatic Saving

Trainer saves:
- **`checkpoint_latest.pt`** - After every epoch
- **`checkpoint_best.pt`** - Best validation loss
- **`checkpoint_epoch_N.pt`** - Every N epochs

### Manual Saving

On interrupt (Ctrl+C):
```
⚠️ Training interrupted by user
Saving checkpoint to: checkpoint_interrupted.pt
```

### Checkpoint Contents

```python
checkpoint = {
    'epoch': 15,
    'global_step': 5000,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'scheduler_state_dict': {...},
    'metrics': {...},
    'best_metric': 0.3456
}
```

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `batch_size: 1`
2. Use gradient accumulation: `gradient_accumulation_steps: 8`
3. Reduce model size: `hidden_dim: 128`, `num_decoder_layers: 3`
4. Reduce input size: `random_crop: [256, 456]`
5. Use fewer frames: `num_frames: 3`

### Slow Training

**Solutions:**
1. Increase workers: `num_workers: 8`
2. Enable prefetching: `prefetch_factor: 4`
3. Use smaller backbone: `backbone: "resnet18"`
4. Reduce validation frequency: `eval_every: 5`

### Loss Not Decreasing

**Solutions:**
1. Check learning rate (might be too high/low)
2. Verify data loading (print some batches)
3. Try smaller `max_grad_norm: 0.5`
4. Increase warmup: `warmup_epochs: 10`
5. Check loss weights (mask/dice/class)

### NaN Loss

**Causes & fixes:**
1. **Learning rate too high** → Reduce to `5e-5`
2. **Gradient explosion** → Lower `max_grad_norm: 0.5`
3. **Mixed precision issue** → Disable `use_amp: false`
4. **Bad initialization** → Change `seed: 123`

---

## Training Time Estimates

### Single GPU (RTX 3090)

**Quick config (10 epochs):**
- ResNet-18 backbone: ~2 hours
- ResNet-50 backbone: ~4 hours

**Full config (50 epochs):**
- DAVIS only: ~20 hours
- All tasks: ~60 hours

### Multi-GPU (4x A100)

**Full config (50 epochs):**
- All tasks: ~15 hours

---

## Advanced: Distributed Training

### Multiple GPUs on Single Node

```bash
# 4 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py --config configs/training_config.yaml
```

Update config:
```yaml
hardware:
  distributed: true
  num_gpus: 4
```

### Multiple Nodes

```bash
# Node 0 (master)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=12345 \
    scripts/train.py --config configs/training_config.yaml

# Node 1 (worker)
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=12345 \
    scripts/train.py --config configs/training_config.yaml
```

---

## Debug Mode

Quick verification before long training:

```bash
python scripts/train.py --config configs/training_config.yaml --debug
```

Debug mode:
- Runs only 2 epochs
- Uses single worker (`num_workers: 0`)
- Faster iteration

Or enable in config:
```yaml
debug:
  enabled: true
  num_batches: 10              # Train on 10 batches only
  overfit_single_batch: true   # Overfit test (should reach ~0 loss)
```

---

## Best Practices

### 1. Start Small
- Test with `quick_train.yaml` first
- Verify data loads correctly
- Check loss decreases

### 2. Monitor Closely
- Watch first few epochs carefully
- Check TensorBoard regularly
- Validate loss should decrease

### 3. Save Often
- Use `save_every: 5` minimum
- Keep multiple checkpoints
- Test resuming from checkpoint

### 4. Experiment Systematically
- Change one thing at a time
- Keep notes of what works
- Use consistent seeds for comparison

### 5. Use Validation
- Don't overtrain on training set
- Stop when validation plateaus
- Use `eval_every: 1` initially

---

## Example Training Commands

### Basic Training
```bash
python scripts/train.py --config configs/training_config.yaml
```

### Quick Test
```bash
python scripts/train.py --config configs/quick_train.yaml --debug
```

### Resume Training
```bash
python scripts/train.py \
    --config configs/training_config.yaml \
    --resume checkpoints/tarvis/checkpoint_latest.pt
```

### Single Task (VOS only)
```bash
# Edit config to only include DAVIS
python scripts/train.py --config configs/vos_only.yaml
```

### Custom Config
```bash
python scripts/train.py --config configs/my_experiment.yaml
```