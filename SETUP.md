# TarViS Environment Setup Guide

This guide provides step-by-step instructions for setting up the TarViS development environment on macOS with Apple Silicon (M1/M2/M3).
Get the paper here: https://openaccess.thecvf.com/content/CVPR2023/papers/Athar_TarViS_A_Unified_Approach_for_Target-Based_Video_Segmentation_CVPR_2023_paper.pdf

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3)
- **Python 3.10** installed via Homebrew
- **Git** for cloning repositories

### Install Python 3.10 (if not already installed)
```bash
brew install python@3.10
```

Verify installation:
```bash
python3.10 --version
# Should output: Python 3.10.x
```

## Installation Steps

### 1. Create Virtual Environment

Navigate to the project directory and create a virtual environment:

```bash
cd TarViS_implementation
python3.10 -m venv tarvis_env
source tarvis_env/bin/activate
```

Your terminal prompt should now show `(tarvis_env)` indicating the virtual environment is active.

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install Cython

Cython is required for building `pycocotools` from source:

```bash
pip install cython
```

### 4. Install Main Dependencies

Install all core dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch 1.13.1 and TorchVision 0.14.1
- TensorFlow for macOS 2.13.0
- All other project dependencies

### 5. Install pycocotools

Due to build issues on macOS ARM, `pycocotools` must be installed separately:

```bash
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

### 6. Verify Installation

Test that all critical packages are installed correctly:

```bash
python -c "import torch; import torchvision; import pycocotools; import tensorflow; print('✓ All packages installed successfully!')"
```

If this runs without errors, your environment is ready!

## requirements.txt

The project uses the following dependencies:

```txt
# Core ML Frameworks
torch==1.13.1
torchvision==0.14.1

# Project Dependencies
colorama==0.4.5
einops==0.4.1
fvcore==0.1.5.post20220512
imgaug==0.4.0
numpy==1.23.5
opencv_python==4.6.0.66
git+https://github.com/cocodataset/panopticapi.git
Pillow==9.5.0
PyYAML==6.0
scipy==1.9.3
tensorflow-macos==2.13.0  # For KITTI-STEP evaluation metrics
timm==0.5.4
tqdm==4.64.0
wandb==0.13.0
```

## Platform-Specific Notes

### macOS Apple Silicon (M1/M2/M3)

- **TensorFlow**: Uses `tensorflow-macos` instead of `tensorflow-cpu` for native ARM support
- **NumPy**: Version 1.23.5 is used to satisfy both PyTorch 1.13 and TensorFlow 2.13 requirements
- **pycocotools**: Must be installed from source due to pre-built wheel incompatibilities

### Other Platforms

For Linux or Intel-based Macs, you may need to adjust:
- Replace `tensorflow-macos` with `tensorflow` or `tensorflow-cpu`
- Check PyTorch installation instructions at https://pytorch.org for your specific platform

## Troubleshooting

### Issue: numpy version conflicts

**Error**: `ERROR: ResolutionImpossible... numpy version conflict`

**Solution**: Ensure you're using Python 3.10 and following the installation order exactly (Cython → requirements.txt → pycocotools).

### Issue: pycocotools build failure

**Error**: `clang: error: no such file or directory: '../common/maskApi.c'`

**Solution**: Install pycocotools from the GitHub repository as shown in Step 5, not from PyPI.

### Issue: torch not found in requirements.txt

**Explanation**: Research repositories often assume PyTorch is installed separately. This `requirements.txt` includes PyTorch explicitly for convenience.

### Issue: Missing Command Line Tools

**Error**: `xcrun: error: invalid active developer path`

**Solution**: Install Xcode Command Line Tools:
```bash
xcode-select --install
```

## Deactivating the Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

## Reactivating the Environment

To work on the project again:

```bash
cd TarViS_implementation
source tarvis_env/bin/activate
```

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorFlow for macOS](https://developer.apple.com/metal/tensorflow-plugin/)
- [COCO API](https://github.com/cocodataset/cocoapi)

---

**Last Updated**: December 2025  
**Python Version**: 3.10  
**Platform**: macOS Apple Silicon