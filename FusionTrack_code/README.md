# FusionTrack

## Table of Contents

- [Environment Setup](#environment-setup)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [FAQ](#faq)

---

## Environment Setup

### 1. Base Environment

```bash
# Python 3.8+
# CUDA 11.3+
# PyTorch 1.12+

# Create conda environment
conda create -n fusiontrack python=3.8
conda activate fusiontrack

# Install PyTorch (adjust for your CUDA version)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Build Deformable Attention

```bash
cd models/ops
python setup.py build install
cd ../..
```

---

## Dataset Preparation

### Supported Datasets

- **UAV_V**: Multi-view UAV dataset (primary)


### UAV_V Dataset Structure

```
dataset/
└── UAV_V/
    └── images/
        ├── train/           # Training set
        │   ├── seq1/
        │   │   ├── c001/    # View 1
        │   │   │   ├── 000001.jpg
        │   │   │   ├── 000002.jpg
        │   │   │   └── ...
        │   │   ├── c002/    # View 2
        │   │   ├── c003/    # View 3
        │   │   ├── c004/    # View 4
        │   │   └── c005/    # View 5
        │   ├── seq2/
        │   └── ...
        ├── val/             # Validation set (same structure as train)
        └── test/            # Test set (same structure as train)
```

### Annotation Format

Each sequence requires GT annotation files (MOT format):

```
dataset/UAV_V/images/train/seq1/gt/gt.txt
```

Annotation format:
```
<frame_id>,<track_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
```

---

## Training

### 1. Basic Training Command

```bash
# Single-GPU training
python main.py --mode train --config configs/train_uav_multiview_reid.yaml

# Multi-GPU distributed training (recommended)
CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --mode train \
    --config configs/train_uav_multiview_reid.yaml \
    --use-distributed
```

### 2. Resume Training from Checkpoint

```bash
python main.py \
    --mode train \
    --config configs/train_uav_multiview_reid.yaml \
    --resume1 outputs/uav_multiview_reid/model_checkpointpth \
    --resume2 outputs/uav_multiview_reid/reid_checkpoint.pth
```

### 3. Training Configuration

Main config file: `configs/train_uav_multiview_reid.yaml`


### 4. Training Outputs

```
outputs/
└── uav_multiview_reid/
    ├── train/
    │   ├── config.yaml          # Backed-up training config
    │   └── log.txt              # Training log
    ├── model_checkpoint_*.pth   # Main model checkpoints
    ├── reid_checkpoint_*.pth    # ReID model checkpoints
    └── optimizer_checkpoint_*.pth  # Optimizer state
```

---

## Inference

### 1. Basic Inference Command

```bash
python main.py \
    --mode submit \
    --config configs/submit_uav_multiview_reid.yaml \
    --submit-dir outputs/uav_multiview_reid \
    --submit-model1 model_checkpoint.pth \
    --submit-model2 reid_checkpoint.pth \
    --submit-data-split test
```

### 2. Inference Configuration

Main config file: `configs/submit_uav_multiview_reid.yaml`

### 3. Inference Outputs

```
outputs/uav_multiview_reid/test/
└── tracker/
    ├── seq1_c001.txt        # Cross-view association results
    ├── seq1_c002.txt
    ├── seq1_c001_single.txt # Single-view results (for comparison)
    ├── seq1_c002_single.txt
    └── ...
```

Output format (MOT format):
```
<frame_id>,<track_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
```
