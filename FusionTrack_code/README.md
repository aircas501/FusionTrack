# FusionTrack

## 目录

- [环境配置](#环境配置)
- [数据集准备](#数据集准备)
- [训练](#训练)
- [推理](#推理)
- [评估](#评估)
- [配置说明](#配置说明)
- [常见问题](#常见问题)

---

## 环境配置

### 1. 基础环境

```bash
# Python 3.8+
# CUDA 11.3+
# PyTorch 1.12+

# 创建conda环境
conda create -n fusiontrack python=3.8
conda activate fusiontrack

# 安装PyTorch（根据你的CUDA版本调整）
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 编译Deformable Attention

```bash
cd models/ops
python setup.py build install
cd ../..
```

---

## 数据集准备

### 支持的数据集

- **UAV_V**: 多视角无人机数据集（主要）


### UAV_V 数据集结构

```
dataset/
└── UAV_V/
    └── images/
        ├── train/           # 训练集
        │   ├── seq1/
        │   │   ├── c001/    # 视角1
        │   │   │   ├── 000001.jpg
        │   │   │   ├── 000002.jpg
        │   │   │   └── ...
        │   │   ├── c002/    # 视角2
        │   │   ├── c003/    # 视角3
        │   │   ├── c004/    # 视角4
        │   │   └── c005/    # 视角5
        │   ├── seq2/
        │   └── ...
        ├── val/             # 验证集（结构同train）
        └── test/            # 测试集（结构同train）
```

### 标注文件格式

每个序列需要包含GT标注文件（MOT格式）：

```
dataset/UAV_V/images/train/seq1/gt/gt.txt
```

标注格式：
```
<frame_id>,<track_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
```

---

## 训练

### 1. 基础训练命令

```bash
# 单GPU训练
python main.py --mode train --config configs/train_uav_multiview_reid.yaml

# 多GPU分布式训练（推荐）
CUDA_VISIBLE_DEVICES=0,1,2 python main.py \
    --mode train \
    --config configs/train_uav_multiview_reid.yaml \
    --use-distributed
```

### 2. 从checkpoint恢复训练

```bash
python main.py \
    --mode train \
    --config configs/train_uav_multiview_reid.yaml \
    --resume1 outputs/uav_multiview_reid/model_checkpointpth \
    --resume2 outputs/uav_multiview_reid/reid_checkpoint.pth
```

### 3. 训练配置文件

主要配置文件：`configs/train_uav_multiview_reid.yaml`


### 4. 训练输出

```
outputs/
└── uav_multiview_reid/
    ├── train/
    │   ├── config.yaml          # 训练配置备份
    │   └── log.txt              # 训练日志
    ├── model_checkpoint_*.pth   # 主模型检查点
    ├── reid_checkpoint_*.pth    # ReID模型检查点
    └── optimizer_checkpoint_*.pth  # 优化器状态
```

---

## 推理

### 1. 基础推理命令

```bash
python main.py \
    --mode submit \
    --config configs/submit_uav_multiview_reid.yaml \
    --submit-dir outputs/uav_multiview_reid \
    --submit-model1 model_checkpoint.pth \
    --submit-model2 reid_checkpoint.pth \
    --submit-data-split test
```

### 2. 推理配置文件

主要配置文件：`configs/submit_uav_multiview_reid.yaml`
```

### 3. 推理输出

```
outputs/uav_multiview_reid/test/
└── tracker/
    ├── seq1_c001.txt        # 跨视角关联结果
    ├── seq1_c002.txt
    ├── seq1_c001_single.txt # 单视角结果（对比）
    ├── seq1_c002_single.txt
    └── ...
```

输出格式（MOT格式）：
```
<frame_id>,<track_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
```