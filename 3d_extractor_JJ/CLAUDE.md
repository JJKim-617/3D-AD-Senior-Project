# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **RAD (Is Training Necessary for Anomaly Detection?)**, a training-free industrial anomaly detection system. It uses frozen pretrained DINOv3 vision transformers to extract multi-layer features, builds a reference memory bank from normal training images, and detects anomalies at test time via k-NN patch-level similarity matching — no model training required.

## Environment Setup

```bash
conda create -n rad python=3.11 -y
conda activate rad
cd RAD
pip install -r requirements.txt
```

Developed on NVIDIA GeForce RTX 5090 (32GB), CUDA 12.8, torch 2.7.0+cu128.

## Running the Pipeline

The pipeline has two sequential phases.

### Phase 1: Build Memory Bank

```bash
# MVTec AD
python RAD/build_bank_multilayer.py \
    --data_path ../mvtec_anomaly_detection \
    --item_list carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper \
    --encoder_name dinov3_vitb16 \
    --encoder_weight ../weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --image_size 512 --crop_size 448 --batch_size 32 --num_workers 4 \
    --layer_idx_list 3 6 9 11 \
    --bank_path ./bank/mvtec_dinov3_vitb16_multilayer36911_448_bank.pth

# Real-IAD uses a separate script:
python RAD/build_realiad_bank_multilayer.py --data_path ../Real-IAD ...
```

### Phase 2: Detection & Evaluation

```bash
# MVTec AD / VisA / 3D-ADAM
python RAD/rad_mvtec_visa_3dadam.py \
    --data_path ../mvtec_anomaly_detection \
    --bank_path ./bank/mvtec_dinov3_vitb16_multilayer36911_448_bank.pth \
    --encoder_name dinov3_vitb16 \
    --encoder_weight ../weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
    --save_dir ./saved_results_mvtec \
    --save_name mvtec_patch_knn_multilayer_36911_448 \
    --k_image 150 --resize_mask 256 --max_ratio 0.01 \
    --vis_max 8 --use_positional_bank --pos_radius 1 \
    --item_list carpet grid leather tile wood bottle cable capsule hazelnut metal_nut pill screw toothbrush transistor zipper

# Real-IAD
python RAD/rad_real_iad.py --data_path ../Real-IAD ...
```

All configuration is via command-line arguments — there are no config files.

### Dataset Preparation

```bash
# VisA preprocessing
python RAD/prepare_data/prepare_visa.py --split-type 1cls --data-folder ../VisA --save-folder ../VisA_pytorch --split-file RAD/prepare_data/split_csv/1cls.csv

# 3D-ADAM preprocessing
python RAD/prepare_data/prepare_3dadam.py --data-folder ../3D-ADAM_anomalib --save-folder ../3D-ADAM
```

Datasets are expected in `../` relative to the `RAD/` directory.

## Architecture

### Data Flow

```
Normal training images
    → Resize 512×512 + CenterCrop 448×448
    → DINOv3 ViT-B/16 (frozen, pretrained)
    → Intermediate features from layers [3, 6, 9, 11]
    → cls_banks (per-layer CLS tokens) + patch_banks (per-layer patch tokens)
    → Saved as .pth memory bank

Test image
    → Same preprocessing
    → DINOv3 forward pass
    → Global k-NN on CLS tokens → top-K nearest training images
    → Positional patch k-NN within top-K images
    → Aggregate scores across layers → anomaly map
    → Metrics: I-AUROC, I-AP, I-F1 (image-level); P-AUROC, P-AP, P-F1, AUPRO (pixel-level)
```

### Key Files

| File | Role |
|------|------|
| `RAD/build_bank_multilayer.py` | Builds multi-layer memory bank from normal train images |
| `RAD/build_realiad_bank_multilayer.py` | Same, for Real-IAD dataset |
| `RAD/rad_mvtec_visa_3dadam.py` | Detection + evaluation for MVTec, VisA, 3D-ADAM |
| `RAD/rad_real_iad.py` | Detection + evaluation for Real-IAD |
| `RAD/dataset.py` | Dataset loaders and transforms |
| `RAD/utils.py` | Evaluation metrics (ROC-AUC, AP, F1, AUPRO) and visualization |
| `RAD/dinov3/` | Meta's official DINOv3 ViT implementation (primary encoder) |
| `RAD/models/` | Backup model implementations (ResNet, ViT variants) — not the primary path |

### Key Hyperparameters

- `--layer_idx_list`: Which transformer layers to extract features from (0-indexed; default: 3 6 9 11)
- `--k_image`: Number of nearest training images for local patch bank (MVTec: 150, VisA: 900, Real-IAD: 900, 3D-ADAM: 48)
- `--pos_radius`: Spatial patch neighborhood radius for position-aware matching (0 = disabled)
- `--use_positional_bank`: Enables position-aware patch comparison
- `--max_ratio`: Fraction of top anomalous pixels used to compute image-level score (default: 0.01)

### Supported Datasets

- **MVTec AD**: 15 classes — `../mvtec_anomaly_detection/`
- **VisA**: 12 classes — `../VisA_pytorch/1cls/`
- **Real-IAD**: multi-view industrial — `../Real-IAD/`
- **3D-ADAM**: 24 3D object classes — `../3D-ADAM/`

### Encoder

Primary encoder: DINOv3 ViT-B/16 (86M parameters). Weights downloaded from Meta AI and placed in `../weights/`. Other encoder variants (DINOv1, DINOv2, BEiT) are available in their respective subdirectories but not the default.
