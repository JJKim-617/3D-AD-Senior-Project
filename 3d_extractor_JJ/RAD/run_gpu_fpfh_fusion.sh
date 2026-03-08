#!/bin/bash
# RGB + FPFH score-level fusion (1 view) — Inference only
# FPFH bank은 이미 빌드 완료, raw score fusion

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD"
ENCODER_WEIGHT="./dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
BANK_DIR="/home/ryukimlee/3D-AD-Senior-Project/3d_extractor_JJ/bank_views"
ITEMS="bagel cable_gland carrot cookie dowel foam peach potato rope tire"
DEVICE="cuda:1"
FPFH_ALPHA=0.1

echo "=========================================="
echo "[RGB+FPFH] Inference + Evaluation (raw score fusion)"
echo "  items: $ITEMS"
echo "  fpfh_alpha: $FPFH_ALPHA"
echo "=========================================="

.venv/bin/python rad_3d.py \
    --data_path "$DATA_PATH" \
    --bank_dir "$BANK_DIR" \
    --encoder_weight "$ENCODER_WEIGHT" \
    --item_list $ITEMS \
    --k_image 48 --use_positional_bank --pos_radius 1 \
    --num_views 1 \
    --fpfh_alpha "$FPFH_ALPHA" \
    --voxel_size 0.05 \
    --save_name "3d_rgb_fpfh_a${FPFH_ALPHA}_raw" \
    --device "$DEVICE"

echo ""
echo "[RGB+FPFH] All done."
