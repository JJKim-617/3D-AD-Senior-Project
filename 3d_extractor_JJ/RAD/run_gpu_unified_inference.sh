#!/bin/bash
# Unified bank inference: 원본 RAD 방식 (전체 카테고리 1 bank + CLS top-K)
# RGB only, 1 view

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD"
ENCODER_WEIGHT="./dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
UNIFIED_BANK="/home/ryukimlee/3D-AD-Senior-Project/3d_extractor_JJ/bank_views/unified_bank.pth"
ITEMS="bagel cable_gland carrot cookie dowel foam peach potato rope tire"
DEVICE="cuda:1"

echo "=========================================="
echo "[Unified] Inference with unified bank (pure RAD style)"
echo "=========================================="

.venv/bin/python rad_3d.py \
    --data_path "$DATA_PATH" \
    --bank_dir "dummy" \
    --encoder_weight "$ENCODER_WEIGHT" \
    --item_list $ITEMS \
    --k_image 150 --use_positional_bank --pos_radius 1 \
    --num_views 1 \
    --unified_bank_path "$UNIFIED_BANK" \
    --save_name "3d_unified_pure_rad" \
    --device "$DEVICE"

echo ""
echo "[Unified] All done."
