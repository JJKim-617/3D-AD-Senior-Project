#!/bin/bash
# RGB-only inference: --num_views 1 (렌더링 스킵, RGB bank만 사용)
# 뱅크는 이미 빌드되어 있으므로 Phase 2만 실행

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD"
ENCODER_WEIGHT="./dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
BANK_DIR="../bank_views"
ITEMS="bagel cable_gland carrot cookie dowel foam peach potato rope tire"
DEVICE="cuda:1"

echo "=========================================="
echo "[RGB-only] Inference + Evaluation"
echo "  items: $ITEMS"
echo "  num_views: 1 (RGB only)"
echo "=========================================="

.venv/bin/python rad_3d.py \
    --data_path "$DATA_PATH" \
    --bank_dir "$BANK_DIR" \
    --encoder_weight "$ENCODER_WEIGHT" \
    --item_list $ITEMS \
    --k_image 48 --use_positional_bank --pos_radius 1 \
    --num_views 1 \
    --save_name "3d_rgb_only" \
    --device "$DEVICE"

echo ""
echo "[RGB-only] All done."
