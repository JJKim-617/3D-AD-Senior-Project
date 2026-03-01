#!/bin/bash
# GPU 0: cable_gland carrot cookie dowel foam
# Phase 1 완료 후 자동으로 Phase 2 실행

set -e  # 에러 발생 시 즉시 중단

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD"
ENCODER_WEIGHT="./dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
BANK_DIR="../bank_views"
CACHE_DIR="../bank_views/render_cache"
ITEMS="cable_gland carrot cookie dowel foam"
DEVICE="cuda:0"

export EGL_PLATFORM=surfaceless
export DISPLAY=""

echo "=========================================="
echo "[GPU0] Phase 1: Building memory banks"
echo "  items: $ITEMS"
echo "=========================================="

.venv/bin/python build_bank_3d.py \
    --data_path "$DATA_PATH" \
    --item_list $ITEMS \
    --encoder_weight "$ENCODER_WEIGHT" \
    --bank_dir "$BANK_DIR" \
    --image_size 512 --crop_size 448 --batch_size 512 --num_workers 4 \
    --layer_idx_list 3 6 9 11 \
    --render_workers 28 \
    --device "$DEVICE"

echo ""
echo "=========================================="
echo "[GPU0] Phase 2: Inference + Evaluation"
echo "  items: $ITEMS"
echo "=========================================="

.venv/bin/python rad_3d.py \
    --data_path "$DATA_PATH" \
    --bank_dir "$BANK_DIR" \
    --encoder_weight "$ENCODER_WEIGHT" \
    --item_list $ITEMS \
    --k_image 48 --use_positional_bank --pos_radius 1 \
    --cache_dir "$CACHE_DIR" \
    --save_name "3d_multilayer_36911_448_gpu0" \
    --render_workers 28 \
    --device "$DEVICE"

echo ""
echo "[GPU0] All done."
