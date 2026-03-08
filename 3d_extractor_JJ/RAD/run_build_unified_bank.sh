#!/bin/bash
# 전체 카테고리 RGB 이미지를 하나의 통합 메모리 뱅크로 빌드 (원본 RAD 방식)
# GPU 사용 (DINOv3 forward pass)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD"
ENCODER_WEIGHT="./dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
BANK_PATH="/home/ryukimlee/3D-AD-Senior-Project/3d_extractor_JJ/bank_views/unified_bank.pth"
DEVICE="cuda:1"

echo "=========================================="
echo "[Unified Bank] Build single bank for all MVTec 3D-AD categories"
echo "=========================================="

.venv/bin/python build_bank_unified.py \
    --data_path "$DATA_PATH" \
    --encoder_weight "$ENCODER_WEIGHT" \
    --bank_path "$BANK_PATH" \
    --item_list bagel cable_gland carrot cookie dowel foam peach potato rope tire \
    --image_size 512 --crop_size 448 \
    --batch_size 32 --num_workers 16 \
    --layer_idx_list 3 6 9 11 \
    --device "$DEVICE"

echo ""
echo "[Unified Bank] Done."
