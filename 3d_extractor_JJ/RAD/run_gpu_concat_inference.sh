#!/bin/bash
# RGB + FPFH feature-level concat (3D-ADS style) — Inference only
# concat_bank.pth 필요 (run_build_concat_bank.sh 로 사전 빌드)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD"
ENCODER_WEIGHT="./dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
BANK_DIR="/home/ryukimlee/3D-AD-Senior-Project/3d_extractor_JJ/bank_views"
ITEMS="bagel cable_gland carrot cookie dowel foam peach potato rope tire"
DEVICE="cuda:1"

echo "=========================================="
echo "[Concat] Inference + Evaluation (3D-ADS style)"
echo "  items: $ITEMS"
echo "=========================================="

.venv/bin/python rad_3d.py \
    --data_path "$DATA_PATH" \
    --bank_dir "$BANK_DIR" \
    --encoder_weight "$ENCODER_WEIGHT" \
    --item_list $ITEMS \
    --k_image 48 --use_positional_bank --pos_radius 1 \
    --num_views 1 \
    --use_concat_bank \
    --voxel_size 0.05 \
    --save_name "3d_concat_3dads_style" \
    --device "$DEVICE"

echo ""
echo "[Concat] All done."
