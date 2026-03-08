#!/bin/bash
# FPFH bank 빌드 (CPU only, GPU 불필요)
# 다른 터미널에서 독립적으로 실행 가능

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_PATH="/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD"
BANK_DIR="/home/ryukimlee/3D-AD-Senior-Project/3d_extractor_JJ/bank_views"
ITEMS="bagel cable_gland carrot cookie dowel foam peach potato rope tire"

echo "=========================================="
echo "[FPFH-Bank] Building FPFH banks (CPU)"
echo "  items: $ITEMS"
echo "=========================================="

.venv/bin/python build_fpfh_bank.py \
    --data_path "$DATA_PATH" \
    --item_list $ITEMS \
    --bank_dir "$BANK_DIR" \
    --voxel_size 0.05

echo ""
echo "[FPFH-Bank] All done."
