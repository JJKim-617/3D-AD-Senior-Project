#!/bin/bash
# RGB + FPFH feature-level concat bank 빌드 (3D-ADS 방식)
# FPFH bank이 이미 빌드된 상태에서 실행
# CPU only, GPU 불필요

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BANK_DIR="/home/ryukimlee/3D-AD-Senior-Project/3d_extractor_JJ/bank_views"
ITEMS="bagel cable_gland carrot cookie dowel foam peach potato rope tire"

echo "=========================================="
echo "[Concat-Bank] Building RGB+FPFH concat banks (3D-ADS style)"
echo "  items: $ITEMS"
echo "=========================================="

.venv/bin/python build_bank_concat.py \
    --bank_dir "$BANK_DIR" \
    --item_list $ITEMS

echo ""
echo "[Concat-Bank] All done."
