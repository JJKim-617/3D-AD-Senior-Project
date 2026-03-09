#!/bin/bash
# Phase 0: Curvature prior bank 빌드
# cwd: ~/3D-AD-Senior-Project/3d_extractor_JJ/
# venv: source RAD/.venv/bin/activate

source /home/ryukimlee/3D-AD-Senior-Project/3d_extractor_JJ/RAD/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

DATA=/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD
ITEMS="bagel cable_gland carrot cookie dowel foam peach potato rope tire"

python RAD/build_curvature_bank.py \
  --data_path $DATA \
  --item_list $ITEMS \
  --bank_dir ./bank_views \
  --num_workers 16
