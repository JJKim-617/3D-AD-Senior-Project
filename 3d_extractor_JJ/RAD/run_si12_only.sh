#!/bin/bash
# SI-12 curvature descriptor만으로 anomaly detection 평가 (RGB 미사용)
# cwd: ~/3D-AD-Senior-Project/3d_extractor_JJ/

source /home/ryukimlee/3D-AD-Senior-Project/3d_extractor_JJ/RAD/.venv/bin/activate

DATA=/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD
ITEMS="bagel cable_gland carrot cookie dowel foam peach potato rope tire"

python RAD/eval_si12_only.py \
  --data_path $DATA \
  --item_list $ITEMS \
  --bank_dir ./bank_views \
  --num_workers 16 \
  --save_dir ./saved_results_3d \
  --save_name si12_only
