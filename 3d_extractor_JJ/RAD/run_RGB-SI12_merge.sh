#!/bin/bash
# RAD RGB anomaly map + SI-12 curvature anomaly map pixel-level gating fusion
# cwd: ~/3D-AD-Senior-Project/3d_extractor_JJ/

source /home/ryukimlee/3D-AD-Senior-Project/3d_extractor_JJ/RAD/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=1

DATA=/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD
ITEMS="bagel cable_gland carrot cookie dowel foam peach potato rope tire"

python RAD/rad_3d.py \
  --data_path $DATA \
  --bank_dir ./bank_views \
  --encoder_weight /home/ryukimlee/3D-AD-Senior-Project/3d_extractor_JJ/RAD/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --item_list $ITEMS \
  --k_image 48 \
  --num_views 1 \
  --save_dir ./saved_results_3d \
  --save_name rgb_si12_a02 \
  --curv_fusion pixel_gating \
  --curv_alpha 0.2
