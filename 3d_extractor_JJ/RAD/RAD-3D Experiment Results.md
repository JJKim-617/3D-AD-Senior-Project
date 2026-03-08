# RAD-3D Experiment Results

---

## MVTec 3D-AD SOTA

> 3DSR (Cheating Depth, WACV 2024) / G2SF (Geometry-Guided Score Fusion, ICCV 2025)

### I-AUROC

| Category | 3DSR | G2SF | RAD-by category (Exp2) |
|----------|------|------|-------------|
| bagel | 98.1 | **99.7** | 98.1 |
| cable_gland | 86.7 | **92.3** | 96.8 |
| carrot | **99.6** | 99.3 | 95.1 |
| cookie | **98.1** | 96.7 | 90.9 |
| dowel | **100.0** | 96.6 | 96.9 |
| foam | **99.4** | 99.1 | 87.7 |
| peach | 98.6 | **99.4** | 99.2 |
| potato | 97.8 | **98.8** | 80.5 |
| rope | **100.0** | 96.6 | 99.0 |
| tire | **99.5** | 92.2 | 93.3 |
| **Mean** | **97.8** | 97.1 | 93.8 |

### AUPRO@30%

| Category | 3DSR | G2SF | RAD-by category (Exp2) |
|----------|------|------|-------------|
| bagel | 98.1 | **98.2** | 97.7 |
| cable_gland | 97.3 | **97.7** | 97.9 |
| carrot | 98.2 | 98.2 | 98.2 |
| cookie | 97.1 | **97.9** | 94.5 |
| dowel | 96.2 | **97.1** | 97.6 |
| foam | **97.8** | 97.6 | 84.7 |
| peach | 98.1 | **98.2** | 98.3 |
| potato | 98.3 | 98.3 | 98.1 |
| rope | 97.4 | **97.8** | 97.5 |
| tire | 97.5 | **98.1** | 97.4 |
| **Mean** | 97.6 | **97.9** | 96.2 |

---

## Experiment 0: Pure RAD (Unified Bank + CLS Top-K)

### Method

원본 RAD와 동일한 방식: **10개 카테고리의 train RGB 이미지를 하나의 통합 메모리 뱅크로 구축**하고, CLS top-K로 테스트 이미지와 가장 유사한 K개 이미지를 선별하여 patch k-NN 수행. 3D 정보 미사용, RGB 1-view only.

### Settings

| Parameter | Value |
|-----------|-------|
| Dataset | MVTec 3D-AD (10 categories) |
| Encoder | DINOv3 ViT-B/16 (frozen, 86M params) |
| Layers | [3, 6, 9, 11] |
| Image size | 512 → CenterCrop 448 |
| Views | 1 (RGB only) |
| **Memory Bank** | **Unified (all 10 categories, N=2656)** |
| **k_image** | **150 (CLS top-K)** |
| pos_radius | 1 (3×3 neighborhood) |
| max_ratio | 0.01 (top 1%) |
| resize_mask | 256 |

### Results

| Category | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|----------|---------|------|------|---------|------|------|---------|
| bagel | 0.9835 | 0.9961 | 0.9721 | 0.9952 | 0.7719 | 0.7169 | 0.9772 |
| cable_gland | 0.9672 | 0.9925 | 0.9609 | 0.9962 | 0.6193 | 0.6263 | 0.9788 |
| carrot | 0.9697 | 0.9939 | 0.9594 | 0.9982 | 0.5968 | 0.6067 | 0.9818 |
| cookie | 0.8963 | 0.9724 | 0.9067 | 0.9813 | 0.7375 | 0.6922 | 0.9444 |
| dowel | 0.9682 | 0.9923 | 0.9573 | 0.9968 | 0.6283 | 0.6059 | 0.9758 |
| foam | 0.8731 | 0.9686 | 0.9045 | 0.9498 | 0.4407 | 0.4736 | 0.8548 |
| peach | 0.9938 | 0.9985 | 0.9813 | 0.9989 | 0.8040 | 0.7375 | 0.9829 |
| potato | 0.8142 | 0.9488 | 0.9167 | 0.9979 | 0.5850 | 0.5612 | 0.9812 |
| rope | 0.9887 | 0.9955 | 0.9853 | 0.9953 | 0.5775 | 0.5811 | 0.9763 |
| tire | 0.9320 | 0.9821 | 0.9274 | 0.9970 | 0.5780 | 0.5482 | 0.9756 |
| **Mean** | **0.9387** | **0.9841** | **0.9471** | **0.9907** | **0.6339** | **0.6149** | **0.9629** |

### Comparison vs Exp2 (Per-Category Bank)

| Category | Exp0 (unified, K=150) | Exp2 (per-category, K=48) | Delta |
|----------|-----------------------|----------------------|-------|
| bagel | 0.9835 | 0.9814 | +0.0021 |
| cable_gland | 0.9672 | 0.9677 | -0.0005 |
| carrot | 0.9697 | 0.9506 | +0.0191 |
| cookie | 0.8963 | 0.9088 | -0.0125 |
| dowel | 0.9682 | 0.9689 | -0.0007 |
| foam | 0.8731 | 0.8769 | -0.0038 |
| peach | 0.9938 | 0.9917 | +0.0021 |
| potato | 0.8142 | 0.8053 | +0.0089 |
| rope | 0.9887 | 0.9900 | -0.0013 |
| tire | 0.9320 | 0.9333 | -0.0013 |
| **Mean** | **0.9387** | **0.9375** | **+0.0012** |

### Commit

```

```
---

## Experiment 1: 3D-Baseline

### Method

RAD(training-free anomaly detection)를 3D 포인트 클라우드 데이터에 확장한 파이프라인.

**핵심 아이디어**: 3D 포인트 클라우드를 CPMF 방식으로 다시점 2D 이미지로 렌더링한 뒤, 기존 RAD의 patch-level k-NN 파이프라인을 뷰별로 적용하고 결과를 fusion.

#### Pipeline

```
[Train] 3D point cloud (.tiff)
    → CPMF 27-view rendering (512×512) + original RGB (1장)
    → Resize 512 → CenterCrop 448
    → DINOv3 ViT-B/16 (frozen) intermediate features [layer 3, 6, 9, 11]
    → 뷰별 cls_bank + patch_bank 저장 (28개 .pth 파일/카테고리)

[Test] 3D point cloud (.tiff)
    → 동일 렌더링 + 전처리
    → DINOv3 forward (28장 일괄)
    → 뷰별 CLS k-NN → top-K 이웃 검색
    → 뷰별 positional patch k-NN (4-layer weighted fusion)
    → 28뷰 anomaly map max fusion
    → Upsample → Gaussian smoothing → 메트릭 산출
```

#### View Rendering (CPMF)

- 27개 고정 시점에서 Open3D OffscreenRenderer로 렌더링
- 좌표 변환: `[[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]]`
- 회전: `o3d.geometry.get_rotation_matrix_from_xyz([rx, ry, rz])`
- 렌더링 해상도: 512×512

#### Scoring

- **Pixel-level**: 28뷰 anomaly map의 max fusion → bilinear upsample (256×256) → Gaussian smoothing
- **Image-level**: anomaly map 상위 1% 픽셀 평균 (`max_ratio=0.01`)
- **AUPRO**: FPR 0~30% 구간 적분 / 0.3 정규화 (MVTec 3D-AD 공식 평가 코드 기반)

### Settings

| Parameter | Value |
|-----------|-------|
| Dataset | MVTec 3D-AD (10 categories) |
| Encoder | DINOv3 ViT-B/16 (frozen, 86M params) |
| Layers | [3, 6, 9, 11] |
| Image size | 512 → CenterCrop 448 |
| Views | 28 (RGB 1 + CPMF rendered 27) |
| View fusion | max |
| k_image | 48 |
| pos_radius | 1 (3×3 neighborhood) |
| max_ratio | 0.01 (top 1%) |
| resize_mask | 256 |
| color | rgb |
| AUPRO integration limit | 0.3 (30%) |

### Results

| Category | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|----------|---------|------|------|---------|------|------|---------|
| bagel | 0.9835 | 0.9963 | 0.9708 | 0.9932 | 0.5857 | 0.5994 | 0.9726 |
| cable_gland | 0.8248 | 0.9516 | 0.9040 | 0.9930 | 0.5239 | 0.5096 | 0.9725 |
| carrot | 0.8187 | 0.9493 | 0.9329 | 0.9966 | 0.3742 | 0.4720 | 0.9797 |
| cookie | 0.7236 | 0.8891 | 0.8909 | 0.9743 | 0.4553 | 0.5439 | 0.9260 |
| dowel | 0.7463 | 0.9060 | 0.8927 | 0.9925 | 0.3431 | 0.4391 | 0.9592 |
| foam | 0.6634 | 0.8942 | 0.8889 | 0.9484 | 0.3079 | 0.3279 | 0.8403 |
| peach | 0.9053 | 0.9734 | 0.9401 | 0.9980 | 0.7442 | 0.6676 | 0.9814 |
| potato | 0.7782 | 0.9337 | 0.8976 | 0.9968 | 0.4357 | 0.4882 | 0.9787 |
| rope | 0.8102 | 0.9003 | 0.8679 | 0.9907 | 0.3824 | 0.4240 | 0.9671 |
| tire | 0.8152 | 0.9452 | 0.8854 | 0.9907 | 0.4967 | 0.5158 | 0.9499 |
| **Mean** | **0.8069** | **0.9339** | **0.9071** | **0.9874** | **0.4649** | **0.4988** | **0.9527** |

### Commit

```
31ef4961dc846c21062532b73acfd77c7639f20a
```
---

## Experiment 2: RGB-Only (No 3D Rendering) = Vanilla RAD & by category memory bank

### Method

Baseline과 동일한 RAD 파이프라인에서 **CPMF 27-view 렌더링을 제거**하고, 원본 RGB 이미지 1장만으로 anomaly detection 수행.

3D 포인트 클라우드 정보를 전혀 사용하지 않는 pure 2D baseline으로, 3D 렌더링 뷰가 실제로 기여하는 성능 향상 폭을 측정하기 위한 ablation.

### Settings

| Parameter | Value |
|-----------|-------|
| Dataset | MVTec 3D-AD (10 categories) |
| Encoder | DINOv3 ViT-B/16 (frozen, 86M params) |
| Layers | [3, 6, 9, 11] |
| Image size | 512 → CenterCrop 448 |
| **Views** | **1 (RGB only, no rendering)** |
| View fusion | N/A (single view) |
| k_image | 48 |
| pos_radius | 1 (3×3 neighborhood) |
| max_ratio | 0.01 (top 1%) |
| resize_mask | 256 |
| color | rgb |
| AUPRO integration limit | 0.3 (30%) |

### Results

| Category | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|----------|---------|------|------|---------|------|------|---------|
| bagel | 0.9814 | 0.9956 | 0.9721 | 0.9950 | 0.7682 | 0.7155 | 0.9773 |
| cable_gland | 0.9677 | 0.9926 | 0.9609 | 0.9962 | 0.6333 | 0.6212 | 0.9789 |
| carrot | 0.9506 | 0.9894 | 0.9559 | 0.9979 | 0.5029 | 0.5810 | 0.9815 |
| cookie | 0.9088 | 0.9760 | 0.9109 | 0.9813 | 0.7388 | 0.6917 | 0.9445 |
| dowel | 0.9689 | 0.9925 | 0.9573 | 0.9969 | 0.6283 | 0.6022 | 0.9764 |
| foam | 0.8769 | 0.9696 | 0.9057 | 0.9471 | 0.4417 | 0.4753 | 0.8471 |
| peach | 0.9917 | 0.9980 | 0.9767 | 0.9988 | 0.8026 | 0.7368 | 0.9827 |
| potato | 0.8053 | 0.9456 | 0.9053 | 0.9977 | 0.4950 | 0.5481 | 0.9807 |
| rope | 0.9900 | 0.9960 | 0.9853 | 0.9951 | 0.5743 | 0.5750 | 0.9754 |
| tire | 0.9333 | 0.9828 | 0.9294 | 0.9967 | 0.5772 | 0.5481 | 0.9743 |
| **Mean** | **0.9375** | **0.9838** | **0.9459** | **0.9903** | **0.6162** | **0.6095** | **0.9619** |

### Comparison vs Baseline (Exp1)

| Metric | Baseline (28 views) | RGB-Only (1 view) | Delta |
|--------|---------------------|-------------------|-------|
| I-AUROC | 0.8069 | **0.9375** | **+0.1306** |
| I-AP | 0.9339 | **0.9838** | **+0.0499** |
| I-F1 | 0.9071 | **0.9459** | **+0.0388** |
| P-AUROC | 0.9874 | **0.9903** | **+0.0029** |
| P-AP | 0.4649 | **0.6162** | **+0.1513** |
| P-F1 | 0.4988 | **0.6095** | **+0.1107** |
| P-AUPRO | 0.9527 | **0.9619** | **+0.0092** |

### Commit

```

```
---

