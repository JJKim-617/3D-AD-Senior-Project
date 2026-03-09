# RAD-3D Experiment Results

---

## MVTec 3D-AD SOTA

> 3DSR (Cheating Depth, WACV 2024) / G2SF (Geometry-Guided Score Fusion, ICCV 2025)

### I-AUROC

| Category | 3DSR | G2SF | RAD + SI-12 Gating (Exp3) |
|----------|------|------|-------------|
| bagel | <u>98.1</u> | **99.7** | 97.4 |
| cable_gland | 86.7 | <u>92.3</u> | **99.2** |
| carrot | **99.6** | <u>99.3</u> | 98.1 |
| cookie | **98.1** | <u>96.7</u> | 89.6 |
| dowel | **100.0** | 96.6 | <u>97.0</u> |
| foam | **99.4** | <u>99.1</u> | 89.0 |
| peach | 98.6 | **99.4** | <u>99.3</u> |
| potato | <u>97.8</u> | **98.8** | 81.1 |
| rope | **100.0** | 96.6 | <u>99.1</u> |
| tire | **99.5** | 92.2 | <u>92.5</u> |
| **Mean** | **97.8** | <u>97.1</u> | 94.2 |

### AUPRO@30%

| Category | 3DSR | G2SF | RAD + SI-12 Gating (Exp3) |
|----------|------|------|-------------|
| bagel | <u>98.1</u> | **98.2** | 97.7 |
| cable_gland | 97.3 | <u>97.7</u> | **98.0** |
| carrot | **98.2** | **98.2** | <u>98.2</u> |
| cookie | <u>97.1</u> | **97.9** | 94.1 |
| dowel | 96.2 | <u>97.1</u> | **97.7** |
| foam | **97.8** | <u>97.6</u> | 84.9 |
| peach | 98.1 | <u>98.2</u> | **98.3** |
| potato | **98.3** | **98.3** | <u>98.1</u> |
| rope | 97.4 | **97.8** | <u>97.7</u> |
| tire | <u>97.5</u> | **98.1** | 97.4 |
| **Mean** | <u>97.6</u> | **97.9** | 96.2 |

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

## Experiment 3: SI-12 Multi-scale Curvature Prior (Gating Fusion)

### Method

Exp2 (RGB-only RAD)에 **SI-12 multi-scale curvature prior**를 결합. 이전 Exp3 (dampening, 4-dim)에서 curvature descriptor를 **12-dim quadric fitting** 기반으로 업그레이드하고, fusion 방식을 **gating**으로 변경.

#### Pipeline

```
[Bank Build] Train normal TIFF (.tiff organized point cloud)
    → Depth discontinuity filtering (edge_k=5.0)
    → Multi-scale quadric fitting (Weingarten map, k=15,30,60)
    → κ₁, κ₂ → [SI, C, K, H] × 3 scales = 12-dim per point
    → Per-dim mean → 12-dim sample descriptor
    → Domain-level Gaussian: μ∈R^12, Σ∈R^{12×12} (+ ε*I regularization)
    → Train Mahalanobis min/max → SI-12_bank.pth

[Inference] Test TIFF
    → Same 12-dim descriptor extraction
    → Mahalanobis distance from domain Gaussian (12-dim)
    → Min-max normalization → s_curv_norm ∈ [0, 1]
    → Gating fusion: s_final = s_RAD * (1 + λ * s_curv_norm)
        - typical curvature (s_curv_norm ≈ 0) → s_RAD 유지
        - atypical curvature (s_curv_norm > 0) → s_RAD 증폭
```

#### Curvature Descriptor (SI-12) — Extraction Pipeline

```
TIFF [H, W, 3] organized point cloud
│
├─ 1. Depth Discontinuity Filtering
│     인접 픽셀 간 depth 차이 > median_dist × k (k=5.0) → edge 제거
│     → valid_mask [H, W] bool
│
├─ 2. 좌표 변환
│     y *= -1, z *= -1  (카메라 → world 좌표계)
│
├─ 3. Normal Estimation
│     Open3D KDTree KNN (k=30) → per-point surface normal
│
├─ 4. Multi-scale Quadric Fitting (per point, per scale)
│     각 스케일(k=15, 30, 60)에서:
│     ┌─ k-NN 이웃 탐색 (KDTree)
│     ├─ Local tangent frame 구성 (normal → t1, t2 직교 기저)
│     ├─ 이웃 포인트를 tangent frame으로 투영 → (u, v, w)
│     ├─ 2차 곡면 fitting: w = a·u² + b·u·v + c·v²
│     │   (batch least squares: (AᵀA)⁻¹ Aᵀw, ε=1e-8 regularization)
│     ├─ Weingarten map: S = [[2a, b], [b, 2c]]
│     └─ Eigenvalues → κ₁ (max), κ₂ (min) principal curvatures
│
├─ 5. Curvature Features (per point, per scale → 4-dim)
│     SI = (2/π) · arctan2(κ₁+κ₂, κ₁-κ₂)   — Shape Index ∈ [-1, 1]
│     C  = √((κ₁²+κ₂²)/2)                    — Curvedness (magnitude)
│     K  = κ₁ · κ₂                            — Gaussian curvature
│     H  = (κ₁+κ₂) / 2                        — Mean curvature
│
└─ 6. Concat → 12-dim per-point descriptor
      [SI₁₅, C₁₅, K₁₅, H₁₅, SI₃₀, C₃₀, K₃₀, H₃₀, SI₆₀, C₆₀, K₆₀, H₆₀]
```

| Scale | k | Features | 의미 |
|-------|---|----------|------|
| Fine | 15 | SI, C, K, H | 미세 결함 감도 (작은 이웃 → 국소적 곡률 변화에 민감) |
| Medium | 30 | SI, C, K, H | 균형 (법선 추정과 동일 스케일) |
| Coarse | 60 | SI, C, K, H | 노이즈 강건 (넓은 이웃 → 전체적 형상 포착) |

#### Anomaly Score Computation

**Sample-level** (dampening / gating / additive):

```
Test TIFF → 12-dim per-point features → aggregation (mean 또는 p90)
→ 12-dim sample descriptor d_test

Bank (학습 시 저장):
  μ ∈ R^12         (train descriptor mean)
  Σ ∈ R^{12×12}    (train descriptor covariance + ε·I)
  maha_min, maha_max  (train Mahalanobis p1, p99)

s_curv = Mahalanobis(d_test, μ, Σ) = √((d_test - μ)ᵀ Σ⁻¹ (d_test - μ))
s_curv_norm = clip((s_curv - maha_min) / (maha_max - maha_min), 0, 1)
```

**Pixel-level** (pixel_gating):

```
Test TIFF → 12-dim per-point features (N_valid, 12)

Bank (학습 시 저장, per-point level):
  pt_μ ∈ R^12       (ALL train points mean)
  pt_Σ ∈ R^{12×12}  (ALL train points covariance + ε·I)
  pt_min, pt_max     (train point Mahalanobis p1, p99, subsampled to 500K)

Per point i:
  maha_i = √((f_i - pt_μ)ᵀ pt_Σ⁻¹ (f_i - pt_μ))
  score_i = clip((maha_i - pt_min) / (pt_max - pt_min), 0, ∞)

Map back to organized grid → anomaly_map [H, W]
→ bilinear resize to RAD anomaly map 해상도
→ s_final = s_RAD * (1 + α * anomaly_map)
→ min-max re-normalization → Gaussian smoothing
```

#### Memory Bank Types

| Bank | 파일명 | 내용 | 용도 |
|------|--------|------|------|
| Per-point | `SI-12_points_bank.pth` | 학습 포인트별 raw 12-dim features + global Gaussian (μ, Σ, p1/p99) | pixel_gating (per-point anomaly map 생성) |
| Mean-agg (EXP3사용) | `SI-12_mean_bank.pth` | 샘플별 mean-aggregated 12-dim descriptor + Gaussian | sample-level fusion (기본) |
| P90-agg | `SI-12_p90_bank.pth` | 샘플별 p90-aggregated 12-dim descriptor + Gaussian | sample-level fusion (outlier-sensitive) |


#### Fusion Modes

**A. Sample-level fusion** (`--use_curvature_prior`)

테스트 샘플의 12-dim descriptor → domain Gaussian과의 Mahalanobis distance → s_curv_norm ∈ [0,1] → anomaly map 전체에 scalar multiplier 적용.

| Mode | 수식 | 특성 |
|------|------|------|
| `dampening` | `s_RAD * (α + (1-α) * s_curv^γ)` | 정상 curvature → α까지 억제, 이상 → 유지. γ>1이면 conservative |
| `gating` (EXP3 사용) | `s_RAD * (1 + α * s_curv)` | 정상 → RAD 유지, 이상 → 증폭 |
| `additive` | `s_RAD + α * s_curv` | anomaly map에 직접 가산 |

**B. Pixel-level fusion** (`--curv_fusion pixel_gating`)

Per-point bank에서 포인트별 Mahalanobis distance → organized grid [H,W] anomaly map 생성 → RAD anomaly map과 pixel-wise multiplicative gating.

```
s_final = s_RAD * (1 + α * s_curv_map[h,w])
→ min-max re-normalization
→ Gaussian smoothing
```

RAD anomaly map의 공간 해상도를 유지하면서, curvature 이상이 있는 위치만 선택적으로 증폭.


### Settings

| Parameter | Value |
|-----------|-------|
| Dataset | MVTec 3D-AD (10 categories) |
| Encoder | DINOv3 ViT-B/16 (frozen, 86M params) |
| Layers | [3, 6, 9, 11] |
| Image size | 512 → CenterCrop 448 |
| Views | 1 (RGB only) |
| k_image | 48 |
| use_positional_bank | False |
| max_ratio | 0.01 (top 1%) |
| resize_mask | 256 |
| **Curvature descriptor** | **SI-12 (12-dim, quadric fitting)** |
| **Curvature scales** | **k = 15, 30, 60** |
| **curv_fusion** | **gating** |
| **curv_alpha (λ)** | **0.1** |
| Edge filter k | 5.0 |

### Results

| Category | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|----------|---------|------|------|---------|------|------|---------|
| bagel | 0.9737 | 0.9934 | 0.9667 | 0.9950 | 0.7648 | 0.7148 | 0.9765 |
| cable_gland | 0.9918 | 0.9982 | 0.9885 | 0.9969 | 0.6366 | 0.6415 | 0.9801 |
| carrot | 0.9812 | 0.9963 | 0.9699 | 0.9984 | 0.6222 | 0.6235 | 0.9821 |
| cookie | 0.8960 | 0.9726 | 0.9018 | 0.9796 | 0.7306 | 0.6851 | 0.9413 |
| dowel | 0.9704 | 0.9929 | 0.9665 | 0.9971 | 0.6437 | 0.6098 | 0.9770 |
| foam | 0.8900 | 0.9731 | 0.9161 | 0.9473 | 0.4388 | 0.4703 | 0.8485 |
| peach | 0.9931 | 0.9983 | 0.9813 | 0.9988 | 0.8041 | 0.7426 | 0.9826 |
| potato | 0.8108 | 0.9484 | 0.9128 | 0.9980 | 0.6012 | 0.5809 | 0.9812 |
| rope | 0.9905 | 0.9962 | 0.9781 | 0.9955 | 0.5942 | 0.5857 | 0.9765 |
| tire | 0.9246 | 0.9802 | 0.9213 | 0.9966 | 0.5849 | 0.5549 | 0.9738 |
| **Mean** | **0.9422** | **0.9850** | **0.9503** | **0.9903** | **0.6421** | **0.6209** | **0.9620** |
| **Δ Exp2** | **+0.0047** | **+0.0012** | **+0.0044** | **+0.0000** | **+0.0259** | **+0.0114** | **+0.0001** |

### Comparison vs Exp2: I-AUROC

| Category | Exp2 | Exp3 | Δ |
|----------|------|------|---|
| bagel | 0.9814 | 0.9737 | -0.0077 |
| cable_gland | 0.9677 | **0.9918** | **+0.0241** |
| carrot | 0.9506 | **0.9812** | **+0.0306** |
| cookie | **0.9088** | 0.8960 | -0.0128 |
| dowel | 0.9689 | **0.9704** | **+0.0015** |
| foam | 0.8769 | **0.8900** | **+0.0131** |
| peach | 0.9917 | **0.9931** | **+0.0014** |
| potato | 0.8053 | **0.8108** | **+0.0055** |
| rope | 0.9900 | **0.9905** | **+0.0005** |
| tire | **0.9333** | 0.9246 | -0.0087 |
| **Mean** | **0.9375** | **0.9422** | **+0.0047** |

### Comparison vs Exp2: P-AUPRO

| Category | Exp2 | Exp3 | Δ |
|----------|------|------|---|
| bagel | 0.9773 | 0.9765 | -0.0008 |
| cable_gland | 0.9789 | **0.9801** | **+0.0012** |
| carrot | 0.9815 | **0.9821** | **+0.0006** |
| cookie | **0.9445** | 0.9413 | -0.0032 |
| dowel | 0.9764 | **0.9770** | **+0.0006** |
| foam | 0.8471 | **0.8485** | **+0.0014** |
| peach | 0.9827 | 0.9826 | -0.0001 |
| potato | 0.9807 | **0.9812** | **+0.0005** |
| rope | 0.9754 | **0.9765** | **+0.0011** |
| tire | **0.9743** | 0.9738 | -0.0005 |
| **Mean** | **0.9619** | **0.9620** | **+0.0001** |

### Analysis

- **I-AUROC Mean +0.47%p (0.9375→0.9422)**: SI-12 curvature prior가 RAD baseline 대비 image-level 분류 성능 향상.
- **cable_gland +2.4%p, carrot +3.1%p**: curvature 분포가 compact한 카테고리에서 가장 큰 개선. 정상 curvature에서 벗어난 anomaly를 gating이 효과적으로 증폭.
- **foam +1.3%p (0.8769→0.8900)**: 타겟이었던 rough surface 카테고리에서 개선 확인. Multi-scale descriptor가 정상 roughness의 curvature 분포를 정확히 모델링하여 atypical 결함만 증폭.
- **potato +0.6%p (0.8053→0.8108)**: 마찬가지로 rough surface에서 소폭 개선.
- **cookie -1.3%p (0.9088→0.8960)**: cookie는 정상/이상 간 curvature 차이가 크지 않아 gating 효과 제한적. 다만 gating 특성상 baseline score 자체를 훼손하지 않아 하락폭 소폭.
- **bagel -0.8%p, tire -0.9%p**: curvature 분포 overlap이 있는 카테고리에서 미세 하락. λ=0.1의 약한 증폭이 noise로 작용.
- **P-AUPRO Mean +0.01%p (0.9619→0.9620)**: pixel-level 성능 거의 동일. Gating은 sample-level multiplier이므로 anomaly map 형상 보존.
- **P-AP Mean +2.6%p (0.6162→0.6421)**: pixel-level precision에서 유의미한 개선. Curvature prior가 true anomaly의 score를 증폭하여 precision 향상.

### Commit

```

```
---

