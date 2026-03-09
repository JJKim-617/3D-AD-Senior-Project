# Experiment Design: SI-12 Multi-scale Curvature Prior for RAD

## 1. Problem
RAD는 CLS embedding 기반 cosine similarity로 appearance deviation만 감지한다.
foam, cookie, potato 등 **정상 표면이 rough한 도메인**에서 false positive가 발생한다.

```
normal rough surface → CLS feature deviation → anomaly score 증가 → false positive
```

RAD의 암묵적 가정: `geometry variance ≈ anomaly`
실제: `natural roughness ≠ defect`

## 2. Core Idea
RAD score에 **SI-12 multi-scale curvature prior**를 추가하여 anomaly score를 두 신호의 결합으로 계산한다.

```
appearance anomaly (RAD) + geometry prior (SI-12 curvature descriptor)
```

## 3. Preprocessing: Depth Discontinuity Filtering

TIFF는 organized point cloud [H,W,3] 구조. 물체 경계(edge) 포인트는 kNN 이웃이
한쪽으로만 분포 → 비정상 curvature 추정 → descriptor 오염.

**Adaptive threshold 기반 경계 제거:**
```python
depth = np.linalg.norm(pts, axis=-1)            # [H, W]
dx = np.abs(depth[:, 1:] - depth[:, :-1])       # 수평 차이
dy = np.abs(depth[1:, :] - depth[:-1, :])       # 수직 차이
median_dist = np.median(dx[valid_pairs])         # adaptive 기준
τ = median_dist * k                              # k = 5~10
edge_mask = (dx > τ) | (dy > τ)
# edge_mask 주변 point 제거 후 curvature 계산
```

- 고정 τ 대신 adaptive threshold → 카테고리/스케일에 무관하게 동작
- curvature 계산 전에 전처리로 적용 → 경계 노이즈가 SI/C에 전파되지 않음

## 4. Curvature Feature Extraction (Multi-scale Quadric Fitting)

기존 Open3D covariance eigenvalue ratio 방식에서 **local quadric fitting (Weingarten map)** 방식으로 변경.
(참고: `3d_extractor_ej/geo_features_bench/curvature_features.py`)

**주곡률 추정:**
```python
# 각 포인트의 k-NN을 local tangent frame (t1, t2, n) 으로 투영
# 2차 곡면 z = au² + buv + cv² fitting (batch least squares)
# Weingarten map S = [[2a, b], [b, 2c]] 의 eigenvalue → κ₁, κ₂
```

**Multi-scale:** k = [15, 30, 60] 에서 각각 독립적으로 κ₁, κ₂ 추정.

**각 스케일에서 4개 특징:**
| Feature | 수식 | 의미 |
|---------|------|------|
| Shape Index (SI) | `(2/π) * arctan2(κ₁+κ₂, κ₁-κ₂)` | curvature type [-1,1] |
| Curvedness (C) | `√((κ₁²+κ₂²)/2)` | curvature magnitude |
| Gaussian Curv (K) | `κ₁·κ₂` | intrinsic curvature |
| Mean Curv (H) | `(κ₁+κ₂)/2` | extrinsic curvature |

- k=15: fine scale (고감도, 미세 결함)
- k=30: medium scale (균형)
- k=60: coarse scale (노이즈 강건)

## 5. Sample-level Curvature Descriptor (12-dim)

**Per-point:** [SI, C, K, H] × 3 scales = 12-dim
**Per-sample:** per-dim mean → 12-dim descriptor

```python
v_sample = [
    mean(SI_k15), mean(C_k15), mean(K_k15), mean(H_k15),   # fine
    mean(SI_k30), mean(C_k30), mean(K_k30), mean(H_k30),   # medium
    mean(SI_k60), mean(C_k60), mean(K_k60), mean(H_k60),   # coarse
]
```

이전 4-dim 대비 장점:
- multi-scale → 스케일별 curvature 특성 포착 (fine detail ~ global shape)
- Gaussian/Mean curvature 추가 → intrinsic/extrinsic 구분
- quadric fitting → covariance eigenvalue ratio보다 정확한 κ 추정

## 6. Domain-level Curvature Distribution

Train normal samples에서 모든 12-dim v_sample을 모아 domain-level Gaussian 모델링:

```
P_domain(v_sample) ~ N(μ_domain ∈ R^12, Σ_domain ∈ R^{12×12})
```

**Covariance regularization:** 12-dim에서 train sample 수 대비 차원이 커질 수 있으므로 regularization 필수.
```python
Σ_reg = Σ + ε * I   # ε = 1e-6, 항상 적용
```

**Bank 파일:** `{bank_dir}/{category}/SI-12_bank.pth`
```python
{
    "mu":          Tensor[12],          # domain mean
    "sigma":       Tensor[12, 12],      # regularized covariance
    "curv_min":    float,               # train Mahalanobis min
    "curv_max":    float,               # train Mahalanobis max
    "descriptors": Tensor[N_train, 12], # 개별 descriptor (디버깅용)
    "category":    str,
}
```

## 7. Test Sample Curvature Score

```
s_curv = (v_test − μ_domain)^T * Σ_domain⁻¹ * (v_test − μ_domain)  (Mahalanobis, 12-dim)
```

## 8. Score Normalization & Fusion

### Scale 문제
| Score | 범위 | 일반적 값 |
|-------|------|----------|
| s_RAD | [0, 1] (cosine distance) | 0.0 ~ 0.4 |
| s_curv | [0, +∞) (Mahalanobis) | 0 ~ 수백 |

→ 직접 합산 시 s_curv가 s_RAD를 압도. **Normalization 필수.**

### Normalization 방식: Train set min-max
curvature bank build 시 train normal samples의 Mahalanobis distance를 미리 계산하여 min/max를 저장한다.

```python
# build_curvature_bank.py에서 저장
train_dists = [mahalanobis(v, μ, Σ) for v in train_descriptors]
curv_min, curv_max = min(train_dists), max(train_dists)

# SI-12_bank.pth에 포함
save: {mu, sigma, curv_min, curv_max, descriptors}
```

```python
# rad_3d.py에서 test 시 정규화
s_curv_raw = mahalanobis(v_test, μ, Σ)
s_curv = (s_curv_raw - curv_min) / (curv_max - curv_min + ε)
s_curv = clamp(s_curv, 0, 1)  # outlier 방지
```

### Fusion: Dampening Multiplicative (default)

s_curv는 sample-level, s_RAD는 pixel-level이므로 fusion 방식 선택이 중요하다.

**목표:** typical curvature(정상 roughness) → FP 억제, atypical curvature → score 유지

**Additive의 문제:**
```
pixel_i_final = pixel_i_RAD + λ * s_curv
```
→ 모든 pixel에 동일한 offset → normal pixel도 score 상승 → localization 왜곡

**단순 Multiplicative의 문제:**
```
s_final = s_RAD * (1 + λ * s_curv_norm)
```
→ typical curvature(s_curv_norm ≈ 0) → multiplier ≈ 1 → s_RAD 그대로 → FP 감소 안됨
→ atypical curvature만 증폭할 뿐, 정상 roughness의 FP를 억제하지 못함

**Dampening Multiplicative with Gamma (채택):**
```
w_curv = (s_curv_norm) ^ γ
s_final = s_RAD * (α + (1 − α) * w_curv)
```
- γ > 1: w_curv 곡선이 아래로 눌림 → 낮은 s_curv를 더 강하게 억제 (conservative)
- γ < 1: w_curv 곡선이 위로 올라감 → 약간의 curvature 차이에도 score 유지 (sensitive)
- γ = 1: 기존 linear dampening과 동일
- α ∈ [0, 1]: 최소 보존 비율 (default 0.3)

예시 (α=0.3, γ=2.0):
| 케이스 | s_RAD | s_curv_norm | w_curv | s_final | 효과 |
|--------|-------|-------------|--------|---------|------|
| foam normal (typical) | 0.35 | 0.05 | 0.0025 | 0.35 * 0.302 = **0.106** | FP 강하게 억제 |
| foam borderline | 0.40 | 0.50 | 0.25 | 0.40 * 0.475 = **0.190** | 중간 억제 |
| foam anomaly (atypical) | 0.60 | 0.90 | 0.81 | 0.60 * 0.867 = **0.520** | 유지 |
| smooth normal | 0.05 | 0.02 | 0.0004 | 0.05 * 0.300 = **0.015** | 변화 미미 |

γ=2.0이면 s_curv_norm=0.5 → w_curv=0.25로, 중간 영역의 score도 더 억제한다.

**Default:** `--curv_fusion dampening` `--curv_alpha 0.3` `--curv_gamma 1.0`

## 9. Implementation Plan

### 신규 파일
| 파일 | 역할 |
|------|------|
| `curvature_utils.py` | TIFF → multi-scale quadric fitting (k=15,30,60) → 12-dim descriptor |
| `build_curvature_bank.py` | Train normal → domain μ∈R^12, Σ∈R^{12×12} → `SI-12_bank.pth` |
| `eval_si12_only.py` | SI-12 descriptor만으로 anomaly detection (RGB 미사용) |

### 수정 파일
| 파일 | 변경 내용 |
|------|-----------|
| `rad_3d.py` | `SI-12_bank.pth` 로드 + 12-dim Mahalanobis score + fusion |

### 실행 스크립트
| 파일 | 역할 |
|------|------|
| `run_curv_bank.sh` | SI-12 bank 빌드 (16 workers) |
| `run_curv_inference.sh` | RAD + SI-12 gating fusion 추론 |
| `run_si12_only.sh` | SI-12 curvature only 평가 (RGB 미사용) |

### 추가 CLI args (rad_3d.py)
- `--use_curvature_prior` (flag)
- `--curv_alpha` (float, default=0.3, dampening: 최소 보존 비율 / gating: 증폭 계수 λ)
- `--curv_gamma` (float, default=1.0, dampening 전용: w_curv 곡선 제어)
- `--curv_fusion` (dampening | gating | additive, default=dampening)

## 10. Pipeline

```
[Phase 0] Curvature Bank Build (1회)
  train normal TIFF → edge filtering → multi-scale quadric fitting (k=15,30,60) → [SI,C,K,H]×3 → 12-dim descriptor → μ,Σ → min-max norm stats → .pth

[Phase 1] 기존 RAD Bank Build (변경 없음)

[Phase 2] Inference (rad_3d.py 수정)
  test TIFF → edge filtering → multi-scale quadric → 12-dim v_test
  s_curv_raw = Mahalanobis(v_test, μ, Σ)
  s_curv_norm = min-max normalize + clamp [0,1]
  w_curv = s_curv_norm ^ γ
  s_final = s_RAD * (α + (1-α) * w_curv)
```

## 11. Experiments

**Descriptor:** SI-12 (multi-scale quadric fitting, k=15/30/60, 12-dim per sample)

| # | 실험 | 설명 |
|---|------|------|
| 0 | SI-12 Only | SI-12 Mahalanobis만으로 분류 (RGB 미사용) |
| 1 | Baseline (Exp2) | RAD RGB only (per-category bank) |
| 2 | Proposed (dampening) | RAD * (α + (1-α) * s_curv^γ) |
| 3 | Proposed (gating) | RAD * (1 + λ * s_curv_norm) |
| 4 | Proposed (additive) | RAD + λ * s_curv_norm |
| 5 | Ablation: λ sweep | λ = 0.1, 0.2, 0.3, 0.5 |
| 6 | Ablation: descriptor | SI-12 (12-dim) vs 이전 4-dim |
| 7 | Ablation: distribution | Gaussian / GMM / KDE |

## 12. Evaluation

**Metrics:**
- SI-12 Only: I-AUROC, I-AP, I-F1 (sample-level만 가능)
- RAD + SI-12: I-AUROC, I-AP, I-F1, P-AUROC, P-AP, P-F1, P-AUPRO (전체)

**특히 확인:**
- foam, cookie, potato에서 false positive 감소
- 다른 카테고리 성능 유지
- SI-12 Only에서 geometry-sensitive 도메인(cable_gland 등)의 분류 능력

## 13. Execution Commands

```bash
# cwd: ~/3D-AD-Senior-Project/3d_extractor_JJ/
# venv: source RAD/.venv/bin/activate

DATA=/home/ryukimlee/3D-AD-Senior-Project/Datasets/MVTec3D-AD
ITEMS="bagel cable_gland carrot cookie dowel foam peach potato rope tire"

# Phase 0: SI-12 bank 빌드 (1회)
python RAD/build_curvature_bank.py \
  --data_path $DATA \
  --item_list $ITEMS \
  --bank_dir ./bank_views \
  --num_workers 16

# SI-12 Only 평가 (RGB 미사용)
python RAD/eval_si12_only.py \
  --data_path $DATA \
  --item_list $ITEMS \
  --bank_dir ./bank_views \
  --num_workers 16 \
  --save_dir ./saved_results_3d \
  --save_name si12_only

# RAD + SI-12 Gating 추론
python RAD/rad_3d.py \
  --data_path $DATA \
  --bank_dir ./bank_views \
  --encoder_weight ./RAD/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --item_list $ITEMS \
  --k_image 48 \
  --num_views 1 \
  --save_dir ./saved_results_3d \
  --save_name curv_gating_si12 \
  --use_curvature_prior \
  --curv_alpha 0.1 \
  --curv_fusion gating
```
