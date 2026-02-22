# BTF 논문이 사용하는 Feature Extraction 방식

논문: **"Back to the Feature: Classical 3D Features are (Almost) All You Need for 3D Anomaly Detection"**
코드 위치: `external/3D-ADS/`

---

## 1. 데이터 입력 구조

`data/mvtec3d.py`에서 DataLoader가 반환하는 `sample`의 구성:

```
sample = (
    sample[0],  # RGB 이미지        → 224x224, ImageNet 정규화된 Tensor
    sample[1],  # Organized PC      → .tiff에서 읽은 3D 포인트클라우드 (H x W x 3)
    sample[2],  # Depth map         → Organized PC의 Z축 값 (깊이), 3채널 복제
)
```

**Organized PC란?**
각 픽셀 위치에 (X, Y, Z) 3D 좌표가 대응되는 구조화된 포인트클라우드.
`utils/mvtec3d_util.py`에서 tiff 파일로부터 읽고 224x224로 resize.

```python
# utils/mvtec3d_util.py
organized_pc_to_depth_map(organized_pc)  # Z축만 추출 → depth map
resize_organized_pc(organized_pc)        # 224x224로 nearest interpolation
```

---

## 2. 공통 PatchCore 파이프라인

모든 method가 `feature_extractors/features.py`의 `Features` 클래스를 상속하며 동일한 흐름을 따름.

### Train 단계 (`fit`)
```
train 이미지 → feature 추출 → AvgPool3x3 → AdaptiveAvgPool resize
             → patch 벡터로 reshape → patch_lib(메모리 뱅크)에 저장
             → Coreset subsampling (전체의 10%)
```

### Test 단계 (`evaluate`)
```
test 이미지 → feature 추출 → patch 벡터 변환
           → patch_lib과 KNN 거리 계산
           → 거리가 멀수록 anomaly score 높음
           → pixel-wise anomaly map 생성 → bilinear upsample → Gaussian blur
```

### Anomaly Score 계산 (`compute_s_s_map`, `features.py:59`)
```python
dist = torch.cdist(patch, self.patch_lib)   # test patch ↔ 메모리 뱅크 전체 거리
min_val, min_idx = torch.min(dist, dim=1)   # 각 patch의 최근접 거리
s_star = torch.max(min_val)                 # 이미지 레벨 anomaly score

# Reweighting (논문 Eq.7): KNN 기반 가중치로 보정
w = 1 - (exp(s_star/D) / sum(exp(knn_dists/D)))
s = w * s_star
```

---

## 3. 각 Method별 Feature Extraction

### 3-1. RGB iNet (`RGB_inet_features.py`)
- **입력**: `sample[0]` — RGB 이미지
- **방식**: WideResNet50 backbone (timm, pretrained on ImageNet)의 중간 레이어 feature map 사용
- **추출 레이어**: layer2, layer3 출력 (out_indices=(2,3))
- **처리**:
  ```
  RGB → WideResNet50 → [layer2 feature map, layer3 feature map]
      → AvgPool3x3 → AdaptiveAvgPool(largest_fmap_size)
      → concat → reshape to (N_patches, C)
  ```
- **특징**: 색상/텍스처 이상 탐지에 강함

---

### 3-2. Depth iNet (`depth_inet_features.py`)
- **입력**: `sample[2]` — Depth map (3채널)
- **방식**: RGB iNet과 동일한 WideResNet50 backbone 사용
- **처리**: RGB iNet과 동일, 입력만 depth map으로 교체
- **특징**: 깊이 정보 기반 이상 탐지, 기하학적 변화에 반응

---

### 3-3. Raw (`raw_features.py`)
- **입력**: `sample[2]` — Depth map의 단일 채널 (`sample[2][0,0,:,:]`)
- **방식**: 아무런 feature 변환 없이 raw pixel 값 그대로 사용
- **처리**:
  ```
  depth map → 28x28 grid로 reshape → patch 벡터
  ```
- **특징**: 가장 단순한 baseline, 성능 낮음

---

### 3-4. HoG (`hog_features.py`)
- **입력**: `sample[2]` — Depth map 단일 채널
- **방식**: scikit-image의 `hog()` 함수 사용
- **파라미터**:
  ```python
  hog(depth, orientations=8, pixels_per_cell=(8,8), cells_per_block=(1,1))
  ```
- **처리**:
  ```
  depth map → HoG descriptor → reshape → (N_patches, 8) 벡터
  ```
- **특징**: 엣지/기울기 방향 정보 캡처, 형상 변화 감지

---

### 3-5. SIFT (`sift_features.py`)
- **입력**: `sample[2]` — Depth map 단일 채널
- **방식**: Kornia 기반 Dense SIFT Descriptor (`utils/DenseSIFTDescriptor.py`)
- **처리**:
  ```
  depth map → Dense SIFT → AvgPool3x3 → AdaptiveAvgPool(28x28)
            → reshape to (N_patches, C)
  ```
- **특징**: 스케일 불변 특징, 회전 불변성 일부 보유

---

### 3-6. FPFH (`fpfh_features.py`)
- **입력**: `sample[1]` — Organized Point Cloud
- **방식**: Open3D의 `compute_fpfh_feature()` 사용
- **처리**:
  ```
  Organized PC → Unorganized PC (영벡터 제거)
              → Open3D PointCloud 변환
              → 법선벡터 추정 (radius = voxel_size*2, max_nn=30)
              → FPFH 계산 (radius = voxel_size*5, max_nn=100)
              → 33차원 벡터 → organized 형태로 복원
              → Tensor 변환 → AvgPool → reshape
  ```
- **핵심 파라미터**: `voxel_size=0.05`
- **특징**: 3D 형상의 회전 불변 descriptor, 논문에서 가장 강력한 3D-only 방법

---

### 3-7. BTF = RGB + FPFH (`rgb_fpfh_features.py`) ⭐ 저자 제안 방법
- **입력**: `sample[0]` (RGB) + `sample[1]` (Point Cloud)
- **방식**: RGB iNet feature와 FPFH feature를 채널 방향으로 concat
- **처리**:
  ```
  RGB  → WideResNet50 → resize → rgb_patch  (N_patches, C_rgb)
  PC   → FPFH         → resize → fpfh_patch (N_patches, 33)
                                    ↓
                       concat_patch (N_patches, C_rgb+33)
                                    ↓
                       PatchCore KNN 이상 탐지
  ```
- **특징**: 색상/텍스처(RGB) + 3D 형상(FPFH) 동시 커버 → 최고 성능

---

## 4. Backbone 모델 상세 (`features.py:159`)

```python
timm.create_model(
    model_name='wide_resnet50_2',
    pretrained=True,           # ImageNet pretrained
    features_only=True,        # 중간 feature map 출력
    out_indices=(2, 3),        # layer2, layer3 출력
)
```

- **weight 캐시 위치**: `~/.cache/torch/hub/checkpoints/wide_resnet50_racm-8234f177.pth`
- **출력 feature map 크기**:
  - layer2: (B, 512, 28, 28)
  - layer3: (B, 1024, 14, 14)
- **모든 파라미터 freeze**: `freeze_parameters(layers=[], freeze_bn=True)`

---

## 5. 성능 요약 (MVTec 3D-AD, AU PRO 기준)

| Method     | 입력 | AU PRO (raw) |
|:-----------|:-----|-------------:|
| RGB iNet   | RGB  | 0.876 |
| Depth iNet | Depth| 0.586 |
| Raw        | Depth| 0.191 |
| HoG        | Depth| 0.614 |
| SIFT       | Depth| 0.866 |
| FPFH       | PC   | 0.928 |
| **BTF (RGB+FPFH)** | RGB+PC | **0.964** |
