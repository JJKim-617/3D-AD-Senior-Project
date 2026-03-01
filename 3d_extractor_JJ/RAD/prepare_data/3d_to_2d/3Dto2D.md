# 3D-to-2D 변환 파이프라인 설계

## 개요

RAD(Is Training Necessary for Anomaly Detection?)의 2D feature 추출 + 메모리 뱅크 파이프라인을 **3D 이상탐지**로 확장한다.

핵심 아이디어는 TIFF 포맷의 organized point cloud를 [CPMF 논문](https://github.com/caoyunkang/CPMF) 방식으로 **27개의 다시점 2D 이미지**로 렌더링하고, 기존 RGB 이미지 1장과 합쳐 총 **28장**을 DINOv3로 feature화한 뒤 RAD의 메모리 뱅크에 저장하는 것이다.

```
3D 입력 (RGB + TIFF/PCD)
    ↓ 렌더링 (CPMF)
27장 다시점 2D 이미지
    ↓
+ 1장 RGB 이미지
= 28장 / 물체
    ↓ DINOv3 feature 추출 (RAD와 동일)
    ↓ 뷰별 파일 분리 저장
bank_view_00_rgb.pth   → RAD 뱅크 구조 동일 [N_obj, 784, 768]
bank_view_01.pth       → 동일
...
bank_view_27.pth       → 동일
    ↓ 추론 시 뷰 v 파일만 로드 → k-NN → fusion
```

---

## 1. 입력 포맷

물체 1개당 다음 두 가지 파일이 묶음으로 들어온다:

| 파일 | 설명 |
|------|------|
| `image.png` (또는 `.jpg`) | RGB 2D 이미지, (H, W, 3) |
| `pointcloud.tiff` (또는 `.pcd`) | Organized point cloud (TIFF: H×W×3 float32 XYZ) 또는 `.pcd` |

### TIFF 포맷 상세

- **Organized point cloud**: 이미지와 동일한 H×W 해상도의 XYZ 값 배열
- 각 픽셀 위치 (r, c)에 해당하는 3D 좌표 (X, Y, Z)가 저장됨
- 유효하지 않은 포인트는 (0, 0, 0) 또는 NaN으로 표시됨

---

## 2. 3D-to-2D 렌더링 파이프라인 (CPMF 방식)

### 2.1 회전 각도 구성

CPMF의 `MultiViewRender`는 x, y, z 세 축에 대해 각각 3개의 회전 각도를 조합한다:

```python
angles_x = [0,    -π/12, +π/12]   # [0°, -15°, +15°]
angles_y = [0,    -π/12, +π/12]
angles_z = [0,    -π/12, +π/12]
```

총 3 × 3 × 3 = **27개 조합** (첫 번째 뷰 (0, 0, 0)이 기준 뷰)

| 뷰 인덱스 | (x, y, z) 회전각 |
|-----------|-----------------|
| 0 | (0°, 0°, 0°) — 기준 뷰 |
| 1 | (-15°, 0°, 0°) |
| 2 | (+15°, 0°, 0°) |
| 3 | (0°, -15°, 0°) |
| ... | ... |
| 26 | (+15°, +15°, +15°) |

### 2.2 렌더링 프로세스

```
TIFF organized point cloud
    ↓ read_pcd()          # 유효 포인트 필터링 + Open3D PointCloud 변환
    ↓ 좌표 변환 행렬 적용  # 카메라 기준 좌표계로 변환
    ↓ rotate_render() × 27  # 각 각도 조합마다 회전 행렬 적용 후 렌더링
27장 이미지 (224×224, BGR)
```

### 2.3 색상화(Coloring) 옵션

CPMF는 다양한 색상화 방식을 지원한다:

| 방식 | 설명 |
|------|------|
| `xyz` | XYZ 좌표값을 RGB로 매핑 |
| `normal` | 표면 법선 벡터 |
| `fpfh` | FPFH 특성 (PCA로 3D 축소) |
| `gray` | 균일 회색 |
| `rgb` | 입력 RGB 데이터 |

> **선택**: xyz 또는 normal 권장 (3D 구조 정보를 가장 잘 표현)

### 2.4 출력

- 뷰당: `224 × 224` BGR 이미지
- 총 27장의 렌더링 이미지

---

## 3. Feature 추출 (RAD와 동일)

### 3.1 이미지 전처리

28장 모두 동일한 RAD 전처리 적용:

```python
transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.CenterCrop(448),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])
```

> 렌더링 이미지(224×224)는 512로 업스케일 후 448 크롭 적용됨

### 3.2 DINOv3 ViT-B/16 Feature 추출

```python
encoder.get_intermediate_layers(
    img,                        # [B, 3, 448, 448]
    n=(3, 6, 9, 11),            # 0-indexed 레이어
    return_class_token=True,
    reshape=False,
    norm=True,
)
```

| 출력 | Shape | 의미 |
|------|-------|------|
| `patch_tokens` | `[B, 784, 768]` | 28×28 패치 × 768차원 |
| `cls_token` | `[B, 768]` | 이미지 전체 전역 표현 |

- 패치 수: 448 ÷ 16 = 28 → 28×28 = **784 패치**
- 임베딩 차원: **768** (ViT-Base)
- 추출 레이어: **[3, 6, 9, 11]** (총 4개 레이어)

---

## 4. 메모리 뱅크 구성 (뷰별 파일 분리)

### 4.1 핵심 전략

**뷰마다 별도 `.pth` 파일로 저장**한다.
각 파일의 내부 구조는 기존 RAD 뱅크와 완전히 동일하며,
추론 시 뷰 v 파일만 로드해 k-NN을 수행한 뒤 언로드한다.

| 뷰 인덱스 | 파일명 | 의미 |
|-----------|--------|------|
| 0 | `bank_view_00_rgb.pth` | RGB 원본 이미지 |
| 1 | `bank_view_01.pth` | 렌더링 뷰 1 (-15°, 0°, 0°) |
| ... | ... | ... |
| 27 | `bank_view_27.pth` | 렌더링 뷰 27 (+15°, +15°, +15°) |

### 4.2 파일별 뱅크 구조 (RAD와 동일)

```python
# bank_view_VV.pth  (VV = 00 ~ 27)
bank = {
    "layers": [3, 6, 9, 11],

    "cls_banks": [                        # 이미지 레벨 전역 특성
        Tensor[N_obj, 768],   # Layer  3
        Tensor[N_obj, 768],   # Layer  6
        Tensor[N_obj, 768],   # Layer  9
        Tensor[N_obj, 768],   # Layer 11
    ],

    "patch_banks": [                      # 패치 레벨 지역 특성
        Tensor[N_obj, 784, 768],  # Layer  3
        Tensor[N_obj, 784, 768],  # Layer  6
        Tensor[N_obj, 784, 768],  # Layer  9
        Tensor[N_obj, 784, 768],  # Layer 11
    ]
}
```

파일당 크기 (N_obj=200 기준): `cls` ~2.4MB + `patch` ~1.92GB → **파일 28개 합계 ~53.8GB** (디스크)

### 4.3 뱅크 빌드 흐름

```
for each 뷰 v (0~27):
    cls_feats_v   = []   # [N_obj, 768]
    patch_feats_v = []   # [N_obj, 784, 768]

    for each 객체 obj (N_obj개):
        image_v[obj] → DINOv3 → cls_tok: [768], patch_tok: [784, 768]
        cls_feats_v.append(cls_tok)
        patch_feats_v.append(patch_tok)

    bank_v = {
        "layers":      [3, 6, 9, 11],
        "cls_banks":   [cat(cls_feats_v)   ...],  # [N_obj, 768]
        "patch_banks": [cat(patch_feats_v) ...],  # [N_obj, 784, 768]
    }
    torch.save(bank_v, f"bank_view_{v:02d}.pth")
```

> 기존 `build_bank_multilayer.py`를 뷰 루프로 감싸는 구조 — 코드 재사용 극대화

### 4.4 추론(Inference) 흐름

View-consistent matching: **뷰 v끼리만 비교, 파일 단위 로드/언로드**

```python
# 테스트 물체 28장 → DINOv3 일괄 forward
imgs_28 = stack([rgb_img] + rendered_imgs)          # [28, 3, 448, 448]
inter = encoder.get_intermediate_layers(
    imgs_28, n=(3,6,9,11), return_class_token=True
)
patch_list = [patch_tok for patch_tok, _ in inter]  # 4 × [28, 784, 768]
cls_list   = [cls_tok   for _, cls_tok   in inter]  # 4 × [28, 768]

anomaly_maps = []

for v in range(28):
    # Step 1: 뷰 v 뱅크 파일 로드
    bank_v       = torch.load(f"bank_view_{v:02d}.pth")
    cls_bank_v   = bank_v["cls_banks"][-1].to(device)              # [N_obj, 768]
    patch_bank_v = [b.to(device) for b in bank_v["patch_banks"]]   # list of 4 × [N_obj, 784, 768]

    # Step 2: CLS 기반 k-NN 객체 검색 (마지막 레이어 CLS만 사용 — RAD와 동일)
    cls_test_v = F.normalize(cls_list[-1][v], dim=-1)               # [768]
    cls_bank_v_norm = F.normalize(cls_bank_v, dim=-1)               # [N_obj, 768]
    sim = cls_test_v @ cls_bank_v_norm.T                            # [N_obj]
    K = min(k_image, cls_bank_v.shape[0])   # ← K > N_obj 방어
    topk_idx = sim.topk(K).indices                                  # [K]

    # Step 3: 4-layer 패치 k-NN + weighted fusion (RAD eval 루프 구조 동일)
    scores_per_layer = []
    for li in range(4):
        bank_l     = patch_bank_v[li]                   # [N_obj, 784, 768]
        neigh_feat = bank_l[topk_idx]                   # [K, 784, 768]
        q_feat     = patch_list[li][v]                  # [784, 768]  ← view v 선택

        q_feat     = F.normalize(q_feat, dim=-1)
        neigh_feat = F.normalize(neigh_feat, dim=-1)
        bank_local = neigh_feat.reshape(-1, 768)        # [K*784, 768]
        sim_p      = q_feat @ bank_local.T              # [784, K*784]
        nn_sim     = sim_p.max(dim=-1)[0]               # [784]
        scores_per_layer.append(1.0 - nn_sim)

    fused = sum(w * s for w, s in zip(layer_weights, scores_per_layer))  # [784]
    anomaly_maps.append(fused)

    del cls_bank_v, patch_bank_v
    torch.cuda.empty_cache()

# Step 4: 28개 뷰 이상 맵 fusion → upsample
final_map = torch.stack(anomaly_maps).max(dim=0).values  # [784]
patch_map = final_map.view(1, 1, 28, 28)
anomaly_map_up = F.interpolate(patch_map, size=256, mode='bilinear')  # [1,1,256,256]
```

### 4.5 메모리 비교

| 항목 | 단일 파일 `[N_obj,28,...]` | **뷰별 파일 분리** |
|------|--------------------------|-------------------|
| VRAM (추론 시) | ~53.7GB (불가) | **~1.92GB (RAD와 동일)** |
| CPU RAM | ~53.7GB | **~1.92GB** |
| 디스크 총량 | ~53.8GB | ~53.8GB (동일) |
| 파일 수 | 1개 | 28개 |
| 구현 복잡도 | 중간 | **낮음** (RAD 코드 재사용) |

---

## 5. RAD와의 차이점 요약

| 항목 | 기존 RAD | 확장 파이프라인 |
|------|---------|----------------|
| 입력 모달리티 | RGB 1장 | RGB 1장 + TIFF/PCD 1개 |
| 이미지 수 / 물체 | 1장 | 28장 (RGB 1 + 렌더링 27) |
| 뱅크 파일 수 | 1개 `.pth` | 28개 `.pth` (뷰별) |
| 뱅크 cls_banks shape | `[N, 768]` | `[N_obj, 768]` × 28파일 |
| 뱅크 patch_banks shape | `[N, 784, 768]` | `[N_obj, 784, 768]` × 28파일 |
| 추론 VRAM | ~1.92GB | **~1.92GB (동일)** |
| 추론 매칭 방식 | 전체 entry 비교 | 뷰 v끼리만 비교 (view-consistent) |
| 추론 코드 | 1장 처리 | 28장 처리 → 뷰별 파일 로드/언로드 → fusion |
| 3D 정보 | 없음 | TIFF 렌더링으로 간접 활용 |

---

## 6. 구현 예정 스크립트

| 파일 | 역할 |
|------|------|
| `render_utils.py` | CPMF `MultiViewRender` 래퍼 (TIFF/PCD → 27장 렌더링) |
| `dataset_3d.py` | RGB + TIFF 쌍을 로드하는 Dataset 클래스 |
| `build_bank_3d.py` | 28장 기반 메모리 뱅크 빌더 (`build_bank_multilayer.py` 확장) |
| `rad_3d.py` | 3D 데이터셋용 추론 + 평가 스크립트 (`rad_mvtec_visa_3dadam.py` 확장) |

---

## 7. 코드 수정 시 주석 규칙

RAD 코드 대비 변경이 생기는 모든 지점 위에 **반드시** 아래 형식의 주석을 명시한다.

```python
# [3D-PATCH] <변경 이유>
# 원본 (RAD <파일명>:<줄번호>): <원본 코드 한 줄>
# 변경: <변경 내용 설명>
```

### 적용 예시

```python
# [3D-PATCH] view 인덱스 v 추가 — 28뷰 중 현재 뷰의 patch feature 선택
# 원본 (RAD rad_mvtec_visa_3dadam.py:170): q_feat = patches_x_l[b]
# 변경: batch 차원(b) 대신 28뷰 중 view 차원(v)으로 인덱싱
q_feat = patch_list[li][v]    # [784, 768]

# [3D-PATCH] K > N_obj 방어 — 학습 물체 수가 k_image보다 적을 때 topk 에러 방지
# 원본 (RAD rad_mvtec_visa_3dadam.py:153): topk_idx = sim_img.topk(k_image, ...)
# 변경: K = min(k_image, N_obj) 로 상한 제한
K = min(k_image, cls_bank_v.shape[0])

# [3D-PATCH] 뷰별 bank 파일 순차 로드 — 전체 뱅크 단일 로드 대신 뷰별 분리 로드
# 원본 (RAD rad_mvtec_visa_3dadam.py:469-472): bank_ckpt = torch.load(args.bank_path)
# 변경: 28개 .pth 파일을 루프 안에서 하나씩 로드/해제 → VRAM ~1.92GB 유지
bank_v = torch.load(f"bank_view_{v:02d}.pth")
```

---

## 8. 확인된 버그 및 수정 방법

로직 점검에서 발견된 문제들. 모두 RAD 기존 코드 재사용으로 해결 가능.

| # | 버그 | 위치 | 심각도 | 수정 |
|---|------|------|--------|------|
| 1 | `patch_bank_v[topk_idx]` — list에 tensor 인덱싱 | `rad_3d.py` Step 3 | **런타임 에러** | `for li: bank_l = patch_bank_v[li]; bank_l[topk_idx]` |
| 2 | 4-layer fusion 루프 누락 | `rad_3d.py` Step 3 | **결과 오류** | RAD `rad_mvtec_visa_3dadam.py:166-221` 블록 그대로 재사용 |
| 3 | `K > N_obj` 시 `.topk()` 에러 | `rad_3d.py` Step 2 | **런타임 에러** | `K = min(k_image, N_obj)` 추가 |
| 4 | 렌더링 224×224 → 512 업스케일 품질 손실 | `render_utils.py` | 품질 문제 | `MultiViewRender(image_size=512)` |

> 버그 1, 2는 위 **4.4 추론 흐름**의 수정된 pseudocode에 이미 반영됨.

---

## 10. 참조

- **RAD**: [Is Training Necessary for Anomaly Detection?](https://github.com/LehongWu/RAD)
- **CPMF 렌더링 유틸**: [caoyunkang/CPMF - render_utils.py](https://github.com/caoyunkang/CPMF/blob/master/utils/render_utils.py)
- **DINOv3**: Meta AI — ViT-B/16, pretrained on LVD-1689M
