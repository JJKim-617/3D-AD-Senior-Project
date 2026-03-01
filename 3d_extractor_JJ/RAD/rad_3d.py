"""
rad_3d.py — 3D 포인트 클라우드 다시점 이상탐지 추론 + 평가.

rad_mvtec_visa_3dadam.py 를 28뷰(RGB 1 + 렌더링 27) 파이프라인으로 확장한다.
주요 변경점은 [3D-PATCH] 주석으로 명시되어 있다.

추론 흐름:
    0. 카테고리 시작 시 28개 뱅크 CPU 프리로드 + positional 인덱스 사전 계산
    for each test object:
        1. RGB + 렌더링 27장 → [28, 3, 448, 448]
        2. DINOv3 일괄 forward → patch_list (4×[28,784,768]), cls_list (4×[28,768])
        3. for v in range(28):
               bank_v = preloaded_banks[v] → GPU 전송
               CLS k-NN (K > N_obj 가드 포함)
               4-layer 벡터화 패치 k-NN + weighted fusion → anomaly_map[v] [784]
        4. 28뷰 max fusion → [784] → upsample → [1,1,256,256]
        5. Gaussian smoothing, 메트릭 수집
"""

import multiprocessing
import os
import random
import warnings
import logging
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

from dataset import get_data_transforms
from dataset_3d import MVTec3DRawDataset
from dinov3.hub.backbones import load_dinov3_model
from render_utils import load_and_render, get_viewpoints
# [3D-PATCH] AUPRO 구현을 3D-ADS(MVTec3D-AD 공식 평가 코드 기반)로 교체
# 원본 (RAD utils.py:compute_pro): 등간격 임계값, FPR 최대값 정규화, sklearn.auc 사용
# 변경: 3D-ADS au_pro_util.calculate_au_pro — 분위수 기반 임계값, /0.3 정규화, 커스텀 trapezoid
from au_pro_util import calculate_au_pro
from utils import f1_score_max, get_gaussian_kernel

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

# [3D-PATCH] 뷰 수 28 = RGB 1 + 렌더링 27
# 원본 (RAD): 뷰 개념 없음, 이미지 1장 처리
# 변경: 28뷰 처리 구조
NUM_VIEWS = 28    # 0=RGB, 1..27=렌더링

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------

def setup_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    fmt = logging.Formatter('%(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def tensor_to_np(img_tensor):
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = img_tensor.detach().cpu().unsqueeze(0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return img


def _view_bank_name(v: int) -> str:
    return f"bank_view_{v:02d}_rgb.pth" if v == 0 else f"bank_view_{v:02d}.pth"


def _test_cache_path(cache_root: str, category: str, obj_id: str, v: int) -> str:
    # [3D-PATCH] test 렌더 캐시 경로 헬퍼 — build_bank_3d.py 의 train 캐시 구조와 대칭
    # <cache_root>/<category>/test/<obj_id>/view_<v:02d>.png
    return os.path.join(cache_root, category, "test", obj_id, f"view_{v:02d}.png")


# ---------------------------------------------------------------------------
# 병렬 테스트 렌더링
# ---------------------------------------------------------------------------

def _test_render_worker(task):
    """
    [3D-PATCH] 테스트 오브젝트 병렬 렌더링 워커
    원본: eval_3d() 루프 내 순차 렌더링 (~1h/category)
    변경: multiprocessing.Pool(spawn)으로 오브젝트 단위 병렬 렌더링
    """
    tiff_path, rgb_path, cache_paths, color = task
    if all(os.path.exists(p) for p in cache_paths):
        return
    rendered = load_and_render(tiff_path, color=color, rgb_path=rgb_path)
    for pil, p in zip(rendered, cache_paths):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        pil.save(p)


def _prerender_test_objects(
    raw_dataset: "MVTec3DRawDataset",
    cache_root: str,
    category: str,
    color: str,
    render_workers: int = 8,
):
    """
    [3D-PATCH] 테스트 오브젝트 사전 병렬 렌더링
    원본: eval_3d() 루프 내에서 오브젝트마다 순차 렌더링
    변경: inference 루프 시작 전 모든 테스트 오브젝트를 병렬로 렌더링 → 캐시 저장
          이후 inference 루프는 캐시에서 로드만 하므로 렌더링 시간 제거
    """
    N = len(raw_dataset)
    tasks = []
    for i in range(N):
        obj_id = os.path.splitext(os.path.basename(raw_dataset.rgb_paths[i]))[0]
        cache_paths = [_test_cache_path(cache_root, category, obj_id, v)
                       for v in range(1, NUM_VIEWS)]
        if not all(os.path.exists(p) for p in cache_paths):
            tasks.append((raw_dataset.tiff_paths[i], raw_dataset.rgb_paths[i], cache_paths, color))

    if not tasks:
        print(f"  [PreRender-Test] All {N} objects already cached.")
        return

    print(f"  [PreRender-Test] {len(tasks)}/{N} objects to render  workers={render_workers}")
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=render_workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(_test_render_worker, tasks), 1):
            if done % 10 == 0 or done == len(tasks):
                print(f"  [PreRender-Test] {done}/{len(tasks)} done")


# ---------------------------------------------------------------------------
# 추론 함수
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_3d(
    encoder,
    raw_dataset: MVTec3DRawDataset,
    bank_dir: str,
    device,
    data_transform,
    gt_transform,
    image_size: int,
    crop_size: int,
    color: str,
    k_image: int,
    resize_mask: int,
    max_ratio: float,
    use_positional_bank: bool,
    pos_radius: int,
    layer_weights=None,
    vis_dir=None,
    max_vis: int = 0,
    category: str = "",
    cache_root: str = None,
    render_workers: int = 8,
):
    """
    3D 데이터셋에 대한 패치 k-NN 이상탐지 추론 + 평가.

    [3D-PATCH] 객체 단위 처리 루프 — RAD의 배치 DataLoader 루프와 다름
    원본 (RAD rad_mvtec_visa_3dadam.py:130):
        for img, gt, label, img_path in test_dataloader:
    변경: 객체 1개씩 처리 (28뷰를 배치로 묶어 인코더 forward)

    Returns:
        [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]
    """
    encoder.eval()
    device = torch.device(device)

    gt_list_px, pr_list_px = [], []
    gt_list_sp, pr_list_sp = [], []

    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=1.0).to(device)

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)
    remain_vis = max_vis

    # [3D-PATCH] 28개 뷰 뱅크를 CPU 메모리에 한 번에 프리로드
    # 원본: for v in range(28): torch.load() 를 매 샘플마다 반복 (28 × N회 디스크 I/O)
    # 변경: 카테고리 시작 시 1회만 로드 → CPU RAM에 float16 유지 → GPU 전송 시 float32 변환
    #       디스크 I/O: 28×N회 → 28회로 감소 (cable_gland 기준 3024→28회)
    print(f"  [Preload] Loading {NUM_VIEWS} view banks into CPU memory ...")
    preloaded_banks = []
    for v in range(NUM_VIEWS):
        bank_path = os.path.join(bank_dir, _view_bank_name(v))
        bank_v = torch.load(bank_path, map_location='cpu')
        if v == 0:
            layer_indices = bank_v["layers"]
            num_layers    = len(layer_indices)
        preloaded_banks.append({
            'cls':     bank_v['cls_banks'][-1],    # [N_obj, 768] float16
            'patches': bank_v['patch_banks'],       # 4×[N_obj, 784, 768] float16
        })
        del bank_v
    print(f"  [Preload] Done. {NUM_VIEWS} banks loaded (float16 on CPU).")

    if layer_weights is None or len(layer_weights) != num_layers:
        layer_weights = [1.0 / num_layers] * num_layers
    else:
        s = sum(layer_weights)
        layer_weights = [w / s for w in layer_weights] if s > 0 else [1.0 / num_layers] * num_layers

    N = len(raw_dataset)

    # [3D-PATCH] positional k-NN 이웃 인덱스 사전 계산 — 벡터화에 필요
    # 원본: for j in range(784): ... 루프 안에서 매번 이웃 인덱스 생성
    # 변경: 한 번만 계산하여 GPU 텐서로 보관, 배치 gather에 사용
    _h_pos = _w_pos = crop_size // 16   # 28 for crop_size=448
    _L_pos = _h_pos * _w_pos            # 784
    if use_positional_bank:
        pos_max_neigh = (2 * pos_radius + 1) ** 2   # 9 for radius=1
        pos_padded_idx  = torch.zeros(_L_pos, pos_max_neigh, dtype=torch.long, device=device)
        pos_neigh_mask  = torch.zeros(_L_pos, pos_max_neigh, dtype=torch.bool, device=device)
        for j in range(_L_pos):
            r, c = j // _w_pos, j % _w_pos
            r_min = max(0, r - pos_radius)
            r_max = min(_h_pos - 1, r + pos_radius)
            c_min = max(0, c - pos_radius)
            c_max = min(_w_pos - 1, c + pos_radius)
            idx = [rr * _w_pos + cc for rr in range(r_min, r_max + 1)
                   for cc in range(c_min, c_max + 1)]
            pos_padded_idx[j, :len(idx)] = torch.tensor(idx, dtype=torch.long)
            pos_neigh_mask[j, :len(idx)] = True
        print(f"  [Vectorize] Positional k-NN index precomputed: "
              f"L={_L_pos}, max_neigh={pos_max_neigh}")

    # [3D-PATCH] 테스트 오브젝트 사전 병렬 렌더링 — inference 루프 전 캐시 준비
    # 원본: for 루프 내 순차 렌더링 (~1h/category)
    # 변경: cache_root 지정 시 병렬 사전 렌더링 → inference 루프는 캐시 로드만
    if cache_root is not None:
        _prerender_test_objects(raw_dataset, cache_root, category, color, render_workers)

    for obj_idx in range(N):
        rgb_path  = raw_dataset.rgb_paths[obj_idx]
        tiff_path = raw_dataset.tiff_paths[obj_idx]
        gt_path   = raw_dataset.gt_paths[obj_idx]
        label     = int(raw_dataset.labels[obj_idx])

        # ----------------------------------------------------------------
        # Step A: 28장 이미지 준비 [28, 3, 448, 448]
        # ----------------------------------------------------------------

        # [3D-PATCH] RGB + 렌더링 27장을 배치로 스택
        # 원본 (RAD rad_mvtec_visa_3dadam.py:131): img = img.to(device)  (DataLoader 배치)
        # 변경: RGB 1장 + TIFF 렌더링 27장을 하나의 배치 [28, 3, H, W] 로 구성
        rgb_pil = Image.open(rgb_path).convert('RGB')
        rgb_tensor = data_transform(rgb_pil)  # [3, 448, 448]

        # [3D-PATCH] test 렌더 캐싱 — build_bank_3d.py prerender_train_objects() 와 동일한 패턴
        # 원본: 매 inference마다 27뷰를 새로 렌더링 (CPU 병목, GPU 유휴)
        # 변경: cache_root 지정 시 PNG 캐시 확인 → 있으면 로드, 없으면 렌더 후 저장
        #       두 번째 실행부터 렌더링 건너뜀 (~2시간 → ~5분)
        obj_id = os.path.splitext(os.path.basename(rgb_path))[0]
        if cache_root is not None:
            cache_paths = [_test_cache_path(cache_root, category, obj_id, v)
                           for v in range(1, NUM_VIEWS)]
            if all(os.path.exists(p) for p in cache_paths):
                rendered_pils = [Image.open(p).convert('RGB') for p in cache_paths]
            else:
                rendered_pils = load_and_render(tiff_path, color=color, rgb_path=rgb_path)
                for pil, p in zip(rendered_pils, cache_paths):
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    pil.save(p)
        else:
            rendered_pils = load_and_render(tiff_path, color=color, rgb_path=rgb_path)  # 27 PIL Images
        rendered_tensors = torch.stack([data_transform(p) for p in rendered_pils])  # [27, 3, 448, 448]

        imgs_28 = torch.cat([rgb_tensor.unsqueeze(0), rendered_tensors], dim=0)  # [28, 3, 448, 448]
        imgs_28 = imgs_28.to(device)

        # ----------------------------------------------------------------
        # Step B: DINOv3 일괄 forward (배치 28)
        # ----------------------------------------------------------------

        # [3D-PATCH] 28장을 한 번에 DINOv3 forward
        # 원본 (RAD rad_mvtec_visa_3dadam.py:135-147): batch B 처리
        # 변경: batch=28뷰 (한 객체의 모든 뷰를 한 번에 처리)
        inter = encoder.get_intermediate_layers(
            imgs_28,
            n=layer_indices,
            return_class_token=True,
            reshape=False,
            norm=True,
        )

        patch_list = [pt for pt, _  in inter]   # 4 × [28, 784, 768]
        cls_list   = [ct for  _, ct in inter]   # 4 × [28, 768]

        L = patch_list[0].shape[1]   # 784
        h = w = int(L ** 0.5)        # 28

        # ----------------------------------------------------------------
        # Step C: 뷰별 k-NN + fusion
        # ----------------------------------------------------------------

        # [3D-PATCH] 28개 뷰 순회 — 뷰별 bank 파일 로드/언로드
        # 원본 (RAD rad_mvtec_visa_3dadam.py:161-223): for b in range(B) (배치 내 샘플 순회)
        # 변경: for v in range(28) (28개 뷰 순회), 뷰별 bank 파일을 순차 로드
        anomaly_maps = []

        for v in range(NUM_VIEWS):
            # Step C-1: 프리로드된 뱅크에서 GPU로 전송
            # [3D-PATCH] CPU 프리로드 뱅크 참조 → GPU 전송 + float32 변환
            # 원본: torch.load() 매 샘플마다 디스크 I/O
            # 변경: preloaded_banks[v] 에서 .to(device).float() 만 수행 (CPU→GPU 전송만)
            cls_bank_v   = preloaded_banks[v]['cls'].to(device).float()               # [N_obj, 768]
            patch_bank_v = [pb.to(device).float() for pb in preloaded_banks[v]['patches']]  # 4×[N_obj,784,768]

            N_obj = cls_bank_v.shape[0]

            # Step C-2: CLS 기반 k-NN 검색
            cls_test_v      = cls_list[-1][v]                                  # [768]
            cls_test_v_norm = F.normalize(cls_test_v.unsqueeze(0), dim=-1)    # [1, 768]
            cls_bank_norm   = F.normalize(cls_bank_v, dim=-1)                  # [N_obj, 768]
            sim_img = torch.matmul(cls_test_v_norm, cls_bank_norm.t())         # [1, N_obj]

            # [3D-PATCH] K > N_obj 방어 — 학습 물체 수가 k_image 보다 적을 때 topk 에러 방지
            # 원본 (RAD rad_mvtec_visa_3dadam.py:153): _, topk_idx = torch.topk(sim_img, k_image, ...)
            # 변경: K = min(k_image, N_obj) 로 상한 제한
            K = min(k_image, N_obj)
            _, topk_idx = torch.topk(sim_img, K, dim=-1)   # [1, K]
            neigh_indices = topk_idx[0]                     # [K]

            # Step C-3: 4-layer 패치 k-NN + weighted fusion
            # [3D-PATCH] 4-layer 루프 + weighted fusion — RAD eval 루프 구조 그대로 재사용
            # 원본 (RAD rad_mvtec_visa_3dadam.py:166-221): for li in range(num_layers): ...
            # 변경: q_feat = patch_list[li][v] 로 view 차원 v 인덱싱 추가
            scores_per_layer = []

            for li in range(num_layers):
                # [3D-PATCH] view 인덱스 v 추가 — 28뷰 중 현재 뷰의 patch feature 선택
                # 원본 (RAD rad_mvtec_visa_3dadam.py:170): q_feat = patches_x_l[b]
                # 변경: batch 차원(b) 대신 view 차원(v) 으로 인덱싱
                q_feat = patch_list[li][v]                   # [784, 768]

                # [3D-PATCH] list 원소를 텐서 인덱싱으로 가져옴 — Bug #1 수정
                # 원본 (계획 pseudocode 오류): patch_bank_v[topk_idx] — list에 tensor 인덱스 불가
                # 변경: patch_bank_v[li] 로 레이어 분리 후 bank_l[neigh_indices] 로 인덱싱
                bank_l     = patch_bank_v[li]                # [N_obj, 784, 768]
                neigh_feat = bank_l[neigh_indices]           # [K, 784, 768]

                q_feat     = F.normalize(q_feat,     dim=-1)
                neigh_feat = F.normalize(neigh_feat, dim=-1)

                if use_positional_bank:
                    # [3D-PATCH] 벡터화 positional k-NN — Python 784회 루프 제거
                    # 원본: for j in range(L): 개별 matmul (87,808회/샘플)
                    # 변경: 배치 gather + bmm 으로 한 번에 계산 (~50× 속도 향상)
                    gathered = neigh_feat[:, pos_padded_idx, :]             # [K, L, max_neigh, 768]
                    gathered = gathered.permute(1, 0, 2, 3)                 # [L, K, max_neigh, 768]
                    gathered = gathered.reshape(L, K * pos_max_neigh, 768)  # [L, K*max_neigh, 768]

                    sim = torch.bmm(
                        q_feat.unsqueeze(1),           # [L, 1, 768]
                        gathered.transpose(1, 2),      # [L, 768, K*max_neigh]
                    ).squeeze(1)                       # [L, K*max_neigh]

                    # 에지/코너 패치의 패딩된 이웃 마스킹 (9개 미만)
                    mask_exp = pos_neigh_mask.unsqueeze(1).expand(
                        -1, K, -1
                    ).reshape(L, K * pos_max_neigh)
                    sim[~mask_exp] = -1.0

                    nn_sim = sim.max(dim=-1)[0]        # [L]
                    patch_score_l = 1.0 - nn_sim
                else:
                    bank_local = neigh_feat.reshape(-1, neigh_feat.shape[-1])  # [K*L, 768]
                    sim_p      = torch.matmul(q_feat, bank_local.t())          # [L, K*L]
                    nn_sim, _  = sim_p.max(dim=-1)                             # [L]
                    patch_score_l = 1.0 - nn_sim

                scores_per_layer.append(patch_score_l)

            # multi-layer weighted fusion
            fused = torch.zeros_like(scores_per_layer[0])
            for li in range(num_layers):
                fused = fused + layer_weights[li] * scores_per_layer[li]

            anomaly_maps.append(fused)   # [784]

            # [3D-PATCH] 뷰 v 뱅크 GPU 메모리 해제 — GC에 위임
            # 원본: del + torch.cuda.empty_cache() (28회 CUDA 동기화 강제)
            # 변경: del 만 수행, empty_cache 제거 → CUDA 동기화 오버헤드 제거
            del cls_bank_v, patch_bank_v

        # ----------------------------------------------------------------
        # Step D: 28뷰 max fusion → upsample → GT 정렬
        # ----------------------------------------------------------------

        # [3D-PATCH] 28뷰 max fusion — RAD에는 없는 다시점 fusion 단계
        # 원본 (RAD): 단일 뷰 anomaly_map 그대로 사용
        # 변경: 28개 anomaly_map 중 최대값으로 fusion → 최악의 뷰를 이상 점수로 채택
        final_map = torch.stack(anomaly_maps).max(dim=0).values   # [784]
        patch_map = final_map.view(1, 1, h, w)                    # [1, 1, 28, 28]

        anomaly_map_up = F.interpolate(
            patch_map,
            size=resize_mask,
            mode='bilinear',
            align_corners=False,
        )  # [1, 1, resize_mask, resize_mask]

        # GT 마스크 준비
        if label == 0 or gt_path is None:
            gt_tensor = torch.zeros(1, 1, resize_mask, resize_mask, dtype=torch.bool)
        else:
            gt_pil    = Image.open(gt_path).convert('L')
            gt_tensor = gt_transform(gt_pil)               # [1, H', W'] float
            gt_tensor = F.interpolate(
                gt_tensor.unsqueeze(0),
                size=resize_mask,
                mode='nearest',
            ).squeeze(0)                                    # [1, resize_mask, resize_mask]
            gt_tensor = (gt_tensor > 0.5).bool().unsqueeze(0)  # [1, 1, resize_mask, resize_mask]

        anomaly_map_up = gaussian_kernel(anomaly_map_up)   # Gaussian smoothing

        # ----------------------------------------------------------------
        # Step E: 메트릭 수집
        # ----------------------------------------------------------------

        gt_list_px.append(gt_tensor.cpu())
        pr_list_px.append(anomaly_map_up.cpu())
        gt_list_sp.append(torch.tensor([label]))

        amap_flat = anomaly_map_up.flatten(1)
        if max_ratio == 0:
            sp_score = amap_flat.max(dim=1)[0]
        else:
            k = max(1, int(amap_flat.shape[1] * max_ratio))
            sp_score = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)
        pr_list_sp.append(sp_score.cpu())

        # ----------------------------------------------------------------
        # Step F: 시각화 (선택)
        # ----------------------------------------------------------------

        if vis_dir is not None and remain_vis > 0:
            inp_np   = np.array(Image.open(rgb_path).convert('RGB').resize((resize_mask, resize_mask)))
            amap_np  = anomaly_map_up[0, 0].detach().cpu().numpy()
            gt_np    = gt_tensor[0, 0].cpu().numpy().astype(np.float32)

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(inp_np);       axes[0].set_title("RGB Input"); axes[0].axis('off')
            im = axes[1].imshow(amap_np, cmap='jet')
            axes[1].set_title("Anomaly Map"); axes[1].axis('off')
            fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            axes[2].imshow(gt_np, cmap='gray'); axes[2].set_title("GT Mask"); axes[2].axis('off')

            plt.tight_layout()
            stem = os.path.splitext(os.path.basename(rgb_path))[0]
            plt.savefig(os.path.join(vis_dir, f"vis_{stem}.png"), dpi=150)
            plt.close()
            remain_vis -= 1

        print(f"  [{obj_idx + 1}/{N}] label={label}  sp_score={sp_score.item():.4f}")

    # ----------------------------------------------------------------
    # Step G: 프리로드 뱅크 해제 + AUROC / AUPRO etc.
    # ----------------------------------------------------------------

    del preloaded_banks
    torch.cuda.empty_cache()

    gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].numpy()   # [N, H, W]
    pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].numpy()
    gt_list_sp = torch.cat(gt_list_sp).numpy()
    pr_list_sp = torch.cat(pr_list_sp).numpy()

    # [3D-PATCH] AUPRO: 3D-ADS calculate_au_pro 사용 — MVTec3D-AD 공식 평가 코드와 동일
    # 원본 (RAD utils.compute_pro): 3D numpy 배열 입력, FPR 최대값 정규화
    # 변경: list of 2D arrays 입력, /0.3 정규화, 분위수 기반 임계값 (더 정밀한 PRO 곡선)
    aupro_px, _ = calculate_au_pro(
        list(gt_list_px.astype(np.int32)),   # list of [H,W] int
        list(pr_list_px),                     # list of [H,W] float
    )

    gt_px_flat = gt_list_px.ravel()
    pr_px_flat = pr_list_px.ravel()

    auroc_px = roc_auc_score(gt_px_flat, pr_px_flat)
    auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
    ap_px    = average_precision_score(gt_px_flat, pr_px_flat)
    ap_sp    = average_precision_score(gt_list_sp, pr_list_sp)
    f1_sp    = f1_score_max(gt_list_sp, pr_list_sp)
    f1_px    = f1_score_max(gt_px_flat, pr_px_flat)

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='3D Multi-View Patch-KNN anomaly detection with DINOv3'
    )

    # [3D-PATCH] data_path → 3D-ADAM_anomalib 형식 경로
    # 원본 (RAD rad_mvtec_visa_3dadam.py): MVTec 경로 (train/good/에 PNG 직접 존재)
    # 변경: 3D-ADAM_anomalib 형식 (train/good/rgb/, xyz/ 포함)
    parser.add_argument('--data_path', type=str, required=True,
                        help='3D-ADAM_anomalib 스타일 데이터셋 루트 경로')

    # [3D-PATCH] bank_path → bank_dir (28개 파일을 담는 카테고리별 디렉토리)
    # 원본 (RAD rad_mvtec_visa_3dadam.py): --bank_path (단일 .pth 파일)
    # 변경: --bank_dir (category/bank_view_vv.pth 가 들어있는 디렉토리)
    parser.add_argument('--bank_dir', type=str, required=True,
                        help='build_bank_3d.py 가 출력한 뷰별 .pth 파일 디렉토리 (bank_dir/<category>/)')

    parser.add_argument('--encoder_name',   type=str, default='dinov3_vitb16')
    parser.add_argument('--encoder_weight', type=str,
                        default='./dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    parser.add_argument('--save_dir',  type=str, default='./saved_results_3d')
    parser.add_argument('--save_name', type=str, default='3d_multilayer_36911_448')

    parser.add_argument('--item_list', nargs='+', required=True,
                        help='평가할 카테고리 이름 목록')

    parser.add_argument('--image_size',  type=int,   default=512)
    parser.add_argument('--crop_size',   type=int,   default=448)
    parser.add_argument('--k_image',     type=int,   default=48,
                        help='이미지 레벨 k-NN 이웃 수 (3D-ADAM 기본값 48)')
    parser.add_argument('--resize_mask', type=int,   default=256)
    parser.add_argument('--max_ratio',   type=float, default=0.01)
    parser.add_argument('--vis_max',     type=int,   default=8)

    parser.add_argument('--use_positional_bank', action='store_true')
    parser.add_argument('--pos_radius', type=int, default=1)

    parser.add_argument('--layer_weights', nargs='+', type=float, default=None)

    # [3D-PATCH] test 렌더 캐시 디렉토리 — 지정 시 첫 실행에 PNG 저장, 이후 재사용
    # 원본: 캐싱 없음 (매 inference ~2시간)
    # 변경: --cache_dir 지정 시 두 번째 실행부터 ~5분으로 단축
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='test 렌더 캐시 디렉토리 (지정 시 캐싱 활성화, 미지정 시 캐싱 안 함)')

    # [3D-PATCH] 포인트 클라우드 색상화 방식
    # 원본 (RAD): RGB 이미지 직접 사용
    # 변경: TIFF 렌더링 시 색상화 방식 선택
    parser.add_argument('--color', type=str, default='rgb',
                        choices=['rgb', 'xyz', 'normal', 'gray'])

    # [3D-PATCH] GPU 선택 인수 — 카테고리별 병렬 실행 시 GPU 지정에 사용
    # 원본: cuda:0 하드코딩
    # 변경: --device 로 cuda:0 / cuda:1 / cpu 선택 가능
    parser.add_argument('--device', type=str, default=None,
                        help='사용할 디바이스 (예: cuda:0, cuda:1, cpu). 미지정 시 자동 선택')
    parser.add_argument('--render_workers', type=int, default=8,
                        help='병렬 테스트 렌더링 워커 수 (기본: 8, GPU당 16~28 권장)')

    args = parser.parse_args()

    setup_seed(1)

    os.makedirs(os.path.join(args.save_dir, args.save_name), exist_ok=True)
    logger   = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(21 / 24, 0)
        device = 'cuda:0'
    else:
        device = 'cpu'
    print_fn(f"device: {device}")
    print_fn(f"use_positional_bank={args.use_positional_bank}  pos_radius={args.pos_radius}")
    print_fn(f"color={args.color}  k_image={args.k_image}")
    print_fn(f"item_list={args.item_list}")

    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)

    # DINOv3 인코더 로드 (RAD 와 동일)
    encoder = load_dinov3_model(
        args.encoder_name,
        pretrained_weight_path=args.encoder_weight,
    )
    encoder = encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # 카테고리별 평가
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    for category in args.item_list:
        print_fn(f"\n{'='*60}")
        print_fn(f"[Eval3D] Category: {category}")

        raw_dataset = MVTec3DRawDataset(args.data_path, category, phase='test')
        print_fn(str(raw_dataset))

        if len(raw_dataset) == 0:
            print_fn(f"  Warning: no test objects found for '{category}', skipping.")
            continue

        # [3D-PATCH] 카테고리별 bank_dir 경로 구성
        # 원본 (RAD): args.bank_path (단일 파일)
        # 변경: args.bank_dir/<category>/ 하위의 bank_view_vv.pth 파일들
        bank_dir_cat = os.path.join(args.bank_dir, category)
        if not os.path.isdir(bank_dir_cat):
            print_fn(f"  Error: bank_dir not found: {bank_dir_cat}  (run build_bank_3d.py first)")
            continue

        vis_dir = os.path.join(
            args.save_dir, args.save_name, "test_vis", category
        )

        results = eval_3d(
            encoder           = encoder,
            raw_dataset       = raw_dataset,
            bank_dir          = bank_dir_cat,
            device            = device,
            data_transform    = data_transform,
            gt_transform      = gt_transform,
            image_size        = args.image_size,
            crop_size         = args.crop_size,
            color             = args.color,
            k_image           = args.k_image,
            resize_mask       = args.resize_mask,
            max_ratio         = args.max_ratio,
            use_positional_bank = args.use_positional_bank,
            pos_radius        = args.pos_radius,
            layer_weights     = args.layer_weights,
            vis_dir           = vis_dir,
            max_vis           = args.vis_max,
            category          = category,
            cache_root        = args.cache_dir,
            render_workers    = args.render_workers,
        )

        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        auroc_sp_list.append(auroc_sp);  ap_sp_list.append(ap_sp);   f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px);  ap_px_list.append(ap_px);   f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        print_fn(
            f'{category}: '
            f'I-AUROC:{auroc_sp:.4f}, I-AP:{ap_sp:.4f}, I-F1:{f1_sp:.4f}, '
            f'P-AUROC:{auroc_px:.4f}, P-AP:{ap_px:.4f}, P-F1:{f1_px:.4f}, '
            f'P-AUPRO:{aupro_px:.4f}'
        )

    if auroc_sp_list:
        print_fn(
            '\nMean: '
            'I-AUROC:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
            'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                np.mean(auroc_px_list), np.mean(ap_px_list),
                np.mean(f1_px_list),    np.mean(aupro_px_list),
            )
        )


if __name__ == '__main__':
    main()
