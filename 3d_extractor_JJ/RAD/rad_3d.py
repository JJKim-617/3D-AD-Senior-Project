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
from build_fpfh_bank import compute_fpfh_from_tiff

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
    num_views: int = NUM_VIEWS,
    fpfh_bank_path: str = None,
    fpfh_alpha: float = 0.5,
    voxel_size: float = 0.05,
    use_concat_bank: bool = False,
    skip_cls_topk: bool = False,
    unified_bank: dict = None,
    curv_prior: dict = None,
    curv_alpha: float = 0.3,
    curv_gamma: float = 1.0,
    curv_fusion: str = "dampening",
    curv_percentile: float = 0,
    curv_points_bank: dict = None,
    curv_workers: int = 2,
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
    from collections import defaultdict
    vis_count_per_type = defaultdict(int)  # defect_type → 저장 카운트

    # [3D-PATCH] 뱅크 프리로드
    if unified_bank is not None:
        # Unified bank: 전체 카테고리가 하나의 bank에 통합 (원본 RAD 방식)
        layer_indices = unified_bank["layers"]
        num_layers    = len(layer_indices)
        preloaded_banks = [{
            'cls':     unified_bank['cls_banks'][-1],   # [N_total, 768]
            'patches': unified_bank['patch_banks'],      # 4×[N_total, 784, 768]
        }]
        num_views = 1  # unified bank은 RGB 1뷰만
        print(f"  [Unified] Using unified bank: N={unified_bank['cls_banks'][-1].shape[0]}")
    else:
        # Per-category bank: 뷰별 bank 파일 로드
        print(f"  [Preload] Loading {num_views} view banks into CPU memory ...")
        preloaded_banks = []
        for v in range(num_views):
            if use_concat_bank:
                bank_path = os.path.join(bank_dir, "concat_bank.pth")
            else:
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
        print(f"  [Preload] Done. {num_views} banks loaded (float16 on CPU).")

    # FPFH bank 프리로드
    fpfh_bank = None
    if fpfh_bank_path is not None and os.path.exists(fpfh_bank_path):
        fpfh_data = torch.load(fpfh_bank_path, map_location='cpu')
        fpfh_bank = fpfh_data['fpfh_patches']  # [N_obj, 784, 33] float16
        print(f"  [FPFH] Bank loaded: {tuple(fpfh_bank.shape)}  alpha={fpfh_alpha}")
    elif fpfh_bank_path is not None:
        print(f"  [FPFH] Warning: bank not found: {fpfh_bank_path}, FPFH disabled")

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
    if cache_root is not None and num_views > 1:
        _prerender_test_objects(raw_dataset, cache_root, category, color, render_workers)

    # [3D-PATCH] Curvature precompute (병렬)
    curv_descs = {}
    curv_amaps = {}  # pixel_gating: tiff_path → [H, W] anomaly map
    if curv_prior is not None or curv_points_bank is not None:
        from functools import partial as _partial
        import multiprocessing as _mp

        _tiff_paths = [raw_dataset.tiff_paths[i] for i in range(N)]
        _num_curv_workers = min(curv_workers, N)

        if curv_fusion == "pixel_gating" and curv_points_bank is not None:
            # Pixel-level curvature anomaly map precompute
            from curvature_utils import compute_curvature_anomaly_map_from_tiff

            # Gaussian stats: use precomputed if available, else compute on-the-fly
            if "pt_mu" in curv_points_bank:
                _pt_mu = curv_points_bank["pt_mu"].numpy()
                _pt_sigma_inv = np.linalg.inv(curv_points_bank["pt_sigma"].numpy())
                _pt_min = curv_points_bank["pt_min"]
                _pt_max = curv_points_bank["pt_max"]
            else:
                # On-the-fly: compute from point_features
                _all_pts = np.concatenate(
                    [pf.numpy() for pf in curv_points_bank["point_features"] if len(pf) > 0],
                    axis=0,
                )
                _pt_mu = np.mean(_all_pts, axis=0).astype(np.float64)
                _reg = 1e-6 * np.eye(_all_pts.shape[1])
                _pt_sigma = np.cov(_all_pts, rowvar=False).astype(np.float64) + _reg
                _pt_sigma_inv = np.linalg.inv(_pt_sigma)
                # Normalization stats (p1/p99)
                if len(_all_pts) > 500_000:
                    _rng = np.random.default_rng(42)
                    _sub = _all_pts[_rng.choice(len(_all_pts), 500_000, replace=False)]
                else:
                    _sub = _all_pts
                _diff = _sub.astype(np.float64) - _pt_mu
                _maha = np.sqrt(np.clip(np.sum(_diff @ _pt_sigma_inv * _diff, axis=1), 0, None))
                _pt_min = float(np.percentile(_maha, 1))
                _pt_max = float(np.percentile(_maha, 99))
                print(f"  [Curvature] On-the-fly Gaussian: {len(_all_pts)} pts, "
                      f"Maha p1-p99: [{_pt_min:.4f}, {_pt_max:.4f}]")

            _curv_map_fn = _partial(
                compute_curvature_anomaly_map_from_tiff,
                pt_mu=_pt_mu, pt_sigma_inv=_pt_sigma_inv,
                pt_min=_pt_min, pt_max=_pt_max,
            )

            if _num_curv_workers > 1:
                print(f"  [Curvature] Precomputing {N} pixel anomaly maps (workers={_num_curv_workers}) ...")
                _ctx = _mp.get_context("spawn")
                with _ctx.Pool(_num_curv_workers) as _pool:
                    _results = list(_pool.map(_curv_map_fn, _tiff_paths))
                for _i, _amap in enumerate(_results):
                    curv_amaps[_tiff_paths[_i]] = _amap
            else:
                print(f"  [Curvature] Precomputing {N} pixel anomaly maps (sequential) ...")
                for _i in range(N):
                    curv_amaps[_tiff_paths[_i]] = _curv_map_fn(_tiff_paths[_i])
            print(f"  [Curvature] Pixel map precompute done.")

        elif curv_prior is not None:
            # Sample-level descriptor precompute
            from curvature_utils import compute_curvature_descriptor_from_tiff

            _curv_fn = _partial(compute_curvature_descriptor_from_tiff, percentile=curv_percentile)

            if _num_curv_workers > 1:
                print(f"  [Curvature] Precomputing {N} descriptors (workers={_num_curv_workers}, pctl={curv_percentile}) ...")
                _ctx = _mp.get_context("spawn")
                with _ctx.Pool(_num_curv_workers) as _pool:
                    _results = list(_pool.map(_curv_fn, _tiff_paths))
                for _i, _desc in enumerate(_results):
                    curv_descs[_tiff_paths[_i]] = _desc
            else:
                print(f"  [Curvature] Precomputing {N} descriptors (sequential, pctl={curv_percentile}) ...")
                for _i in range(N):
                    curv_descs[_tiff_paths[_i]] = _curv_fn(_tiff_paths[_i])
            print(f"  [Curvature] Precompute done.")

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

        if num_views == 1:
            # RGB only — 렌더링 스킵
            imgs_28 = rgb_tensor.unsqueeze(0).to(device)  # [1, 3, 448, 448]
        else:
            # [3D-PATCH] test 렌더 캐싱 — build_bank_3d.py prerender_train_objects() 와 동일한 패턴
            # 원본: 매 inference마다 27뷰를 새로 렌더링 (CPU 병목, GPU 유휴)
            # 변경: cache_root 지정 시 PNG 캐시 확인 → 있으면 로드, 없으면 렌더 후 저장
            #       두 번째 실행부터 렌더링 건너뜀 (~2시간 → ~5분)
            obj_id = os.path.splitext(os.path.basename(rgb_path))[0]
            if cache_root is not None:
                cache_paths = [_test_cache_path(cache_root, category, obj_id, v)
                               for v in range(1, num_views)]
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

        # [Concat] 테스트 FPFH 추출 → DINOv3 patch와 concat용
        concat_fpfh_test = None
        if use_concat_bank:
            _fpfh_pool = torch.nn.AdaptiveAvgPool2d((h, w))
            _fpfh_raw = compute_fpfh_from_tiff(tiff_path, voxel_size)  # [1, 33, H, W]
            _fpfh_down = _fpfh_pool(_fpfh_raw).squeeze(0).permute(1, 2, 0)  # [28, 28, 33]
            concat_fpfh_test = _fpfh_down.reshape(L, 33).to(device)  # [784, 33]

        # ----------------------------------------------------------------
        # Step C: 뷰별 k-NN + fusion
        # ----------------------------------------------------------------

        # [3D-PATCH] 28개 뷰 순회 — 뷰별 bank 파일 로드/언로드
        # 원본 (RAD rad_mvtec_visa_3dadam.py:161-223): for b in range(B) (배치 내 샘플 순회)
        # 변경: for v in range(28) (28개 뷰 순회), 뷰별 bank 파일을 순차 로드
        anomaly_maps = []

        for v in range(num_views):
            # Step C-1: 뱅크 로드
            is_unified = (unified_bank is not None)

            if is_unified:
                # Unified bank: CLS만 GPU로 (작음), patch는 CPU에 유지
                cls_bank_v = preloaded_banks[v]['cls'].to(device).float()  # [N_total, 768]
                patch_bank_cpu = preloaded_banks[v]['patches']              # 4×[N_total, 784, 768] CPU
                patch_bank_v = None  # GPU에 올리지 않음
            else:
                # Per-category bank: 전체 GPU 전송
                cls_bank_v   = preloaded_banks[v]['cls'].to(device).float()               # [N_obj, 768]
                patch_bank_v = [pb.to(device).float() for pb in preloaded_banks[v]['patches']]  # 4×[N_obj,784,768]
                patch_bank_cpu = None

            N_obj = cls_bank_v.shape[0]

            # Step C-2: CLS 기반 k-NN 검색
            if skip_cls_topk:
                # CLS top-K 스킵: 전체 train 샘플을 이웃으로 사용
                K = N_obj
                neigh_indices = torch.arange(N_obj, device=device)
            else:
                cls_test_v      = cls_list[-1][v]                                  # [768]
                cls_test_v_norm = F.normalize(cls_test_v.unsqueeze(0), dim=-1)    # [1, 768]
                cls_bank_norm   = F.normalize(cls_bank_v, dim=-1)                  # [N_obj, 768]
                sim_img = torch.matmul(cls_test_v_norm, cls_bank_norm.t())         # [1, N_obj]

                # [3D-PATCH] K > N_obj 방어 — 학습 물체 수가 k_image 보다 적을 때 topk 에러 방지
                K = min(k_image, N_obj)
                _, topk_idx = torch.topk(sim_img, K, dim=-1)   # [1, K]
                neigh_indices = topk_idx[0]                     # [K]

            # Step C-3: 4-layer 패치 k-NN + weighted fusion
            scores_per_layer = []

            for li in range(num_layers):
                q_feat = patch_list[li][v]                   # [784, 768]

                # [Concat] DINOv3 patch + FPFH concat → [784, 801]
                if concat_fpfh_test is not None:
                    q_feat = torch.cat([q_feat, concat_fpfh_test], dim=-1)

                # Patch bank 로드: unified면 CPU에서 top-K만 선별 후 GPU 전송
                if is_unified:
                    # CPU에서 인덱싱 → 선별된 K개만 GPU 전송 (VRAM 절약)
                    neigh_idx_cpu = neigh_indices.cpu()
                    neigh_feat = patch_bank_cpu[li][neigh_idx_cpu].to(device).float()  # [K, 784, 768]
                else:
                    bank_l = patch_bank_v[li]                # [N_obj, 784, 768]
                    if skip_cls_topk:
                        neigh_feat = bank_l                  # 복사 없이 직접 참조
                    else:
                        neigh_feat = bank_l[neigh_indices]   # [K, 784, 768]

                q_feat     = F.normalize(q_feat,     dim=-1)
                neigh_feat = F.normalize(neigh_feat, dim=-1)

                if use_positional_bank:
                    # [3D-PATCH] 벡터화 positional k-NN — Python 784회 루프 제거
                    # K가 클 때 (skip_cls_topk) OOM 방지를 위해 chunk 단위 처리
                    D = neigh_feat.shape[-1]                                # 768 or 801 (concat)
                    CHUNK = 48  # 한 번에 처리할 이웃 수
                    if K <= CHUNK:
                        # 기존 방식: 한 번에 전체 처리
                        gathered = neigh_feat[:, pos_padded_idx, :]             # [K, L, max_neigh, D]
                        gathered = gathered.permute(1, 0, 2, 3)                 # [L, K, max_neigh, D]
                        gathered = gathered.reshape(L, K * pos_max_neigh, D)    # [L, K*max_neigh, D]

                        sim = torch.bmm(
                            q_feat.unsqueeze(1),           # [L, 1, D]
                            gathered.transpose(1, 2),      # [L, D, K*max_neigh]
                        ).squeeze(1)                       # [L, K*max_neigh]

                        mask_exp = pos_neigh_mask.unsqueeze(1).expand(
                            -1, K, -1
                        ).reshape(L, K * pos_max_neigh)
                        sim[~mask_exp] = -1.0

                        nn_sim = sim.max(dim=-1)[0]        # [L]
                    else:
                        # Chunk 처리: VRAM 절약
                        nn_sim = torch.full((L,), -1.0, device=device)
                        q_unsq = q_feat.unsqueeze(1)       # [L, 1, D]
                        for c_start in range(0, K, CHUNK):
                            c_end = min(c_start + CHUNK, K)
                            c_k = c_end - c_start
                            chunk_feat = neigh_feat[c_start:c_end]              # [c_k, L, D]
                            gathered = chunk_feat[:, pos_padded_idx, :]         # [c_k, L, max_neigh, D]
                            gathered = gathered.permute(1, 0, 2, 3)             # [L, c_k, max_neigh, D]
                            gathered = gathered.reshape(L, c_k * pos_max_neigh, D)

                            sim_chunk = torch.bmm(
                                q_unsq,                        # [L, 1, D]
                                gathered.transpose(1, 2),      # [L, D, c_k*max_neigh]
                            ).squeeze(1)                       # [L, c_k*max_neigh]

                            mask_chunk = pos_neigh_mask.unsqueeze(1).expand(
                                -1, c_k, -1
                            ).reshape(L, c_k * pos_max_neigh)
                            sim_chunk[~mask_chunk] = -1.0

                            chunk_max = sim_chunk.max(dim=-1)[0]    # [L]
                            nn_sim = torch.max(nn_sim, chunk_max)
                            del gathered, sim_chunk, chunk_feat

                    patch_score_l = 1.0 - nn_sim
                else:
                    # non-positional도 chunk 처리
                    D = neigh_feat.shape[-1]
                    CHUNK = 48
                    if K <= CHUNK:
                        bank_local = neigh_feat.reshape(-1, D)          # [K*L, D]
                        sim_p      = torch.matmul(q_feat, bank_local.t())  # [L, K*L]
                        nn_sim, _  = sim_p.max(dim=-1)                     # [L]
                    else:
                        nn_sim = torch.full((L,), -1.0, device=device)
                        for c_start in range(0, K, CHUNK):
                            c_end = min(c_start + CHUNK, K)
                            chunk_local = neigh_feat[c_start:c_end].reshape(-1, D)  # [c_k*L, D]
                            sim_chunk = torch.matmul(q_feat, chunk_local.t())       # [L, c_k*L]
                            chunk_max, _ = sim_chunk.max(dim=-1)
                            nn_sim = torch.max(nn_sim, chunk_max)
                            del chunk_local, sim_chunk
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
            del cls_bank_v
            if patch_bank_v is not None:
                del patch_bank_v

        # ----------------------------------------------------------------
        # Step D: 28뷰 max fusion → upsample → GT 정렬
        # ----------------------------------------------------------------

        # [3D-PATCH] 28뷰 max fusion — RAD에는 없는 다시점 fusion 단계
        # 원본 (RAD): 단일 뷰 anomaly_map 그대로 사용
        # 변경: 28개 anomaly_map 중 최대값으로 fusion → 최악의 뷰를 이상 점수로 채택
        rgb_final_map = torch.stack(anomaly_maps).max(dim=0).values   # [784]

        # ----------------------------------------------------------------
        # Step D-2: FPFH score fusion (optional)
        # ----------------------------------------------------------------
        if fpfh_bank is not None:
            # FPFH 추출: TIFF → [1, 33, H, W] → pool → [784, 33]
            fpfh_pool = torch.nn.AdaptiveAvgPool2d((h, w))
            fpfh_map_raw = compute_fpfh_from_tiff(tiff_path, voxel_size)  # [1, 33, H, W]
            fpfh_down = fpfh_pool(fpfh_map_raw).squeeze(0).permute(1, 2, 0)  # [28, 28, 33]
            fpfh_test = fpfh_down.reshape(L, 33).to(device)  # [784, 33]

            # FPFH bank → GPU (view 0의 top-K 이웃 인덱스 재사용)
            fpfh_bank_gpu = fpfh_bank.to(device).float()  # [N_obj, 784, 33]
            fpfh_neigh = fpfh_bank_gpu[neigh_indices]  # [K, 784, 33]

            # Cosine similarity k-NN
            q_fpfh = F.normalize(fpfh_test, dim=-1)  # [784, 33]
            n_fpfh = F.normalize(fpfh_neigh, dim=-1)  # [K, 784, 33]

            if use_positional_bank:
                gathered_f = n_fpfh[:, pos_padded_idx, :]  # [K, L, max_neigh, 33]
                gathered_f = gathered_f.permute(1, 0, 2, 3).reshape(L, K * pos_max_neigh, 33)
                sim_f = torch.bmm(
                    q_fpfh.unsqueeze(1),
                    gathered_f.transpose(1, 2),
                ).squeeze(1)  # [L, K*max_neigh]
                mask_f = pos_neigh_mask.unsqueeze(1).expand(-1, K, -1).reshape(L, K * pos_max_neigh)
                sim_f[~mask_f] = -1.0
                fpfh_score = 1.0 - sim_f.max(dim=-1)[0]  # [784]
            else:
                fpfh_local = n_fpfh.reshape(-1, 33)  # [K*784, 33]
                sim_f = torch.matmul(q_fpfh, fpfh_local.t())  # [784, K*784]
                fpfh_score = 1.0 - sim_f.max(dim=-1)[0]  # [784]

            del fpfh_bank_gpu, fpfh_neigh

            # Raw score fusion (both are cosine distance, same [0,1] scale)
            final_map = (1.0 - fpfh_alpha) * rgb_final_map + fpfh_alpha * fpfh_score
        else:
            final_map = rgb_final_map

        # ----------------------------------------------------------------
        # Step D-3: Curvature prior fusion (optional)
        # ----------------------------------------------------------------
        if curv_prior is not None and curv_fusion != "pixel_gating":
            from scipy.spatial.distance import mahalanobis as _mahalanobis

            mu_curv = curv_prior["mu"].numpy()
            sigma_inv = curv_prior["sigma_inv"]
            c_min = curv_prior["curv_min"]
            c_max = curv_prior["curv_max"]
            nm = curv_prior.get("_norm_mean")
            ns = curv_prior.get("_norm_std")

            v_test = curv_descs[tiff_path]
            if nm is not None:
                v_test = (v_test - nm) / ns
            s_curv_raw = _mahalanobis(v_test, mu_curv, sigma_inv)

            # min-max normalization + clamp
            s_curv_norm = (s_curv_raw - c_min) / (c_max - c_min + 1e-8)
            s_curv_norm = float(np.clip(s_curv_norm, 0.0, 1.0))

            if curv_fusion == "dampening":
                w_curv = s_curv_norm ** curv_gamma
                multiplier = curv_alpha + (1.0 - curv_alpha) * w_curv
                final_map = final_map * multiplier
            elif curv_fusion == "gating":
                # s_curv ≈ 0 (typical) → s_RAD 유지
                # s_curv > 0 (atypical) → s_RAD 증폭
                final_map = final_map * (1.0 + curv_alpha * s_curv_norm)
            elif curv_fusion == "additive":
                final_map = final_map + curv_alpha * s_curv_norm

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

        # ----------------------------------------------------------------
        # Step D-4: Pixel-level curvature gating (optional, curv_fusion="pixel_gating")
        # ----------------------------------------------------------------
        if curv_fusion == "pixel_gating" and tiff_path in curv_amaps:
            curv_map_hw = curv_amaps[tiff_path]  # [H_org, W_org] numpy
            curv_map_t = torch.from_numpy(curv_map_hw).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            curv_map_resized = F.interpolate(
                curv_map_t, size=resize_mask, mode='bilinear', align_corners=False,
            ).to(anomaly_map_up.device)  # [1, 1, resize_mask, resize_mask]
            # Multiplicative gating: s_final = s_RAD * (1 + λ * s_curv_map)
            anomaly_map_up = anomaly_map_up * (1.0 + curv_alpha * curv_map_resized)
            # Min-max re-normalization
            a_min = anomaly_map_up.min()
            a_max = anomaly_map_up.max()
            if a_max - a_min > 1e-8:
                anomaly_map_up = (anomaly_map_up - a_min) / (a_max - a_min)

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

        defect_type = raw_dataset.types[obj_idx]
        if vis_dir is not None and vis_count_per_type[defect_type] < max_vis:
            defect_vis_dir = os.path.join(vis_dir, defect_type)
            os.makedirs(defect_vis_dir, exist_ok=True)

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
            plt.savefig(os.path.join(defect_vis_dir, f"vis_{stem}.png"), dpi=150)
            plt.close()
            vis_count_per_type[defect_type] += 1

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
    parser.add_argument('--vis_max',     type=int,   default=5,
                        help='카테고리별 시각화 저장 장수 (default: 5)')

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
    parser.add_argument('--num_views', type=int, default=28,
                        help='사용할 뷰 수 (1=RGB only, 28=전체)')

    # FPFH score-level fusion
    parser.add_argument('--fpfh_alpha', type=float, default=0.0,
                        help='FPFH fusion weight (0.0=RGB only, 0.5=equal, 1.0=FPFH only)')
    parser.add_argument('--voxel_size', type=float, default=0.05,
                        help='FPFH voxel size for normal/feature radius')

    # 3D-ADS style feature-level concat
    parser.add_argument('--use_concat_bank', action='store_true',
                        help='Use RGB+FPFH concat bank (3D-ADS style, 801-dim)')
    parser.add_argument('--skip_cls_topk', action='store_true',
                        help='Skip CLS top-K filtering, use all train samples as neighbors')

    # Unified bank (원본 RAD 방식: 전체 카테고리 → 1 bank)
    parser.add_argument('--unified_bank_path', type=str, default=None,
                        help='통합 bank .pth 경로 (지정 시 per-category bank 대신 사용)')

    # Curvature prior fusion
    parser.add_argument('--use_curvature_prior', action='store_true',
                        help='curvature prior dampening fusion 활성화')
    parser.add_argument('--curv_alpha', type=float, default=0.3,
                        help='dampening 최소 보존 비율 (0=완전 억제, 1=no dampening)')
    parser.add_argument('--curv_gamma', type=float, default=1.0,
                        help='w_curv = s_curv^γ 곡선 제어 (>1: conservative, <1: sensitive)')
    parser.add_argument('--curv_fusion', type=str, default='dampening',
                        choices=['dampening', 'gating', 'additive', 'pixel_gating'],
                        help='curvature prior fusion 방식')
    parser.add_argument('--curv_normalize', action='store_true',
                        help='SI-12 descriptor per-dim z-score normalization')
    parser.add_argument('--curv_percentile', type=float, default=0,
                        help='SI-12 aggregation percentile (0=mean, 90=p90)')
    parser.add_argument('--curv_workers', type=int, default=2,
                        help='curvature descriptor 병렬 worker 수')

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
    print_fn(f"fpfh_alpha={args.fpfh_alpha}  voxel_size={args.voxel_size}  use_concat_bank={args.use_concat_bank}  skip_cls_topk={args.skip_cls_topk}")
    print_fn(f"unified_bank_path={args.unified_bank_path}")
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

    # Unified bank 로드 (전체 카테고리 1회 로드)
    unified_bank = None
    if args.unified_bank_path is not None:
        print_fn(f"[Unified] Loading unified bank: {args.unified_bank_path}")
        unified_bank = torch.load(args.unified_bank_path, map_location='cpu')
        N_total = unified_bank['cls_banks'][-1].shape[0]
        print_fn(f"[Unified] Loaded: N_total={N_total}, layers={unified_bank['layers']}")

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

        # bank 경로 구성
        bank_dir_cat = None
        if unified_bank is None:
            # Per-category bank
            bank_dir_cat = os.path.join(args.bank_dir, category)
            if not os.path.isdir(bank_dir_cat):
                print_fn(f"  Error: bank_dir not found: {bank_dir_cat}  (run build_bank_3d.py first)")
                continue

        vis_dir = os.path.join(
            args.save_dir, args.save_name, "test_vis", category
        )

        # FPFH bank 경로 구성 (score fusion: alpha > 0, concat: use_concat_bank)
        fpfh_bank_path = None
        if args.fpfh_alpha > 0 and not args.use_concat_bank and bank_dir_cat is not None:
            fpfh_bank_path = os.path.join(bank_dir_cat, "fpfh_bank.pth")

        # Curvature prior 로드
        curv_prior = None
        if args.use_curvature_prior and bank_dir_cat is not None:
            # Bank 파일 선택: --curv_percentile 에 따라 mean/p90 bank
            if args.curv_percentile > 0:
                bank_fname = f"SI-12_p{int(args.curv_percentile)}_bank.pth"
            else:
                bank_fname = "SI-12_mean_bank.pth"
            curv_prior_path = os.path.join(bank_dir_cat, bank_fname)
            # fallback: 기존 SI-12_bank.pth
            if not os.path.exists(curv_prior_path):
                curv_prior_path = os.path.join(bank_dir_cat, "SI-12_bank.pth")
            if os.path.exists(curv_prior_path):
                curv_prior = torch.load(curv_prior_path, map_location='cpu')
                # On-the-fly z-score normalization (--curv_normalize)
                if args.curv_normalize and "descriptors" in curv_prior:
                    from scipy.spatial.distance import mahalanobis as _maha
                    descs_raw = curv_prior["descriptors"].numpy()
                    nm = np.mean(descs_raw, axis=0)
                    ns = np.std(descs_raw, axis=0)
                    ns[ns < 1e-10] = 1.0
                    descs_n = (descs_raw - nm) / ns
                    mu_n = np.mean(descs_n, axis=0)
                    sig_n = np.cov(descs_n, rowvar=False) + 1e-6 * np.eye(descs_n.shape[1])
                    sig_inv_n = np.linalg.inv(sig_n)
                    dists_n = [_maha(descs_n[i], mu_n, sig_inv_n) for i in range(len(descs_n))]
                    curv_prior["mu"] = torch.from_numpy(mu_n).float()
                    curv_prior["sigma_inv"] = sig_inv_n
                    curv_prior["curv_min"] = float(min(dists_n))
                    curv_prior["curv_max"] = float(max(dists_n))
                    curv_prior["_norm_mean"] = nm
                    curv_prior["_norm_std"] = ns
                    print_fn(f"  [Curvature] Loaded + normalized: {curv_prior_path}")
                else:
                    curv_prior["sigma_inv"] = np.linalg.inv(curv_prior["sigma"].numpy())
                    print_fn(f"  [Curvature] Loaded: {curv_prior_path}")
            else:
                print_fn(f"  [Curvature] Warning: not found: {curv_prior_path}, disabled")

        # Curvature points bank 로드 (pixel_gating 전용)
        curv_points_bank = None
        if args.curv_fusion == "pixel_gating" and bank_dir_cat is not None:
            pts_bank_path = os.path.join(bank_dir_cat, "SI-12_points_bank.pth")
            if os.path.exists(pts_bank_path):
                curv_points_bank = torch.load(pts_bank_path, map_location='cpu')
                print_fn(f"  [Curvature] Points bank loaded: {pts_bank_path}")
            else:
                print_fn(f"  [Curvature] Warning: points bank not found: {pts_bank_path}")

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
            num_views         = args.num_views,
            fpfh_bank_path    = fpfh_bank_path,
            fpfh_alpha        = args.fpfh_alpha,
            voxel_size        = args.voxel_size,
            use_concat_bank   = args.use_concat_bank,
            skip_cls_topk     = args.skip_cls_topk,
            unified_bank      = unified_bank,
            curv_prior        = curv_prior,
            curv_alpha        = args.curv_alpha,
            curv_gamma        = args.curv_gamma,
            curv_fusion       = args.curv_fusion,
            curv_percentile   = args.curv_percentile,
            curv_points_bank  = curv_points_bank,
            curv_workers      = args.curv_workers,
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
