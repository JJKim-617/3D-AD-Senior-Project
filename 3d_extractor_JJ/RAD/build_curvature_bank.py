"""
build_curvature_bank.py — Train normal samples에서 curvature prior bank 빌드.

Per-point curvature features를 한 번만 계산하고 3가지 뱅크를 동시에 저장:

1. SI-12_points_bank.pth  — per-point raw features (aggregation 없음)
2. SI-12_mean_bank.pth    — mean aggregation + Gaussian
3. SI-12_p90_bank.pth     — p90 aggregation + Gaussian

Descriptor: [SI, C, K, H] × 3 scales (k=15, 30, 60) = 12-dim per point
"""

import argparse
import os
import warnings
from functools import partial
import multiprocessing as mp

import numpy as np
import torch
from scipy.spatial.distance import mahalanobis

from dataset_3d import MVTec3DRawDataset
from curvature_utils import compute_point_features_from_tiff

warnings.filterwarnings("ignore")


def _worker(tiff_path, edge_k):
    """멀티프로세싱 워커: 단일 TIFF → per-point (N, 12) features."""
    return compute_point_features_from_tiff(tiff_path, edge_k=edge_k)


def _fit_gaussian_and_save(
    descriptors: np.ndarray,
    category: str,
    save_path: str,
    reg_eps: float = 1e-6,
    agg_name: str = "mean",
):
    """Descriptor 행렬 [N_train, 12] → Gaussian fitting → .pth 저장."""
    N = len(descriptors)
    mu = np.mean(descriptors, axis=0)
    sigma = np.cov(descriptors, rowvar=False)
    sigma_reg = sigma + reg_eps * np.eye(sigma.shape[0])

    sigma_inv = np.linalg.inv(sigma_reg)
    train_dists = [mahalanobis(descriptors[i], mu, sigma_inv) for i in range(N)]
    curv_min = float(min(train_dists))
    curv_max = float(max(train_dists))

    print(f"  [{agg_name}] Mahalanobis range: [{curv_min:.4f}, {curv_max:.4f}]")

    torch.save(
        {
            "mu":          torch.from_numpy(mu).float(),
            "sigma":       torch.from_numpy(sigma_reg).float(),
            "curv_min":    curv_min,
            "curv_max":    curv_max,
            "descriptors": torch.from_numpy(descriptors).float(),
            "category":    category,
        },
        save_path,
    )
    print(f"  [{agg_name}] Saved: {save_path}")


def build_curvature_bank(
    data_path: str,
    category: str,
    bank_dir: str,
    edge_k: float = 5.0,
    reg_eps: float = 1e-6,
    num_workers: int = 0,
):
    """
    한 카테고리의 train normal samples에서 3가지 curvature bank 동시 빌드.

    Per-point features를 한 번만 계산:
      1. SI-12_points_bank.pth — per-point raw features
      2. SI-12_mean_bank.pth   — mean aggregation + Gaussian
      3. SI-12_p90_bank.pth    — p90 aggregation + Gaussian
    """
    raw_dataset = MVTec3DRawDataset(data_path, category, phase='train')
    N = len(raw_dataset)
    print(f"\n[Curvature-Bank] {category}: {N} train samples (workers={num_workers})")

    tiff_paths = [raw_dataset.tiff_paths[i] for i in range(N)]

    # Step 1: per-point features 계산 (비싼 연산 — 1회만)
    if num_workers > 0:
        worker_fn = partial(_worker, edge_k=edge_k)
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            point_features_list = []
            for i, pf in enumerate(pool.imap(worker_fn, tiff_paths), 1):
                point_features_list.append(pf)
                if i % 10 == 0 or i == N:
                    n_pts = len(pf)
                    print(f"  [{i}/{N}] done  n_points={n_pts}", flush=True)
    else:
        point_features_list = []
        for i in range(N):
            pf = compute_point_features_from_tiff(tiff_paths[i], edge_k=edge_k)
            point_features_list.append(pf)
            if (i + 1) % 10 == 0 or (i + 1) == N:
                print(f"  [{i + 1}/{N}] done  n_points={len(pf)}")

    cat_bank_dir = os.path.join(bank_dir, category)
    os.makedirs(cat_bank_dir, exist_ok=True)

    # Bank 1: per-point raw features + global point-level Gaussian
    all_points = np.concatenate(
        [pf for pf in point_features_list if len(pf) > 0], axis=0
    )  # [N_total_pts, 12]

    pt_mu = np.mean(all_points, axis=0)
    pt_sigma = np.cov(all_points, rowvar=False) + reg_eps * np.eye(12)
    pt_sigma_inv = np.linalg.inv(pt_sigma)

    # Vectorized Mahalanobis for normalization stats (subsample if too large)
    if len(all_points) > 500_000:
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(len(all_points), 500_000, replace=False)
        sub_pts = all_points[sub_idx]
    else:
        sub_pts = all_points
    diff = sub_pts - pt_mu
    maha_sq = np.sum(diff @ pt_sigma_inv * diff, axis=1)
    maha = np.sqrt(np.clip(maha_sq, 0, None))
    pt_min = float(np.percentile(maha, 1))
    pt_max = float(np.percentile(maha, 99))

    print(f"  [points] Global Gaussian: {len(all_points)} points, "
          f"Maha range (p1-p99): [{pt_min:.4f}, {pt_max:.4f}]")

    points_path = os.path.join(cat_bank_dir, "SI-12_points_bank.pth")
    torch.save(
        {
            "point_features": [torch.from_numpy(pf).float() for pf in point_features_list],
            "pt_mu":          torch.from_numpy(pt_mu).float(),
            "pt_sigma":       torch.from_numpy(pt_sigma).float(),
            "pt_min":         pt_min,
            "pt_max":         pt_max,
            "category":       category,
        },
        points_path,
    )
    total_pts = sum(len(pf) for pf in point_features_list)
    size_mb = os.path.getsize(points_path) / (1024 * 1024)
    print(f"  [points] Saved: {points_path}  ({total_pts} total points, {size_mb:.1f} MB)")

    # Step 2: aggregation → sample descriptors
    mean_descs = []
    p90_descs = []
    for pf in point_features_list:
        if len(pf) == 0:
            mean_descs.append(np.zeros(12, dtype=np.float64))
            p90_descs.append(np.zeros(12, dtype=np.float64))
        else:
            mean_descs.append(np.mean(pf, axis=0).astype(np.float64))
            p90_descs.append(np.percentile(pf, 90, axis=0).astype(np.float64))

    mean_descs = np.stack(mean_descs, axis=0)  # [N, 12]
    p90_descs  = np.stack(p90_descs, axis=0)   # [N, 12]

    # Bank 2: mean aggregation
    _fit_gaussian_and_save(
        mean_descs, category,
        os.path.join(cat_bank_dir, "SI-12_mean_bank.pth"),
        reg_eps=reg_eps, agg_name="mean",
    )

    # Bank 3: p90 aggregation
    _fit_gaussian_and_save(
        p90_descs, category,
        os.path.join(cat_bank_dir, "SI-12_p90_bank.pth"),
        reg_eps=reg_eps, agg_name="p90",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build curvature prior banks (points + mean + p90)"
    )
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--item_list', nargs='+', required=True)
    parser.add_argument('--bank_dir', type=str, required=True)
    parser.add_argument('--edge_k', type=float, default=5.0)
    parser.add_argument('--reg_eps', type=float, default=1e-6)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='병렬 프로세스 수 (0=sequential)')
    args = parser.parse_args()

    for category in args.item_list:
        build_curvature_bank(
            data_path=args.data_path,
            category=category,
            bank_dir=args.bank_dir,
            edge_k=args.edge_k,
            reg_eps=args.reg_eps,
            num_workers=args.num_workers,
        )

    print("\n[Curvature-Bank] All done.")


if __name__ == '__main__':
    main()
