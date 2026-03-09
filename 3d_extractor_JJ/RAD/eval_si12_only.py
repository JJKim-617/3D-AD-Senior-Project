"""
eval_si12_only.py — SI-12 curvature descriptor만으로 anomaly detection 평가.

RGB (RAD) 없이, SI-12 bank의 Mahalanobis distance만으로 sample-level 분류 성능 측정.
"""

import argparse
import os
import warnings
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from dataset_3d import MVTec3DRawDataset
from curvature_utils import compute_curvature_descriptor_from_tiff

warnings.filterwarnings("ignore")


def f1_score_max(gt, pr):
    """최적 threshold에서의 F1 score."""
    thresholds = np.unique(pr)
    if len(thresholds) > 1000:
        thresholds = np.percentile(pr, np.linspace(0, 100, 1000))
    best_f1 = 0.0
    for t in thresholds:
        pred = (pr >= t).astype(int)
        f1 = f1_score(gt, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


def eval_si12_category(
    data_path: str,
    category: str,
    bank_dir: str,
    num_workers: int = 16,
    log_path: str = None,
    curv_normalize: bool = False,
    curv_percentile: float = 0,
):
    """한 카테고리의 SI-12 only 평가. Returns dict or None."""

    # SI-12 bank 로드 (percentile에 따라 mean/p90 bank 선택)
    if curv_percentile > 0:
        bank_fname = f"SI-12_p{int(curv_percentile)}_bank.pth"
    else:
        bank_fname = "SI-12_mean_bank.pth"
    bank_path = os.path.join(bank_dir, category, bank_fname)
    if not os.path.exists(bank_path):
        bank_path = os.path.join(bank_dir, category, "SI-12_bank.pth")
    if not os.path.exists(bank_path):
        print(f"  [SKIP] {category}: {bank_fname} not found")
        return None

    bank = torch.load(bank_path, map_location='cpu')

    # On-the-fly z-score normalization
    norm_mean_np = None
    norm_std_np = None
    if curv_normalize and "descriptors" in bank:
        descs_raw = bank["descriptors"].numpy()
        norm_mean_np = np.mean(descs_raw, axis=0)
        norm_std_np = np.std(descs_raw, axis=0)
        norm_std_np[norm_std_np < 1e-10] = 1.0
        descs_n = (descs_raw - norm_mean_np) / norm_std_np
        mu = np.mean(descs_n, axis=0)
        sigma_n = np.cov(descs_n, rowvar=False) + 1e-6 * np.eye(descs_n.shape[1])
        sigma_inv = np.linalg.inv(sigma_n)
        dists_n = [mahalanobis(descs_n[i], mu, sigma_inv) for i in range(len(descs_n))]
        c_min = float(min(dists_n))
        c_max = float(max(dists_n))
    else:
        mu = bank["mu"].numpy()
        sigma_inv = np.linalg.inv(bank["sigma"].numpy())
        c_min = bank["curv_min"]
        c_max = bank["curv_max"]

    # Test dataset
    raw_dataset = MVTec3DRawDataset(data_path, category, phase='test')
    N = len(raw_dataset)
    if N == 0:
        print(f"  [SKIP] {category}: no test samples")
        return None

    print(f"\n{'='*60}")
    print(f"[Eval-SI12] Category: {category}")
    print(f"{raw_dataset}")

    # Precompute descriptors (parallel)
    tiff_paths = [raw_dataset.tiff_paths[i] for i in range(N)]
    n_workers = min(num_workers, N)
    _curv_fn = partial(compute_curvature_descriptor_from_tiff, percentile=curv_percentile)

    if n_workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            descriptors = list(pool.map(_curv_fn, tiff_paths))
    else:
        descriptors = [_curv_fn(p) for p in tiff_paths]

    # Mahalanobis scores
    labels = []
    scores = []
    scores_norm = []

    for i in range(N):
        label = int(raw_dataset.labels[i])
        v_test = descriptors[i]
        if norm_mean_np is not None:
            v_test = (v_test - norm_mean_np) / norm_std_np
        s_raw = mahalanobis(v_test, mu, sigma_inv)
        s_norm = float(np.clip((s_raw - c_min) / (c_max - c_min + 1e-8), 0.0, None))

        labels.append(label)
        scores.append(s_raw)
        scores_norm.append(s_norm)

    labels = np.array(labels)
    scores = np.array(scores)
    scores_norm = np.array(scores_norm)

    # Metrics
    auroc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    f1 = f1_score_max(labels, scores)

    # rad_3d.py 동일 포맷 출력 + 즉시 로그 저장
    line = f"{category}: I-AUROC:{auroc:.4f}, I-AP:{ap:.4f}, I-F1:{f1:.4f}"
    print(line)

    if log_path:
        with open(log_path, 'a') as f:
            f.write(line + "\n")

    # Per-defect breakdown
    types = raw_dataset.types
    unique_types = sorted(set(types))

    print(f"  Score stats — normal: {scores_norm[labels==0].mean():.3f}±{scores_norm[labels==0].std():.3f}"
          f"  anomaly: {scores_norm[labels==1].mean():.3f}±{scores_norm[labels==1].std():.3f}")

    for dt in unique_types:
        mask = types == dt
        n_dt = mask.sum()
        mean_s = scores_norm[mask].mean()
        print(f"    {dt:20s}: n={n_dt:3d}  mean_score={mean_s:.4f}")

    return {
        "category": category,
        "auroc": auroc,
        "ap": ap,
        "f1": f1,
        "n_test": N,
    }


def main():
    parser = argparse.ArgumentParser(description="SI-12 curvature-only anomaly detection")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--item_list', nargs='+', required=True)
    parser.add_argument('--bank_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--save_name', type=str, default='si12_only')
    parser.add_argument('--curv_normalize', action='store_true',
                        help='SI-12 descriptor per-dim z-score normalization')
    parser.add_argument('--curv_percentile', type=float, default=0,
                        help='SI-12 aggregation percentile (0=mean, 90=p90)')
    args = parser.parse_args()

    # 로그 파일 준비 (카테고리별 즉시 저장)
    log_dir = None
    log_path = None
    if args.save_dir:
        log_dir = os.path.join(args.save_dir, args.save_name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "log.txt")
        with open(log_path, 'w') as f:
            f.write("SI-12 Curvature-Only Evaluation\n")
            f.write("=" * 60 + "\n")

    results = []
    for category in args.item_list:
        r = eval_si12_category(
            data_path=args.data_path,
            category=category,
            bank_dir=args.bank_dir,
            num_workers=args.num_workers,
            log_path=log_path,
            curv_normalize=args.curv_normalize,
            curv_percentile=args.curv_percentile,
        )
        if r is not None:
            results.append(r)

    # Summary
    if results:
        print("\n" + "=" * 60)
        print(f"{'Category':20s} | {'I-AUROC':>8s} | {'I-AP':>8s} | {'I-F1':>8s}")
        print("-" * 60)
        for r in results:
            print(f"{r['category']:20s} | {r['auroc']:8.4f} | {r['ap']:8.4f} | {r['f1']:8.4f}")
        mean_auroc = np.mean([r['auroc'] for r in results])
        mean_ap = np.mean([r['ap'] for r in results])
        mean_f1 = np.mean([r['f1'] for r in results])
        print("-" * 60)
        print(f"{'Mean':20s} | {mean_auroc:8.4f} | {mean_ap:8.4f} | {mean_f1:8.4f}")
        print("=" * 60)

        if log_path:
            with open(log_path, 'a') as f:
                f.write(f"\nMean: I-AUROC:{mean_auroc:.4f}, I-AP:{mean_ap:.4f}, I-F1:{mean_f1:.4f}\n")
            print(f"\nLog saved: {log_path}")


if __name__ == '__main__':
    main()
