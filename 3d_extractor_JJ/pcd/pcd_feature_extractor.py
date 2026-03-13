"""
pcd_feature_extractor.py — PCD 파일에서 12-dim multi-scale curvature descriptor 추출.

3d_extractor_ej/geo_features_bench/curvature_features.py 의 로직을 기반으로
unorganized PCD 파일을 입력으로 받아 12-dim per-point feature를 추출한다.

12-dim feature layout ([SI, C, K, H] × 3 scales):
    dim  0: shape_index_k15
    dim  1: curvedness_k15
    dim  2: gaussian_curv_k15
    dim  3: mean_curv_k15
    dim  4: shape_index_k30
    dim  5: curvedness_k30
    dim  6: gaussian_curv_k30
    dim  7: mean_curv_k30
    dim  8: shape_index_k60
    dim  9: curvedness_k60
    dim 10: gaussian_curv_k60
    dim 11: mean_curv_k60

References:
    - Koenderink & van Doorn, "Surface shape and curvature scales", 1992
    - 3d_extractor_ej/geo_features_bench/curvature_features.py

Usage:
    # 단일 파일
    python pcd_feature_extractor.py --input cloud.pcd --output features.npy

    # 디렉토리 일괄 처리
    python pcd_feature_extractor.py --input_dir /path/to/pcds --output_dir /path/to/out

    # API
    from pcd_feature_extractor import extract_features_from_pcd
    feat = extract_features_from_pcd("cloud.pcd")  # (N, 12) float32
"""

import os
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree


DESCRIPTOR_DIM = 12
SCALES = [15, 30, 60]
N_POINTS_DEFAULT = 20000

FEATURE_NAMES = [
    f"{feat}_k{k}"
    for k in SCALES
    for feat in ["shape_index", "curvedness", "gaussian_curv", "mean_curv"]
]


# ---------------------------------------------------------------------------
# Principal curvature estimation via local quadric fitting
# (EJ 코드 curvature_features.py 의 _estimate_principal_curvatures 와 동일 로직)
# ---------------------------------------------------------------------------

def _estimate_principal_curvatures(
    pts: np.ndarray,
    normals: np.ndarray,
    tree: KDTree,
    k: int = 30,
    batch_size: int = 2000,
) -> tuple:
    """
    Local quadric fitting으로 주곡률 추정 (batch vectorized).

    각 포인트의 k-NN을 local tangent frame으로 투영한 뒤
    2차 곡면 z = au² + buv + cv² 를 fitting,
    Weingarten map S = [[2a, b], [b, 2c]] 의 eigenvalue → κ₁, κ₂.

    Returns:
        kappa1: (N,) 최대 주곡률
        kappa2: (N,) 최소 주곡률
    """
    N = len(pts)
    _, idxs = tree.query(pts, k=k)  # (N, k)

    kappa1 = np.zeros(N, dtype=np.float32)
    kappa2 = np.zeros(N, dtype=np.float32)

    norms_n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

    ref = np.zeros_like(norms_n)
    mask_x = np.abs(norms_n[:, 0]) < 0.9
    ref[mask_x] = [1, 0, 0]
    ref[~mask_x] = [0, 1, 0]

    t1 = np.cross(norms_n, ref)
    t1 = t1 / (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-10)
    t2 = np.cross(norms_n, t1)
    t2 = t2 / (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-10)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        b = end - start

        neighbors = pts[idxs[start:end]] - pts[start:end, np.newaxis, :]  # (b, k, 3)

        u = np.einsum('bkd,bd->bk', neighbors, t1[start:end])
        v = np.einsum('bkd,bd->bk', neighbors, t2[start:end])
        w = np.einsum('bkd,bd->bk', neighbors, norms_n[start:end])

        A_mat = np.stack([u**2, u * v, v**2], axis=2)  # (b, k, 3)

        ATA = np.einsum('bki,bkj->bij', A_mat, A_mat)  # (b, 3, 3)
        ATw = np.einsum('bki,bk->bi', A_mat, w)         # (b, 3)

        ATA += np.eye(3)[np.newaxis] * 1e-8
        try:
            coeffs = np.linalg.solve(ATA, ATw[:, :, np.newaxis]).squeeze(-1)  # (b, 3)
        except np.linalg.LinAlgError:
            for i in range(b):
                try:
                    c_sol, _, _, _ = np.linalg.lstsq(A_mat[i], w[i], rcond=None)
                except Exception:
                    continue
                S = np.array([[2*c_sol[0], c_sol[1]], [c_sol[1], 2*c_sol[2]]])
                ev = np.linalg.eigvalsh(S)
                kappa2[start + i] = ev[0]
                kappa1[start + i] = ev[1]
            continue

        a_coef = coeffs[:, 0]
        b_coef = coeffs[:, 1]
        c_coef = coeffs[:, 2]

        # Weingarten map eigenvalues
        trace = 2 * a_coef + 2 * c_coef
        det = 4 * a_coef * c_coef - b_coef**2
        disc = np.clip(trace**2 - 4 * det, 0, None)
        sqrt_disc = np.sqrt(disc)

        kappa1[start:end] = (trace + sqrt_disc) / 2
        kappa2[start:end] = (trace - sqrt_disc) / 2

    return kappa1, kappa2


# ---------------------------------------------------------------------------
# 12-dim multi-scale feature computation
# (EJ 코드 compute_curvature_features 와 동일 로직)
# ---------------------------------------------------------------------------

def compute_features(
    pts: np.ndarray,
    scales: list = None,
    normal_k: int = 30,
) -> np.ndarray:
    """
    (N, 3) point cloud → (N, 12) multi-scale curvature descriptor.

    각 스케일(k=15,30,60)에서 [SI, C, K, H] 4개 특징을 계산하여 concat.

    Args:
        pts:      (N, 3) float array.
        scales:   k-NN 스케일 리스트 (default: [15, 30, 60]).
        normal_k: 법선 추정 이웃 수.

    Returns:
        (N, 12) float32.
    """
    if scales is None:
        scales = SCALES

    pts = pts.astype(np.float32)

    # Normal estimation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=normal_k)
    )
    normals = np.asarray(pcd.normals, dtype=np.float32)

    tree = KDTree(pts)

    all_features = []
    for s_k in scales:
        kappa1, kappa2 = _estimate_principal_curvatures(pts, normals, tree, k=s_k)

        # Shape Index: SI = (2/π) * arctan2(κ₁+κ₂, κ₁-κ₂)
        denom = kappa1 - kappa2 + 1e-10
        shape_index = (2.0 / np.pi) * np.arctan2(kappa1 + kappa2, denom)

        # Curvedness: C = sqrt((κ₁² + κ₂²) / 2)
        curvedness = np.sqrt((kappa1**2 + kappa2**2) / 2.0)

        # Gaussian curvature: K = κ₁ · κ₂
        gaussian_curv = kappa1 * kappa2

        # Mean curvature: H = (κ₁ + κ₂) / 2
        mean_curv = (kappa1 + kappa2) / 2.0

        features = np.column_stack([
            shape_index, curvedness, gaussian_curv, mean_curv
        ]).astype(np.float32)
        all_features.append(features)

    return np.concatenate(all_features, axis=1)  # (N, 12)


# ---------------------------------------------------------------------------
# PCD loading with FPS subsampling
# (EJ 코드 run_si_anomalyshapenet.py 의 load_pcd 와 동일 로직)
# ---------------------------------------------------------------------------

def load_pcd(path: str, n_points: int = N_POINTS_DEFAULT) -> np.ndarray:
    """
    PCD 파일 로드 + FPS subsampling.

    Args:
        path:     PCD 파일 경로 (.pcd, .ply 등 Open3D 지원 포맷).
        n_points: FPS 서브샘플링 목표 수 (0이면 서브샘플링 안 함).

    Returns:
        (N, 3) float32.
    """
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)

    if n_points > 0 and len(pts) > n_points:
        pcd_down = pcd.farthest_point_down_sample(n_points)
        pts = np.asarray(pcd_down.points, dtype=np.float32)

    return pts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features_from_pcd(
    pcd_path: str,
    n_points: int = N_POINTS_DEFAULT,
    scales: list = None,
    normal_k: int = 30,
) -> np.ndarray:
    """
    PCD 파일 → per-point 12-dim multi-scale curvature features.

    Args:
        pcd_path: PCD 파일 경로.
        n_points: FPS 서브샘플링 목표 수 (0이면 전체 사용).
        scales:   curvature 계산 스케일 (default: [15, 30, 60]).
        normal_k: 법선 추정 이웃 수.

    Returns:
        (N, 12) float32, 포인트 부족 시 (0, 12).
    """
    if scales is None:
        scales = SCALES

    pts = load_pcd(pcd_path, n_points=n_points)

    if len(pts) < max(scales):
        print(f"  [WARN] 포인트 수 부족 ({len(pts)} < {max(scales)}): {pcd_path}")
        return np.zeros((0, DESCRIPTOR_DIM), dtype=np.float32)

    return compute_features(pts, scales=scales, normal_k=normal_k)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _process_single(pcd_path: str, output_path: str, n_points: int):
    print(f"  {os.path.basename(pcd_path)} ", end="", flush=True)
    feat = extract_features_from_pcd(pcd_path, n_points=n_points)
    np.save(output_path, feat)
    print(f"→ shape={feat.shape}  saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PCD → 12-dim multi-scale curvature features"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', type=str, help='단일 PCD 파일 경로')
    group.add_argument('--input_dir', type=str, help='PCD 파일들이 있는 디렉토리')

    parser.add_argument('--output', type=str, default=None,
                        help='출력 .npy 경로 (--input 사용 시)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='출력 디렉토리 (--input_dir 사용 시, 없으면 input_dir에 저장)')
    parser.add_argument('--n_points', type=int, default=N_POINTS_DEFAULT,
                        help=f'FPS 서브샘플링 목표 수 (0=전체, default={N_POINTS_DEFAULT})')
    parser.add_argument('--ext', type=str, default='.pcd',
                        help='처리할 파일 확장자 (default: .pcd)')
    args = parser.parse_args()

    if args.input is not None:
        out = args.output or os.path.splitext(args.input)[0] + "_feat12.npy"
        _process_single(args.input, out, args.n_points)

    else:
        input_dir = args.input_dir
        output_dir = args.output_dir or input_dir
        os.makedirs(output_dir, exist_ok=True)

        pcd_files = sorted([
            f for f in os.listdir(input_dir)
            if f.lower().endswith(args.ext)
        ])

        if not pcd_files:
            print(f"[WARN] {input_dir} 에서 {args.ext} 파일을 찾을 수 없습니다.")
            return

        print(f"[pcd_feature_extractor] {len(pcd_files)}개 파일 처리 시작 "
              f"(n_points={args.n_points})\n")

        for fname in pcd_files:
            pcd_path = os.path.join(input_dir, fname)
            out_name = os.path.splitext(fname)[0] + "_feat12.npy"
            out_path = os.path.join(output_dir, out_name)
            _process_single(pcd_path, out_path, args.n_points)

        print(f"\n[pcd_feature_extractor] 완료. 결과: {output_dir}")


if __name__ == '__main__':
    main()
