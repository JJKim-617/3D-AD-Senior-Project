"""
curvature_utils.py — TIFF organized point cloud에서 multi-scale curvature descriptor 추출.

Pipeline:
    TIFF [H,W,3] → valid filter → depth discontinuity edge filter
    → local quadric fitting (Weingarten map) → principal curvatures (κ₁, κ₂)
    → [SI, C, K, H] × 3 scales (k=15,30,60)
    → 12-dim sample descriptor (per-dim mean or percentile)

References:
    - curvature estimation: 3d_extractor_ej/geo_features_bench/curvature_features.py
    - Koenderink & van Doorn, "Surface shape and curvature scales", 1992
"""

import numpy as np
import open3d as o3d
import tifffile
from scipy.spatial import KDTree


DESCRIPTOR_DIM = 12
SCALES = [15, 30, 60]


# ---------------------------------------------------------------------------
# Depth discontinuity filtering
# ---------------------------------------------------------------------------

def filter_edge_points(
    data: np.ndarray,
    k: float = 5.0,
) -> np.ndarray:
    """
    Organized point cloud [H, W, 3]에서 depth discontinuity 기반 경계 포인트를 제거.

    물체 외곽 포인트는 kNN 이웃이 한쪽으로만 분포하여
    curvature 추정을 오염시킨다. 인접 픽셀 간 depth 차이로 경계를 탐지.

    Args:
        data: [H, W, 3] float64 organized point cloud.
        k:    adaptive threshold 배수 (τ = median_dist * k).

    Returns:
        valid_mask: [H, W] bool — True인 포인트만 curvature 계산에 사용.
    """
    H, W = data.shape[:2]

    # 유효 포인트 마스크 (NaN 또는 all-zero 제외)
    pts = data.reshape(-1, 3)
    base_valid = (np.any(pts != 0, axis=1) & ~np.isnan(pts).any(axis=1)).reshape(H, W)

    # depth = 원점으로부터 거리
    depth = np.linalg.norm(data, axis=-1)  # [H, W]
    depth[~base_valid] = 0.0

    # 수평/수직 인접 픽셀 간 depth 차이
    dx = np.abs(depth[:, 1:] - depth[:, :-1])  # [H, W-1]
    dy = np.abs(depth[1:, :] - depth[:-1, :])  # [H-1, W]

    # adaptive threshold: 유효 쌍만으로 median 계산
    dx_valid = base_valid[:, 1:] & base_valid[:, :-1]
    dy_valid = base_valid[1:, :] & base_valid[:-1, :]

    valid_dists = np.concatenate([dx[dx_valid], dy[dy_valid]])
    if len(valid_dists) == 0:
        return base_valid

    median_dist = np.median(valid_dists)
    tau = median_dist * k

    # edge mask: depth 차이가 τ 초과인 픽셀 양쪽을 edge로 표시
    edge_mask = np.zeros((H, W), dtype=bool)

    edge_x = dx > tau  # [H, W-1]
    edge_mask[:, :-1] |= edge_x
    edge_mask[:, 1:]  |= edge_x

    edge_y = dy > tau  # [H-1, W]
    edge_mask[:-1, :] |= edge_y
    edge_mask[1:, :]  |= edge_y

    return base_valid & ~edge_mask


# ---------------------------------------------------------------------------
# Principal curvature via local quadric fitting (from EJ)
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

    각 포인트의 k-NN을 local tangent frame으로 투영한 뒤,
    2차 곡면 z = au² + buv + cv²를 fitting하여 Weingarten map eigenvalue → κ₁, κ₂ 추출.

    Args:
        pts:        (N, 3) float32/64 point cloud.
        normals:    (N, 3) float32/64 surface normals.
        tree:       scipy KDTree built from pts.
        k:          이웃 수.
        batch_size: 배치 크기 (메모리 관리).

    Returns:
        kappa1: (N,) 최대 주곡률
        kappa2: (N,) 최소 주곡률
    """
    N = len(pts)
    _, idxs = tree.query(pts, k=k)  # (N, k)

    kappa1 = np.zeros(N, dtype=np.float32)
    kappa2 = np.zeros(N, dtype=np.float32)

    # Vectorized tangent frame construction
    norms_n = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10)

    # Choose reference vector for cross product
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

        # Neighbors: (b, k, 3)
        neighbors = pts[idxs[start:end]] - pts[start:end, np.newaxis, :]

        # Project to local frames: u, v, w (b, k)
        u = np.einsum('bkd,bd->bk', neighbors, t1[start:end])
        v = np.einsum('bkd,bd->bk', neighbors, t2[start:end])
        w = np.einsum('bkd,bd->bk', neighbors, norms_n[start:end])

        # Fit z = a*u² + b_coef*u*v + c*v² per point
        A_mat = np.stack([u**2, u * v, v**2], axis=2)  # (b, k, 3)

        # Batch least squares: (A^T A)^{-1} A^T w
        ATA = np.einsum('bki,bkj->bij', A_mat, A_mat)  # (b, 3, 3)
        ATw = np.einsum('bki,bk->bi', A_mat, w)         # (b, 3)

        # Regularize and solve
        ATA += np.eye(3)[np.newaxis] * 1e-8
        try:
            coeffs = np.linalg.solve(
                ATA, ATw[:, :, np.newaxis]
            ).squeeze(-1)  # (b, 3)
        except np.linalg.LinAlgError:
            for i in range(b):
                try:
                    c_sol, _, _, _ = np.linalg.lstsq(
                        A_mat[i], w[i], rcond=None
                    )
                except Exception:
                    continue
                S = np.array([[2*c_sol[0], c_sol[1]],
                              [c_sol[1], 2*c_sol[2]]])
                ev = np.linalg.eigvalsh(S)
                kappa2[start + i] = ev[0]
                kappa1[start + i] = ev[1]
            continue

        a_coef = coeffs[:, 0]
        b_coef = coeffs[:, 1]
        c_coef = coeffs[:, 2]

        # Weingarten map eigenvalues: S = [[2a, b], [b, 2c]]
        trace = 2 * a_coef + 2 * c_coef
        det = 4 * a_coef * c_coef - b_coef**2
        disc = np.clip(trace**2 - 4 * det, 0, None)
        sqrt_disc = np.sqrt(disc)

        kappa1[start:end] = (trace + sqrt_disc) / 2
        kappa2[start:end] = (trace - sqrt_disc) / 2

    return kappa1, kappa2


# ---------------------------------------------------------------------------
# Multi-scale curvature features (12-dim per point)
# ---------------------------------------------------------------------------

def compute_multiscale_curvature_features(
    pts: np.ndarray,
    scales: list = None,
    normal_k: int = 30,
) -> np.ndarray:
    """
    Multi-scale principal curvature → 12-dim per-point descriptor.

    각 스케일(k=15,30,60)에서 [SI, C, K, H] 4개 특징을 계산하여 concat.

    Args:
        pts:       (N, 3) point cloud.
        scales:    k-NN 스케일 리스트 (default: [15, 30, 60]).
        normal_k:  법선 추정 이웃 수.

    Returns:
        (N, 12) float32 descriptor matrix.
    """
    if scales is None:
        scales = SCALES

    pts = pts.astype(np.float32)

    # Normal estimation via Open3D
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
# End-to-end: TIFF → per-point features or sample descriptor
# ---------------------------------------------------------------------------

def compute_point_features_from_tiff(
    tiff_path: str,
    edge_k: float = 5.0,
) -> np.ndarray:
    """
    TIFF 파일 → per-point 12-dim multi-scale curvature features.

    Returns:
        (N_valid, 12) float32 per-point features, or empty (0, 12) if too few points.
    """
    data = tifffile.imread(tiff_path).astype(np.float64)

    valid_mask = filter_edge_points(data, k=edge_k)
    pts = data[valid_mask]

    if len(pts) < max(SCALES):
        return np.zeros((0, DESCRIPTOR_DIM), dtype=np.float32)

    pts[:, 1] *= -1
    pts[:, 2] *= -1

    return compute_multiscale_curvature_features(pts)


def compute_curvature_descriptor_from_tiff(
    tiff_path: str,
    edge_k: float = 5.0,
    percentile: float = 0,
    **kwargs,
) -> np.ndarray:
    """
    TIFF 파일 → 12-dim multi-scale curvature sample descriptor.

    Pipeline:
        1. TIFF 로드 [H,W,3]
        2. Depth discontinuity filtering
        3. CPMF 좌표 변환
        4. Multi-scale quadric fitting → 12-dim per point
        5. Aggregation → 12-dim sample descriptor
           - percentile=0 (default): per-dim mean
           - percentile>0 (e.g. 90): per-dim Nth percentile

    Args:
        tiff_path:   organized point cloud TIFF 경로.
        edge_k:      depth discontinuity threshold 배수.
        percentile:  aggregation percentile (0=mean, 90=p90 등).

    Returns:
        [12] float64 descriptor.
    """
    point_features = compute_point_features_from_tiff(tiff_path, edge_k=edge_k)

    if len(point_features) == 0:
        return np.zeros(DESCRIPTOR_DIM, dtype=np.float64)

    if percentile > 0:
        descriptor = np.percentile(point_features, percentile, axis=0)
    else:
        descriptor = np.mean(point_features, axis=0)

    return descriptor.astype(np.float64)


def compute_multi_agg_descriptors_from_tiff(
    tiff_path: str,
    edge_k: float = 5.0,
) -> dict:
    """
    TIFF 파일 → mean, p90 두 가지 aggregation descriptor를 동시에 반환.

    Per-point features를 한 번만 계산하고 mean/p90 모두 추출.

    Returns:
        {"mean": [12] float64, "p90": [12] float64}
    """
    point_features = compute_point_features_from_tiff(tiff_path, edge_k=edge_k)

    zeros = np.zeros(DESCRIPTOR_DIM, dtype=np.float64)
    if len(point_features) == 0:
        return {"mean": zeros, "p90": zeros}

    return {
        "mean": np.mean(point_features, axis=0).astype(np.float64),
        "p90":  np.percentile(point_features, 90, axis=0).astype(np.float64),
    }


def compute_curvature_anomaly_map_from_tiff(
    tiff_path: str,
    pt_mu: np.ndarray,
    pt_sigma_inv: np.ndarray,
    pt_min: float,
    pt_max: float,
    edge_k: float = 5.0,
) -> np.ndarray:
    """
    TIFF → per-point Mahalanobis distance → normalized curvature anomaly map [H, W].

    Args:
        tiff_path:     organized point cloud TIFF 경로.
        pt_mu:         [12] global point-level mean.
        pt_sigma_inv:  [12, 12] inverse covariance.
        pt_min:        training point Mahalanobis p1 (normalization).
        pt_max:        training point Mahalanobis p99 (normalization).
        edge_k:        depth discontinuity threshold 배수.

    Returns:
        [H, W] float32 anomaly map, normalized to ~[0, 1].
    """
    data = tifffile.imread(tiff_path).astype(np.float64)  # [H, W, 3]
    H, W = data.shape[:2]

    valid_mask = filter_edge_points(data, k=edge_k)
    pts = data[valid_mask]  # [N_valid, 3]

    anomaly_map = np.zeros((H, W), dtype=np.float32)

    if len(pts) < max(SCALES):
        return anomaly_map

    # CPMF 좌표 변환
    pts[:, 1] *= -1
    pts[:, 2] *= -1

    # Per-point 12-dim features
    point_features = compute_multiscale_curvature_features(pts)  # [N_valid, 12]

    # Vectorized Mahalanobis distance
    diff = point_features.astype(np.float64) - pt_mu  # [N, 12]
    maha_sq = np.sum(diff @ pt_sigma_inv * diff, axis=1)  # [N]
    maha = np.sqrt(np.clip(maha_sq, 0, None))

    # Min-max normalization (using training stats)
    maha_norm = (maha - pt_min) / (pt_max - pt_min + 1e-8)
    maha_norm = np.clip(maha_norm, 0.0, None).astype(np.float32)

    # Map back to organized grid
    anomaly_map[valid_mask] = maha_norm

    return anomaly_map
