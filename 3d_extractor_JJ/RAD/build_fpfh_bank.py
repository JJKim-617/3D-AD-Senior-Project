"""
build_fpfh_bank.py — TIFF organized point cloud에서 FPFH bank 빌드.

각 카테고리의 train 샘플에서 FPFH 33-dim feature를 추출하고,
DINOv3 patch 해상도(28×28)에 맞춰 downsample한 뒤 bank으로 저장.

출력 파일:
    bank_dir/{category}/fpfh_bank.pth
        {
            "fpfh_patches": Tensor[N_obj, 784, 33],  # float16
            "category": str,
        }
"""

import argparse
import os
import warnings

import numpy as np
import open3d as o3d
import torch
import tifffile

from dataset_3d import MVTec3DRawDataset

warnings.filterwarnings("ignore")


def compute_fpfh_from_tiff(tiff_path: str, voxel_size: float = 0.05) -> torch.Tensor:
    """
    TIFF organized point cloud → FPFH feature map [1, 33, H, W].

    Args:
        tiff_path: [H, W, 3] float32 XYZ TIFF 파일 경로.
        voxel_size: FPFH 계산에 사용할 voxel size.

    Returns:
        [1, 33, H, W] float32 tensor.
    """
    data = tifffile.imread(tiff_path).astype(np.float64)  # [H, W, 3]
    H, W = data.shape[:2]
    pts = data.reshape(-1, 3)

    # 유효 포인트 필터
    valid = np.any(pts != 0, axis=1) & ~np.isnan(pts).any(axis=1)
    nonzero_indices = np.nonzero(valid)[0]
    pts_valid = pts[nonzero_indices]

    # Open3D PointCloud 생성 + 법선 추정
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_valid))
    radius_normal = voxel_size * 2
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    # FPFH 계산
    radius_feature = voxel_size * 5
    fpfh_result = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    fpfh = fpfh_result.data.T  # [N_valid, 33]

    # organized grid 복원 (유효하지 않은 점은 0)
    full_fpfh = np.zeros((H * W, fpfh.shape[1]), dtype=np.float32)
    full_fpfh[nonzero_indices] = fpfh.astype(np.float32)
    full_fpfh = full_fpfh.reshape(H, W, 33)  # [H, W, 33]

    # [1, 33, H, W] tensor
    tensor = torch.from_numpy(full_fpfh).permute(2, 0, 1).unsqueeze(0)  # [1, 33, H, W]
    return tensor


def build_fpfh_bank(
    data_path: str,
    category: str,
    bank_dir: str,
    voxel_size: float = 0.05,
    target_size: int = 28,
):
    """
    한 카테고리의 train 샘플에서 FPFH bank 빌드.
    """
    raw_dataset = MVTec3DRawDataset(data_path, category, phase='train')
    N = len(raw_dataset)
    print(f"[FPFH-Bank] {category}: {N} train samples")

    pool = torch.nn.AdaptiveAvgPool2d((target_size, target_size))
    all_patches = []

    for i in range(N):
        tiff_path = raw_dataset.tiff_paths[i]

        # FPFH 추출 [1, 33, H, W]
        fpfh_map = compute_fpfh_from_tiff(tiff_path, voxel_size)

        # downsample to [1, 33, 28, 28]
        fpfh_down = pool(fpfh_map)  # [1, 33, 28, 28]

        # [784, 33]
        patches = fpfh_down.squeeze(0).permute(1, 2, 0).reshape(-1, 33)  # [784, 33]
        all_patches.append(patches)

        if (i + 1) % 10 == 0 or (i + 1) == N:
            print(f"  [{i + 1}/{N}] done")

    # [N_obj, 784, 33]
    fpfh_bank = torch.stack(all_patches, dim=0)  # [N, 784, 33]

    # 저장
    cat_bank_dir = os.path.join(bank_dir, category)
    os.makedirs(cat_bank_dir, exist_ok=True)
    save_path = os.path.join(cat_bank_dir, "fpfh_bank.pth")

    torch.save(
        {
            "fpfh_patches": fpfh_bank.half(),
            "category": category,
        },
        save_path,
    )
    print(f"[FPFH-Bank] Saved: {save_path}  shape={tuple(fpfh_bank.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Build FPFH memory bank from TIFF point clouds")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--item_list', nargs='+', required=True)
    parser.add_argument('--bank_dir', type=str, required=True)
    parser.add_argument('--voxel_size', type=float, default=0.05)
    parser.add_argument('--target_size', type=int, default=28)
    args = parser.parse_args()

    for category in args.item_list:
        build_fpfh_bank(
            data_path=args.data_path,
            category=category,
            bank_dir=args.bank_dir,
            voxel_size=args.voxel_size,
            target_size=args.target_size,
        )

    print("\n[FPFH-Bank] All done.")


if __name__ == '__main__':
    main()
