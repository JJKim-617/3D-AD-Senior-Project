"""
dataset_3d.py — 3D-to-2D 파이프라인용 Dataset 클래스.

[1] CachedViewDataset:
    사전 렌더링된 단일 뷰 이미지 디렉토리 → (img_tensor, label, idx) 반환.
    build_memory_bank_multilayer()의 IndexedDataset과 동일한 인터페이스.

[2] MVTec3DRawDataset:
    3D-ADAM_anomalib 스타일 디렉토리를 파싱하여
    (rgb_path, tiff_path, gt_path_or_None, label, type) 를 수집.

디렉토리 구조 (3D-ADAM_anomalib 예시):
    data_root/
        category/
            train/
                good/
                    rgb/   ← *.png / *.jpg
                    xyz/   ← *.tiff / *.pcd
            test/
                good/
                    rgb/
                    xyz/
                defect_type/
                    rgb/
                    xyz/
                    ground_truth/   ← *.png
"""

import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# CachedViewDataset
# ---------------------------------------------------------------------------

class CachedViewDataset(Dataset):
    """
    사전 렌더링된 단일 뷰 이미지 디렉토리에서 (img_tensor, label, idx)를 반환.

    build_memory_bank_multilayer()에 직접 넘길 수 있는 IndexedDataset 호환 인터페이스.
    (RAD build_bank_multilayer.py의 IndexedDataset 래퍼가 불필요)

    [3D-PATCH] 뷰별 Dataset — RAD ImageFolder 기반 단일 Dataset과 달리 뷰 단위로 분리
    원본 (RAD build_bank_multilayer.py:166): ImageFolder(root=train_path, ...)
    변경: 뷰별로 사전 렌더링된 캐시 디렉토리에서 읽는 Dataset

    디렉토리 구조:
        view_dir/
            obj_0000.png
            obj_0001.png
            ...
    """

    def __init__(self, view_dir: str, labels: list, transform):
        """
        Args:
            view_dir:  해당 뷰의 렌더링 이미지가 저장된 디렉토리 경로.
            labels:    각 이미지의 레이블 list (정수, 길이 N).
            transform: torchvision transforms (RAD get_data_transforms 와 동일).
        """
        self.transform = transform
        self.img_files = sorted(
            glob.glob(os.path.join(view_dir, '*.png')) +
            glob.glob(os.path.join(view_dir, '*.jpg'))
        )
        if len(self.img_files) != len(labels):
            raise ValueError(
                f"img_files({len(self.img_files)}) != labels({len(labels)}) "
                f"in {view_dir}"
            )
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img = Image.open(self.img_files[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx], idx


# ---------------------------------------------------------------------------
# MVTec3DRawDataset
# ---------------------------------------------------------------------------

class MVTec3DRawDataset:
    """
    3D-ADAM_anomalib 스타일 원시 데이터셋 파서.

    각 객체에 대해:
        rgb_path   — RGB 이미지 파일 경로
        tiff_path  — 포인트 클라우드(.tiff/.pcd) 경로
        gt_path    — 결함 마스크 경로 (정상이면 None)
        label      — 0 (정상) / 1 (결함)
        type       — 결함 유형 문자열 ('good' 포함)
    를 수집한다. rad_3d.py 에서 직접 사용된다.
    """

    def __init__(self, data_root: str, category: str, phase: str = 'train'):
        """
        Args:
            data_root: 카테고리 디렉토리들이 있는 루트 경로.
            category:  카테고리 이름 (예: 'bracket_black').
            phase:     'train' 또는 'test'.
        """
        self.data_root = data_root
        self.category  = category
        self.phase     = phase

        (
            self.rgb_paths,
            self.tiff_paths,
            self.gt_paths,
            self.labels,
            self.types,
        ) = self._load()

    # ------------------------------------------------------------------

    def _load(self):
        cat_dir   = os.path.join(self.data_root, self.category)
        phase_dir = os.path.join(cat_dir, self.phase)

        rgb_paths, tiff_paths, gt_paths, labels, types_ = [], [], [], [], []

        for defect_type in sorted(os.listdir(phase_dir)):
            type_dir = os.path.join(phase_dir, defect_type)
            if not os.path.isdir(type_dir):
                continue

            rgb_dir = os.path.join(type_dir, 'rgb')
            xyz_dir = os.path.join(type_dir, 'xyz')

            if not os.path.isdir(rgb_dir):
                continue

            rgb_files = sorted(
                glob.glob(os.path.join(rgb_dir, '*.png')) +
                glob.glob(os.path.join(rgb_dir, '*.jpg'))
            )

            for rgb_f in rgb_files:
                stem = os.path.splitext(os.path.basename(rgb_f))[0]

                # 대응하는 포인트 클라우드 경로 탐색 (.tiff 우선, .pcd 차선)
                tiff_f = None
                if os.path.isdir(xyz_dir):
                    for ext in ('tiff', 'tif', 'pcd'):
                        candidate = os.path.join(xyz_dir, f'{stem}.{ext}')
                        if os.path.exists(candidate):
                            tiff_f = candidate
                            break

                if tiff_f is None:
                    continue   # pointcloud 없는 객체는 건너뜀

                is_anomaly = (defect_type != 'good')
                label = 1 if is_anomaly else 0

                # GT 마스크 탐색 (test 결함 샘플만)
                # MVTec3D-AD 는 gt/, 3D-ADAM_anomalib 는 ground_truth/ 를 사용
                gt_f = None
                if is_anomaly and self.phase == 'test':
                    gt_dir = os.path.join(type_dir, 'gt')
                    for ext in ('png', 'jpg'):
                        candidate = os.path.join(gt_dir, f'{stem}.{ext}')
                        if os.path.exists(candidate):
                            gt_f = candidate
                            break

                rgb_paths.append(rgb_f)
                tiff_paths.append(tiff_f)
                gt_paths.append(gt_f)
                labels.append(label)
                types_.append(defect_type)

        return (
            np.array(rgb_paths),
            np.array(tiff_paths),
            np.array(gt_paths, dtype=object),
            np.array(labels, dtype=np.int64),
            np.array(types_),
        )

    def __len__(self) -> int:
        return len(self.rgb_paths)

    def __repr__(self) -> str:
        n_good = int((self.labels == 0).sum())
        n_bad  = int((self.labels == 1).sum())
        return (
            f"MVTec3DRawDataset(category='{self.category}', phase='{self.phase}', "
            f"normal={n_good}, anomaly={n_bad})"
        )
