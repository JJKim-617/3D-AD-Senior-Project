"""
build_bank_unified.py — MVTec 3D-AD 전체 카테고리를 하나의 메모리 뱅크로 통합.

원본 RAD(build_bank_multilayer.py)와 동일한 방식:
  - 모든 카테고리의 train/good/rgb 이미지를 하나의 데이터셋으로 합침
  - DINOv3로 multi-layer feature 추출
  - 단일 .pth 파일로 저장

추론 시 CLS top-K로 테스트 이미지와 가장 유사한 K개 이미지를 선별.

출력:
    bank_path (단일 .pth):
        {
            "layers":      list[int],
            "cls_banks":   list[Tensor[N_total, 768]],
            "patch_banks": list[Tensor[N_total, 784, 768]],
        }
    N_total = 전체 카테고리 train 이미지 합계 (~2656)
"""

import argparse
import glob
import os
import random
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from dataset import get_data_transforms
from dinov3.hub.backbones import load_dinov3_model
from build_bank_multilayer import build_memory_bank_multilayer

warnings.filterwarnings("ignore")


def setup_seed(seed: int = 1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class UnifiedRGBDataset(Dataset):
    """MVTec 3D-AD 전체 카테고리의 train RGB 이미지를 하나의 Dataset으로."""

    def __init__(self, data_path, item_list, transform):
        self.transform = transform
        self.img_paths = []
        self.labels = []  # category index

        for cat_idx, category in enumerate(item_list):
            rgb_dir = os.path.join(data_path, category, 'train', 'good', 'rgb')
            files = sorted(
                glob.glob(os.path.join(rgb_dir, '*.png')) +
                glob.glob(os.path.join(rgb_dir, '*.jpg'))
            )
            self.img_paths.extend(files)
            self.labels.extend([cat_idx] * len(files))
            print(f"  {category}: {len(files)} images")

        print(f"  Total: {len(self.img_paths)} images")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img, self.labels[idx], idx


def main():
    parser = argparse.ArgumentParser(
        description="Build unified multi-layer memory bank for MVTec 3D-AD (all categories → 1 bank)"
    )
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--item_list', nargs='+', default=[
        'bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
        'foam', 'peach', 'potato', 'rope', 'tire'
    ])
    parser.add_argument('--encoder_name', type=str, default='dinov3_vitb16')
    parser.add_argument('--encoder_weight', type=str,
                        default='./dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--layer_idx_list', nargs='+', type=int, default=[3, 6, 9, 11])
    parser.add_argument('--bank_path', type=str,
                        default='./bank/mvtec3d_unified_dinov3_vitb16_multilayer36911_448_bank.pth')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()
    setup_seed(1)

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f"device: {device}")
    print(f"layers: {args.layer_idx_list}")

    data_transform, _ = get_data_transforms(args.image_size, args.crop_size)

    print("\n[Unified Bank] Collecting train images ...")
    dataset = UnifiedRGBDataset(args.data_path, args.item_list, data_transform)

    encoder = load_dinov3_model(
        args.encoder_name,
        pretrained_weight_path=args.encoder_weight,
    )
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    print("\n[Unified Bank] Building multi-layer features ...")
    layers, cls_banks, patch_banks = build_memory_bank_multilayer(
        encoder=encoder,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        layer_idx_list=args.layer_idx_list,
        num_workers=args.num_workers,
    )

    for li, layer in enumerate(layers):
        print(f"  layer={layer}  cls={tuple(cls_banks[li].shape)}  "
              f"patch={tuple(patch_banks[li].shape)}")

    os.makedirs(os.path.dirname(args.bank_path), exist_ok=True)
    torch.save(
        {
            "layers": layers,
            "cls_banks": cls_banks,
            "patch_banks": patch_banks,
        },
        args.bank_path,
    )
    print(f"\n[Unified Bank] Saved → {args.bank_path}")


if __name__ == '__main__':
    main()
