"""
build_bank_concat.py — RGB + FPFH feature-level concat bank 빌드 (3D-ADS 방식).

기존에 빌드된 RGB bank (view 0)과 FPFH bank을 로드하여
patch 차원에서 concatenate한 뱅크를 생성한다.

DINOv3 patch [N_obj, 784, 768] + FPFH patch [N_obj, 784, 33]
→ concat bank [N_obj, 784, 801]

출력:
    bank_dir/{category}/concat_bank.pth
        {
            "layers":        list[int],
            "cls_banks":     list[Tensor[N_obj, 768]],          # RGB CLS (변경 없음)
            "patch_banks":   list[Tensor[N_obj, 784, 801]],    # RGB+FPFH concat
        }
"""

import argparse
import os

import torch


def build_concat_bank(bank_dir: str, category: str):
    """한 카테고리의 RGB bank (view 0) + FPFH bank → concat bank."""
    cat_dir = os.path.join(bank_dir, category)

    # RGB bank (view 0) 로드
    rgb_path = os.path.join(cat_dir, "bank_view_00_rgb.pth")
    if not os.path.exists(rgb_path):
        print(f"  [Skip] RGB bank not found: {rgb_path}")
        return
    rgb_data = torch.load(rgb_path, map_location='cpu')

    # FPFH bank 로드
    fpfh_path = os.path.join(cat_dir, "fpfh_bank.pth")
    if not os.path.exists(fpfh_path):
        print(f"  [Skip] FPFH bank not found: {fpfh_path}")
        return
    fpfh_data = torch.load(fpfh_path, map_location='cpu')

    layers = rgb_data["layers"]
    cls_banks = rgb_data["cls_banks"]             # list of [N_obj, 768] float16
    rgb_patches = rgb_data["patch_banks"]          # list of [N_obj, 784, 768] float16
    fpfh_patches = fpfh_data["fpfh_patches"]       # [N_obj, 784, 33] float16

    N_rgb = rgb_patches[0].shape[0]
    N_fpfh = fpfh_patches.shape[0]
    if N_rgb != N_fpfh:
        print(f"  [Error] N mismatch: RGB={N_rgb}, FPFH={N_fpfh}")
        return

    # 각 layer의 RGB patch에 FPFH를 concat
    concat_patches = []
    for li, rgb_p in enumerate(rgb_patches):
        # rgb_p: [N_obj, 784, 768] float16
        # fpfh_patches: [N_obj, 784, 33] float16
        concat = torch.cat([rgb_p.float(), fpfh_patches.float()], dim=-1)  # [N_obj, 784, 801]
        concat_patches.append(concat.half())
        print(f"  Layer {layers[li]}: {tuple(rgb_p.shape)} + {tuple(fpfh_patches.shape)} → {tuple(concat.shape)}")

    # 저장
    save_path = os.path.join(cat_dir, "concat_bank.pth")
    torch.save(
        {
            "layers": layers,
            "cls_banks": cls_banks,
            "patch_banks": concat_patches,
        },
        save_path,
    )
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Build RGB+FPFH concat bank (3D-ADS style)")
    parser.add_argument('--bank_dir', type=str, required=True)
    parser.add_argument('--item_list', nargs='+', required=True)
    args = parser.parse_args()

    for category in args.item_list:
        print(f"\n[Concat-Bank] {category}")
        build_concat_bank(args.bank_dir, category)

    print("\n[Concat-Bank] All done.")


if __name__ == '__main__':
    main()
