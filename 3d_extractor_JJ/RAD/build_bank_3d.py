"""
build_bank_3d.py — 3D 포인트 클라우드 기반 다시점 메모리 뱅크 빌더.

build_bank_multilayer.py 를 28개 뷰(RGB 1장 + 렌더링 27장)로 확장한다.
각 뷰를 별도 .pth 파일로 저장하여 추론 시 VRAM ~1.92GB(RAD 와 동일) 유지.

출력 파일:
    bank_dir/
        bank_view_00_rgb.pth   ← view 0 (RGB 원본)
        bank_view_01.pth       ← view 1 (렌더링, -15°, 0°, 0°)
        ...
        bank_view_27.pth       ← view 27 (렌더링, +15°, +15°, +15°)

각 .pth 파일의 구조는 RAD 뱅크와 동일:
    {
        "layers":      list[int],
        "cls_banks":   list[Tensor[N_obj, 768]],   # 레이어별
        "patch_banks": list[Tensor[N_obj, 784, 768]],
    }
"""

import argparse
import multiprocessing
import os
import random
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import get_data_transforms
from dataset_3d import CachedViewDataset, MVTec3DRawDataset
from dinov3.hub.backbones import load_dinov3_model
from render_utils import load_and_render

warnings.filterwarnings("ignore")

# [3D-PATCH] build_bank_multilayer.build_memory_bank_multilayer() 직접 재사용
# 원본 (RAD build_bank_multilayer.py): main() 내에서 호출
# 변경: import 하여 28회 뷰 루프 안에서 재사용
from build_bank_multilayer import build_memory_bank_multilayer


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


def _view_bank_name(v: int) -> str:
    """뷰 인덱스 → .pth 파일 이름."""
    return f"bank_view_{v:02d}_rgb.pth" if v == 0 else f"bank_view_{v:02d}.pth"


# ---------------------------------------------------------------------------
# 병렬 렌더링 워커
# ---------------------------------------------------------------------------

def _render_worker(task):
    """
    [3D-PATCH] 병렬 렌더링 워커 — prerender_train_objects()의 순차 루프를 병렬화
    원본: for i in range(N): load_and_render(...) (순차)
    변경: multiprocessing.Pool로 오브젝트 단위 병렬 렌더링
          spawn 컨텍스트 사용으로 CUDA fork 문제 회피
    """
    tiff_path, rgb_path, view_dirs, image_size, color, stem, num_views = task

    # view 0: RGB 원본 복사
    rgb_dst = os.path.join(view_dirs[0], stem)
    if not os.path.exists(rgb_dst):
        from PIL import Image as _PIL
        _PIL.open(rgb_path).convert('RGB').save(rgb_dst)

    # view 1..27: 필요 시 렌더링
    need_render = any(
        not os.path.exists(os.path.join(view_dirs[v], stem))
        for v in range(1, 1 + num_views)
    )
    if need_render:
        rendered = load_and_render(
            tiff_path, image_size=image_size, color=color, rgb_path=rgb_path,
        )
        for v, pil_img in enumerate(rendered, start=1):
            dst = os.path.join(view_dirs[v], stem)
            if not os.path.exists(dst):
                pil_img.save(dst)


# ---------------------------------------------------------------------------
# 사전 렌더링 (캐시)
# ---------------------------------------------------------------------------

def prerender_train_objects(
    raw_dataset: MVTec3DRawDataset,
    cache_dir: str,
    image_size: int,
    color: str,
    num_views: int = 27,
    render_workers: int = 8,
) -> list:
    """
    훈련 객체 N_obj 개를 렌더링하여 캐시 디렉토리에 저장한다.

    [3D-PATCH] 뷰별 캐시 디렉토리 사전 생성 — RAD에는 없는 렌더링 단계
    원본 (RAD build_bank_multilayer.py): ImageFolder 로 직접 읽기
    변경: TIFF → render_multiview() → 캐시 PNG로 저장 후 CachedViewDataset 으로 읽기

    Returns:
        list[str]: 길이 (1 + num_views) 의 뷰 디렉토리 경로 리스트.
                   view_dirs[0] = RGB 디렉토리,
                   view_dirs[1..27] = 렌더링 디렉토리.
    """
    N = len(raw_dataset)

    # ---- 디렉토리 준비 ----
    # view 0: RGB 원본
    rgb_cache = os.path.join(cache_dir, 'view_00_rgb')
    os.makedirs(rgb_cache, exist_ok=True)
    view_dirs = [rgb_cache]

    # view 1..27: 렌더링 결과
    for v in range(1, 1 + num_views):
        vdir = os.path.join(cache_dir, f'view_{v:02d}')
        os.makedirs(vdir, exist_ok=True)
        view_dirs.append(vdir)

    # ---- 병렬 렌더링 ----
    # [3D-PATCH] 순차 렌더링 → multiprocessing.Pool 병렬화
    # 원본: for i in range(N): load_and_render(...) (순차, ~2h/category)
    # 변경: spawn 컨텍스트 Pool로 오브젝트 단위 병렬 렌더링 (~15min with 16 workers)
    #       spawn 사용 이유: 부모 프로세스의 CUDA 컨텍스트를 자식이 상속하지 않도록
    tasks = [
        (
            raw_dataset.tiff_paths[i],
            raw_dataset.rgb_paths[i],
            view_dirs,
            image_size,
            color,
            f"obj_{i:04d}.png",
            num_views,
        )
        for i in range(N)
    ]

    ctx = multiprocessing.get_context('spawn')
    print(f"  [PreRender] {N} objects  render_workers={render_workers}")
    with ctx.Pool(processes=render_workers) as pool:
        for done, _ in enumerate(pool.imap_unordered(_render_worker, tasks), 1):
            if done % 10 == 0 or done == N:
                print(f"  [PreRender] {done}/{N} objects done")

    return view_dirs


# ---------------------------------------------------------------------------
# 뷰별 뱅크 빌드
# ---------------------------------------------------------------------------

def build_bank_3d(
    raw_dataset: MVTec3DRawDataset,
    encoder,
    device: str,
    bank_dir: str,
    cache_dir: str,
    image_size: int,
    crop_size: int,
    color: str,
    batch_size: int,
    num_workers: int,
    layer_idx_list: list,
    num_views: int = 27,
    skip_existing: bool = True,
    render_workers: int = 8,
):
    """
    28개 뷰 메모리 뱅크를 빌드하여 각각 별도 .pth 파일로 저장한다.

    [3D-PATCH] 28뷰 뱅크 빌드 루프 — RAD 단일 bank 빌드의 확장
    원본 (RAD build_bank_multilayer.py:main): bank 1개 빌드
    변경: for v in range(28) 루프로 28개 뷰 각각에 대해 독립 bank 빌드
    """
    N = len(raw_dataset)
    labels = [0] * N   # 훈련 데이터는 모두 정상(label=0)

    data_transform, _ = get_data_transforms(image_size, crop_size)

    os.makedirs(bank_dir, exist_ok=True)

    print(f"[Build3D] category={raw_dataset.category}  N_obj={N}  "
          f"views={1 + num_views}  device={device}")

    # Step 1: 사전 렌더링
    print("[Build3D] Step 1: Pre-rendering training objects ...")
    view_dirs = prerender_train_objects(
        raw_dataset, cache_dir, image_size, color, num_views, render_workers
    )

    # Step 2: 뷰별 뱅크 빌드
    print("[Build3D] Step 2: Building per-view memory banks ...")
    for v in range(1 + num_views):
        bank_path = os.path.join(bank_dir, _view_bank_name(v))

        if skip_existing and os.path.exists(bank_path):
            print(f"  [View {v:02d}] Skip (already exists): {bank_path}")
            continue

        view_ds = CachedViewDataset(view_dirs[v], labels, data_transform)
        print(f"  [View {v:02d}] Building bank: {len(view_ds)} images  → {bank_path}")

        layers, cls_banks, patch_banks = build_memory_bank_multilayer(
            encoder=encoder,
            dataset=view_ds,
            device=device,
            batch_size=batch_size,
            layer_idx_list=layer_idx_list,
            num_workers=num_workers,
        )

        # [3D-PATCH] float16으로 저장 — 저장 공간 절반 (float32 대비 성능 손실 무시 가능)
        # 원본 (RAD build_bank_multilayer.py:195-202): float32 그대로 저장
        # 변경: .half() 적용 후 저장 → 디스크 ~357GB (float32 ~714GB 대비 절반)
        #       추론 시 .float() upcast 하므로 연산 정밀도는 float32 유지
        torch.save(
            {
                "layers":      layers,
                "cls_banks":   [cb.half() for cb in cls_banks],
                "patch_banks": [pb.half() for pb in patch_banks],
            },
            bank_path,
        )
        print(f"  [View {v:02d}] Saved: {bank_path}  "
              f"cls={tuple(cls_banks[0].shape)}  "
              f"patch={tuple(patch_banks[0].shape)}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build PER-VIEW multi-layer memory banks for 3D datasets (TIFF + RGB)"
    )

    # [3D-PATCH] data_path → 3D-ADAM_anomalib 형식 경로
    # 원본 (RAD build_bank_multilayer.py): MVTec 경로 (train/good/에 직접 PNG)
    # 변경: 3D-ADAM_anomalib 형식 (train/good/rgb/, train/good/xyz/)
    parser.add_argument('--data_path', type=str, required=True,
                        help='3D-ADAM_anomalib 스타일 데이터셋 루트 경로')
    parser.add_argument('--item_list', nargs='+', required=True,
                        help='처리할 카테고리 이름 목록')

    parser.add_argument('--encoder_name', type=str, default='dinov3_vitb16')
    parser.add_argument('--encoder_weight', type=str,
                        default='./dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--crop_size',  type=int, default=448)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--layer_idx_list', nargs='+', type=int, default=[3, 6, 9, 11])

    # [3D-PATCH] 뱅크 저장 경로 (뷰별 파일이므로 디렉토리)
    # 원본 (RAD build_bank_multilayer.py): --bank_path (단일 .pth 파일)
    # 변경: --bank_dir (28개 .pth 파일을 담을 디렉토리)
    parser.add_argument('--bank_dir', type=str, default='./bank_views',
                        help='뷰별 .pth 파일을 저장할 디렉토리')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='렌더링 캐시 디렉토리 (기본: bank_dir/render_cache)')

    # [3D-PATCH] 색상화 방식 — CPMF에 없던 인자
    # 원본 (RAD): 없음 (RGB 이미지를 그대로 사용)
    # 변경: 포인트 클라우드 색상화 방식 선택 (xyz / normal / gray)
    parser.add_argument('--color', type=str, default='rgb',
                        choices=['rgb', 'xyz', 'normal', 'gray'],
                        help='포인트 클라우드 색상화 방식')

    parser.add_argument('--render_workers', type=int, default=8,
                        help='병렬 렌더링 워커 수 (기본: 8, GPU당 16 권장)')
    parser.add_argument('--no_skip', action='store_true',
                        help='이미 존재하는 뱅크도 덮어씀')

    # [3D-PATCH] GPU 선택 인수 — 카테고리별 병렬 실행 시 GPU 지정에 사용
    # 원본: cuda:0 하드코딩
    # 변경: --device 로 cuda:0 / cuda:1 / cpu 선택 가능
    parser.add_argument('--device', type=str, default=None,
                        help='사용할 디바이스 (예: cuda:0, cuda:1, cpu). 미지정 시 자동 선택')

    args = parser.parse_args()

    setup_seed(1)

    if args.device is not None:
        device = args.device
    elif torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(21 / 24, 0)
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f"device: {device}")
    print(f"layers (0-based): {args.layer_idx_list}")
    print(f"color mode: {args.color}")

    cache_dir_base = args.cache_dir or os.path.join(args.bank_dir, 'render_cache')

    # DINOv3 인코더 로드
    encoder = load_dinov3_model(
        args.encoder_name,
        pretrained_weight_path=args.encoder_weight,
    )
    encoder.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    for category in args.item_list:
        print(f"\n{'='*60}")
        print(f"[Build3D] Category: {category}")

        raw_dataset = MVTec3DRawDataset(args.data_path, category, phase='train')
        print(raw_dataset)

        if len(raw_dataset) == 0:
            print(f"  Warning: no training objects found for '{category}', skipping.")
            continue

        # [3D-PATCH] 카테고리별 하위 디렉토리
        # 원본 (RAD): 단일 bank_path
        # 변경: category 이름을 포함한 bank_dir + cache_dir
        bank_dir_cat  = os.path.join(args.bank_dir,  category)
        cache_dir_cat = os.path.join(cache_dir_base, category)

        build_bank_3d(
            raw_dataset    = raw_dataset,
            encoder        = encoder,
            device         = device,
            bank_dir       = bank_dir_cat,
            cache_dir      = cache_dir_cat,
            image_size     = args.image_size,
            crop_size      = args.crop_size,
            color          = args.color,
            batch_size     = args.batch_size,
            num_workers    = args.num_workers,
            layer_idx_list = args.layer_idx_list,
            skip_existing  = not args.no_skip,
            render_workers = args.render_workers,
        )

    print("\n[Build3D] Done.")


if __name__ == '__main__':
    main()
