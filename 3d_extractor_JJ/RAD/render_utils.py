"""
render_utils.py — CPMF 방식 다시점 렌더러 (3D 포인트 클라우드)

CPMF (https://github.com/caoyunkang/CPMF) 렌더링 로직을 헤드리스 서버 환경에 맞게 적용한다.
주요 변경점:
  1. GUI 기반 o3d.visualization.Visualizer  →  OffscreenRenderer (헤드리스)
  2. image_size = 512  (CPMF 기본값 224 → Bug #4 수정)
  3. 색상화 방식: xyz / normal / gray / rgb 지원

참조:
  CPMF render_utils.py: https://github.com/caoyunkang/CPMF/blob/master/utils/render_utils.py
"""

import math

import numpy as np
import open3d as o3d
from PIL import Image

# ---------------------------------------------------------------------------
# [3D-PATCH] 렌더링 해상도 512 고정 — CPMF 기본값 224에서 변경 (Bug #4)
# 원본 (CPMF render_utils.py): 224×224 (MultiViewRender __init__ 기본값)
# 변경: 512로 높여 RAD 전처리(Resize→512→CenterCrop→448) 시 품질 손실 방지
# ---------------------------------------------------------------------------
DEFAULT_IMAGE_SIZE = 512

# ---------------------------------------------------------------------------
# CPMF 좌표 변환 행렬
# 원본 (CPMF render_utils.py:read_pcd):
#   o3d_pc.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
# 변경: 없음 (동일)
# ---------------------------------------------------------------------------
_COORD_TRANSFORM = np.array(
    [[1,  0,  0,  0],
     [0, -1,  0,  0],
     [0,  0, -1,  0],
     [0,  0,  0,  1]],
    dtype=np.float64,
)

# CPMF 회전 각도: 각 축 3개 → 3×3×3 = 27 뷰
_ANGLES = [0.0, -math.pi / 12, math.pi / 12]   # [0°, -15°, +15°]


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------

def get_viewpoints() -> list:
    """
    27개 (rx, ry, rz) 회전 각도 조합을 라디안으로 반환.

    뷰 0 = (0, 0, 0) = 기준 뷰. CPMF MultiViewRender와 동일한 순서.

    Returns:
        list of (float, float, float): 27개 (rx, ry, rz) 튜플.
    """
    return [(rx, ry, rz) for rx in _ANGLES for ry in _ANGLES for rz in _ANGLES]


def read_tiff_pcd(path: str) -> o3d.geometry.PointCloud:
    """
    TIFF organized point cloud → Open3D PointCloud.

    CPMF의 read_pcd() 로직을 재현하되 tifffile로 직접 읽는다.
    유효하지 않은 포인트(NaN 또는 모두 0)를 제거하고 CPMF 좌표 변환을 적용한다.

    Args:
        path: [H, W, 3] float32 XYZ 가 저장된 .tiff / .tif 파일 경로.

    Returns:
        Open3D PointCloud (CPMF 좌표계 변환 완료).
    """
    import tifffile

    data = tifffile.imread(path).astype(np.float64)  # [H, W, 3]
    pts = data.reshape(-1, 3)

    # 유효 포인트 필터: 전체가 0이거나 NaN인 행 제거
    valid = np.any(pts != 0, axis=1) & ~np.isnan(pts).any(axis=1)
    pts = pts[valid]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    # [3D-PATCH] CPMF 좌표 변환 — read_pcd() 와 동일
    # 원본 (CPMF render_utils.py:read_pcd): o3d_pc.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    # 변경: 없음 (동일한 행렬 적용)
    pcd.transform(_COORD_TRANSFORM)
    return pcd


def read_tiff_pcd_with_rgb(tiff_path: str, rgb_path: str) -> o3d.geometry.PointCloud:
    """
    organized TIFF + RGB 이미지 → 실제 색상이 입혀진 Open3D PointCloud.

    [3D-PATCH] RGB 색상 맵핑 — organized point cloud 픽셀 대응 이용
    원본 (render_utils.py): xyz/normal/gray 만 지원
    변경: TIFF 픽셀 (r,c) ↔ RGB 이미지 픽셀 (r,c) 1:1 대응으로 색상 할당
          동일한 valid 마스크를 XYZ와 RGB에 적용하여 색상 대응 보존

    Args:
        tiff_path: [H, W, 3] float32 XYZ 가 저장된 .tiff / .tif 파일 경로.
        rgb_path:  대응하는 RGB 이미지 경로 (.png / .jpg).

    Returns:
        Open3D PointCloud (RGB 색상 + CPMF 좌표계 변환 완료).
    """
    import tifffile
    from PIL import Image as _PIL

    data = tifffile.imread(tiff_path).astype(np.float64)  # [H, W, 3] XYZ
    rgb  = np.array(_PIL.open(rgb_path).convert('RGB'))   # [H, W, 3] uint8

    # RGB를 TIFF 해상도에 맞게 리사이즈 (해상도가 다를 경우 대비)
    H, W = data.shape[:2]
    if rgb.shape[:2] != (H, W):
        rgb = np.array(_PIL.fromarray(rgb).resize((W, H), _PIL.BILINEAR))

    pts  = data.reshape(-1, 3)
    cols = rgb.reshape(-1, 3).astype(np.float64) / 255.0  # [H*W, 3] 0~1

    # 유효 포인트 필터: 동일한 valid 마스크를 XYZ와 RGB에 동시 적용
    valid = np.any(pts != 0, axis=1) & ~np.isnan(pts).any(axis=1)
    pts  = pts[valid]
    cols = cols[valid]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd.colors = o3d.utility.Vector3dVector(cols)

    # [3D-PATCH] CPMF 좌표 변환 — read_pcd() 와 동일
    pcd.transform(_COORD_TRANSFORM)
    return pcd


def read_pcd_file(path: str) -> o3d.geometry.PointCloud:
    """
    PCD 파일 → Open3D PointCloud (CPMF 좌표 변환 적용).

    Args:
        path: .pcd 파일 경로.

    Returns:
        Open3D PointCloud (CPMF 좌표계 변환 완료).
    """
    pcd = o3d.io.read_point_cloud(path)
    pcd.transform(_COORD_TRANSFORM)
    return pcd


def render_multiview(
    pcd: o3d.geometry.PointCloud,
    image_size: int = DEFAULT_IMAGE_SIZE,
    color: str = 'rgb',
) -> list:
    """
    27개 시점에서 포인트 클라우드를 렌더링한다.

    Args:
        pcd:        Open3D PointCloud (CPMF 좌표 변환이 이미 적용된 상태).
        image_size: 출력 이미지 해상도 (너비 = 높이).
        color:      색상화 방식 — 'rgb', 'xyz', 'normal', 'gray'.
                    'rgb': read_tiff_pcd_with_rgb() 로 미리 색상이 설정된 PCD 사용.

    Returns:
        List[PIL.Image]: 27장 RGB PIL 이미지 (get_viewpoints() 와 동일한 순서).
    """
    # 색상을 한 번만 계산
    pcd_colored = _color_pcd(pcd, color)
    pts_orig = np.asarray(pcd_colored.points).copy()
    colors   = np.asarray(pcd_colored.colors).copy()

    images = []
    for rx, ry, rz in get_viewpoints():
        # [3D-PATCH] CPMF와 동일한 회전 행렬 생성 방식 사용
        # 원본 (CPMF render_utils.py:rotate_render):
        #   R = o3d.geometry.get_rotation_matrix_from_xyz(rotate_angle)
        # 변경: 없음 (동일 API 사용)
        R = o3d.geometry.get_rotation_matrix_from_xyz(np.array([rx, ry, rz]))
        pts_rot = (R @ pts_orig.T).T

        pcd_rot = o3d.geometry.PointCloud()
        pcd_rot.points = o3d.utility.Vector3dVector(pts_rot)
        pcd_rot.colors = o3d.utility.Vector3dVector(colors)

        img_np = _render_single_view(pcd_rot, image_size)
        images.append(Image.fromarray(img_np, mode='RGB'))

    return images   # 27 PIL Images


def load_and_render(
    point_cloud_path: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    color: str = 'rgb',
    rgb_path: str = None,
) -> list:
    """
    TIFF / PCD 파일을 로드하여 27개 시점 이미지를 반환한다.

    Args:
        point_cloud_path: .tiff / .tif / .pcd 파일 경로.
        image_size:       출력 해상도.
        color:            색상화 방식 — 'rgb', 'xyz', 'normal', 'gray'.
        rgb_path:         RGB 이미지 경로 (color='rgb' + TIFF 입력 시 사용).

    Returns:
        List[PIL.Image]: 27장 RGB PIL 이미지.
    """
    ext = point_cloud_path.lower().rsplit('.', 1)[-1]

    # [3D-PATCH] RGB 색상 모드 — organized point cloud 픽셀 대응 이용
    # 원본 (render_utils.py): TIFF 로드 시 XYZ만 읽고 xyz/normal/gray 색상화
    # 변경: color='rgb' + rgb_path 제공 시 read_tiff_pcd_with_rgb() 로 실제 색상 할당
    if color == 'rgb' and rgb_path is not None and ext in ('tiff', 'tif'):
        pcd = read_tiff_pcd_with_rgb(point_cloud_path, rgb_path)
    elif ext in ('tiff', 'tif'):
        pcd = read_tiff_pcd(point_cloud_path)
    elif ext == 'pcd':
        pcd = read_pcd_file(point_cloud_path)
    else:
        raise ValueError(f"Unsupported format: '{ext}'. .tiff / .pcd 만 지원")

    return render_multiview(pcd, image_size=image_size, color=color)


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _color_pcd(
    pcd: o3d.geometry.PointCloud,
    color: str,
) -> o3d.geometry.PointCloud:
    """색상 적용 후 복사본 반환."""
    pcd = o3d.geometry.PointCloud(pcd)   # 얕은 복사
    if color == 'rgb':
        # [3D-PATCH] rgb 모드: read_tiff_pcd_with_rgb() 로 이미 색상이 설정된 PCD pass-through
        # 원본 (render_utils.py): xyz/normal/gray 만 지원
        # 변경: 색상이 미리 할당된 경우 그대로 반환 (추가 색상화 불필요)
        if not pcd.has_colors():
            raise ValueError(
                "color='rgb' 이지만 PCD에 colors가 없습니다. "
                "read_tiff_pcd_with_rgb()로 로드하거나 rgb_path를 전달하세요."
            )
        return pcd
    elif color == 'xyz':
        return _apply_xyz_color(pcd)
    elif color == 'normal':
        return _apply_normal_color(pcd)
    elif color == 'gray':
        return _apply_gray_color(pcd)
    else:
        raise ValueError(f"Unknown color mode '{color}'. Choose from: rgb, xyz, normal, gray")


def _apply_xyz_color(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """XYZ 좌표 → RGB (min-max 정규화)."""
    pts = np.asarray(pcd.points)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    ranges = np.where(maxs - mins > 1e-8, maxs - mins, 1.0)
    colors = np.clip((pts - mins) / ranges, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _apply_normal_color(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """표면 법선 → RGB ([-1,1] → [0,1])."""
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    normals = np.asarray(pcd.normals)
    colors = np.clip((normals + 1.0) / 2.0, 0.0, 1.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _apply_gray_color(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """모든 포인트를 균일한 회색으로 설정."""
    pcd.colors = o3d.utility.Vector3dVector(
        np.full((len(pcd.points), 3), 0.7)
    )
    return pcd


def _render_single_view(
    pcd_colored: o3d.geometry.PointCloud,
    image_size: int,
) -> np.ndarray:
    """
    OffscreenRenderer로 포인트 클라우드 단일 뷰 렌더링.

    [3D-PATCH] Open3D OffscreenRenderer 사용 — CPMF GUI Visualizer 대체
    원본 (CPMF render_utils.py:rotate_render):
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=..., height=..., visible=False)
    변경: OffscreenRenderer → 디스플레이 없이 서버 환경에서도 동작

    Returns:
        [H, W, 3] uint8 RGB 이미지.
    """
    renderer = o3d.visualization.rendering.OffscreenRenderer(image_size, image_size)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 2.0

    renderer.scene.add_geometry("pcd", pcd_colored, mat)

    # Open3D CUDA build 은 setup_camera(fov, center, eye, up) API 만 지원
    bounds = pcd_colored.get_axis_aligned_bounding_box()
    center = bounds.get_center().astype(np.float32)
    extent = np.asarray(bounds.get_extent())
    distance = float(max(extent)) * 2.5
    eye = (center + np.array([0.0, 0.0, distance], dtype=np.float32))
    up  = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    renderer.setup_camera(60.0, center, eye, up)

    img = renderer.render_to_image()
    img_np = np.asarray(img)           # [H, W, 3 or 4] uint8
    if img_np.ndim == 3 and img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]      # RGBA → RGB

    renderer.scene.clear_geometry()
    return img_np.copy()
