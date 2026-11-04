"""给仿真场景加"真实场景背景"：围墙/地面的布置与材质绑定计划（纯 Python，可单测）。

为什么需要它
------------
位姿网络在仿真里训练，最怕的是"背景太干净"：灰白地面 + 空墙会让 CNN 把
**背景纹理当成线索**（例如"盒子附近的墙面总是某种颜色"），换到真机上立刻失效。
把工作台四周贴上**真实场景照片**（真实实验室的墙、地面、货架）有两个作用：

1. **外观域随机化**：每回合换一张背景照片，网络学不到固定背景特征；
2. **缩小 sim→real 的外观差**：真机背景就是类似的画面，颜色/纹理分布接近。

实现方式
--------
墙是 ``assets/scene/backdrop_wall.usda`` 里的**薄 Cube 面板**（由 ``tools/make_backdrop_usd.py``
生成，带 24 个 ``primvars:st``，所以能用 ``UsdUVTexture`` 贴图）。本模块只负责算出
"墙在哪、朝向、多大、贴哪张图"，真正的 spawn 在 env cfg 里做：

* 每面墙 spawn 一份实体（``{ENV_REGEX_NS}/Backdrop_PlusX`` 等），所以每个并行环境
  都有自己的背景墙，互不干扰；墙**没有碰撞体**（纯视觉），不影响物理。
* 墙的局部 +Z 是面板法向，故摆放姿态用 :func:`plane_quat_xyzw` 把 +Z 转到"朝向工作区"。
* 照片材质由 ``mdp/events.py: apply_scene_textures`` 在 startup 统一建在 ``/World/Looks``
  下（全局共享），并按 prim 路径绑定各面墙。

用法（真机现场拍照后）::

    # 把照片放到这个目录即可，png/jpg/webp 都行
    openarm_lab/assets/textures/env/*.jpg
    # 或者用环境变量指向别处
    export OPENARM_ENV_TEX_DIR=/data/real_room_photos
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import numpy as np

#: 默认的照片目录（`*.png/jpg/jpeg/webp`，按文件名排序后使用）
DEFAULT_ENV_TEX_DIR = pathlib.Path(__file__).resolve().parents[1] / "assets" / "textures" / "env"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
#: env cfg 里每个背景墙的 prim 路径前缀（spawn 到 /World/<Name>）
SPAWN_ROOT = "/World"

#: 背景墙 USD（``assets/scene/backdrop_wall.usda``，由 tools/make_backdrop_usd.py 生成）
#: 里可见面板的 prim 名
PANEL_SUBPATH = "Panel"

#: GroundPlaneCfg 生成的可见几何固定在这个子路径上（地面材质已经验证过这个路径）
GEOMETRY_SUBPATH = "Environment/Geometry"


@dataclass(frozen=True)
class WallSpec:
    """一面背景墙。``inward_normal`` 指向**工作区**（相机看到的是这一面）。"""

    name: str                     # 也用作 prim 名：/World/<name>
    center: tuple[float, float, float]
    inward_normal: tuple[float, float, float]
    size: tuple[float, float]     # (宽, 高)，米

    @property
    def prim_path(self) -> str:
        return f"{SPAWN_ROOT}/{self.name}"

    @property
    def geometry_path(self) -> str:
        """要绑材质的那个可见几何 prim（``assets/scene/backdrop_wall.usda`` 里的 Panel）。"""
        return f"{self.prim_path}/{PANEL_SUBPATH}"

    @property
    def relative_geometry_path(self) -> str:
        """去掉 ``/World`` 前缀的路径，供"每环境一份"的场景（前面加 ``{ENV_REGEX_NS}``）使用。"""
        return f"{self.name}/{PANEL_SUBPATH}"


# ---------------------------------------------------------------------------
# 姿态：让平面的 +Z 指向指定方向
# ---------------------------------------------------------------------------


def plane_quat_xyzw(inward_normal) -> tuple[float, float, float, float]:
    """返回 (x, y, z, w)：把 ``GroundPlane``（法向 +Z）转到法向 = ``inward_normal``。

    墙都是竖直的（法向水平）。用 ``up=(0,0,1)`` 构造正交基：
    ``x = normalize(up × n)``、``y = n × x``，两者都落在墙面内。
    """
    n = np.asarray(inward_normal, dtype=np.float64)
    ln = np.linalg.norm(n)
    if ln < 1e-9:
        raise ValueError("inward_normal 不能为零向量")
    n = n / ln
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, n))) > 0.99:
        up = np.array([1.0, 0.0, 0.0])      # 水平墙面（地面/天花板）时换参考轴
    x = np.cross(up, n)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    R = np.stack([x, y, n], axis=-1)        # 列 = 新基
    return tuple(float(v) for v in _matrix_to_quat_xyzw(R))


def _matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# 围墙的默认布置
# ---------------------------------------------------------------------------


def default_walls(table_center=(0.30, 0.15), distance: float = 1.20,
                  width: float = 2.60, height: float = 1.90) -> list[WallSpec]:
    """在桌子四周立四面墙（默认离 ``table_center`` 1.2 m 远）。

    为什么要四面**而不是只贴相机背后那一面**：相机/机械臂都会转，腕部相机更是随手臂
    指向各个方向；只补一面会在别的视角里露出"空世界"。四面足够便宜（4 个平面）。

    注意 ``table_center``：并行训练时每个环境只有 ``env_spacing`` 见方，所以通常传
    ``(0.0, 0.0)``（以**环境原点**为中心）并把 ``distance`` 收到 ``env_spacing/2`` 以内，
    这样墙落在自己那格里、不会和邻居的墙交错。
    """
    cx, cy = float(table_center[0]), float(table_center[1])
    zc = height / 2.0

    def along_x(sign: int) -> WallSpec:
        return WallSpec(
            name=f"Backdrop_{'Plus' if sign > 0 else 'Minus'}X",
            center=(cx + sign * distance, cy, zc),
            inward_normal=(-float(sign), 0.0, 0.0),
            size=(width, height),
        )

    def along_y(sign: int) -> WallSpec:
        return WallSpec(
            name=f"Backdrop_{'Plus' if sign > 0 else 'Minus'}Y",
            center=(cx, cy + sign * distance, zc),
            inward_normal=(0.0, -float(sign), 0.0),
            size=(width, height),
        )

    return [along_x(+1), along_x(-1), along_y(+1), along_y(-1)]


# ---------------------------------------------------------------------------
# 真实照片目录
# ---------------------------------------------------------------------------


def env_tex_dir(explicit: str | pathlib.Path | None = None) -> pathlib.Path:
    """照片目录优先级：显式参数 > 环境变量 ``OPENARM_ENV_TEX_DIR`` > 默认目录。"""
    if explicit:
        return pathlib.Path(explicit).expanduser().resolve()
    env = os.environ.get("OPENARM_ENV_TEX_DIR")
    if env:
        return pathlib.Path(env).expanduser().resolve()
    return DEFAULT_ENV_TEX_DIR


def list_env_images(directory: str | pathlib.Path | None = None) -> list[pathlib.Path]:
    """列出可用的真实场景照片（按文件名排序，保证可复现）。"""
    d = env_tex_dir(directory)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def list_wall_images(directory: str | pathlib.Path | None = None) -> list[pathlib.Path]:
    """只列**墙面**照片（排除 ``floor_*``，那些留给地面）。"""
    return [p for p in list_env_images(directory) if not p.name.lower().startswith("floor_")]


def list_floor_images(directory: str | pathlib.Path | None = None) -> list[pathlib.Path]:
    """只列**地面**照片（``floor_*``）。"""
    return [p for p in list_env_images(directory) if p.name.lower().startswith("floor_")]


# ---------------------------------------------------------------------------
# 材质绑定计划
# ---------------------------------------------------------------------------


def material_bindings(walls: list[WallSpec], images: list[pathlib.Path],
                      floor_image: pathlib.Path | None = None,
                      uv_scale: tuple[float, float] = (1.0, 1.0),
                      target_prefix: str = "") -> list[dict]:
    """生成 ``apply_scene_textures`` 用的绑定列表（每面墙一张图，地面可选）。

    图片**按顺序循环分配**，所以 4 面墙 + 3 张照片也能跑。纹理名直接用绝对路径，
    ``events.apply_scene_textures`` 支持"显式路径"（见那边的注释）。

    Args:
        target_prefix: 绑定时加在墙 prim 路径前的命名空间。背景墙是**每环境一份**，
            所以传 ``"{ENV_REGEX_NS}"``（``apply_scene_textures`` 会把它展开成真实
            命名空间并找出其中所有几何 prim）；留空则按 ``/World/<Name>/Panel`` 定位。
        uv_scale: 每面墙重复贴图的次数。默认 (1, 1) = 一张照片铺满整面墙。
    """
    if not images:
        return []
    out: list[dict] = []
    for i, w in enumerate(walls):
        img = images[i % len(images)]
        target = (
            f"{target_prefix}/{w.relative_geometry_path}" if target_prefix else w.geometry_path
        )
        out.append({
            "looks_name": f"{w.name}Mat",
            "texture": str(img),
            "targets": [target],
            "roughness": 0.95,          # 背景墙是哑光的，别反光抢注意力
            "metallic": 0.0,
            "uv_scale": uv_scale,
        })
    if floor_image is not None:
        out.append({
            "looks_name": "FloorRealMat",
            "texture": str(floor_image),
            "targets": [f"/World/GroundPlane/{GEOMETRY_SUBPATH}"],
            "roughness": 0.9,
            "metallic": 0.0,
            "uv_scale": (12.0, 12.0),
        })
    return out


def episode_texture_plan(images: list[pathlib.Path], walls: list[WallSpec],
                         seed: int | None = None) -> dict[str, str]:
    """按回合随机给每面墙挑一张图（域随机化用）。

    返回 ``{looks_name: 图片路径}``；训练/采集脚本可以每个 reset 调一次，
    再让 ``apply_scene_textures`` 重新绑定（材质在 ``/World/Looks`` 下全局共享，
    改纹理路径很便宜，不会复制材质）。
    """
    if not images:
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(images), size=len(walls))
    return {f"{w.name}Mat": str(images[int(j)]) for w, j in zip(walls, idx)}


def summary(walls: list[WallSpec], images: list[pathlib.Path]) -> str:
    """一行摘要，采集/训练启动时打印，确认背景照片真的被用上了。"""
    if not images:
        return "[backdrop] 没有找到真实场景照片 -> 背景保持默认（灰白）。把照片放进 " \
               f"{DEFAULT_ENV_TEX_DIR} 或设置 OPENARM_ENV_TEX_DIR"
    lines = [f"[backdrop] {len(walls)} 面背景墙，{len(images)} 张照片："]
    for i, w in enumerate(walls):
        img = images[i % len(images)]
        lines.append(f"    {w.name:22s} 中心 {w.center} 法向 {w.inward_normal} "
                     f"{w.size[0]:.1f}x{w.size[1]:.1f}m -> {img.name}")
    return "\n".join(lines)
