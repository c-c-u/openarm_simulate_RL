# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""用**状态策略**采集"图像 + 盒子位姿真值"数据集，供感知 CNN 训练。

为什么这样做
------------
状态策略（``models/openarm_flip_state62_best.pt``，62 维观测）已经能稳定翻转盒子
（实测 88.8%）。它的观测里 ``parcel_pos / parcel_quat / label_normal`` 这三段正是
真机上最难获得、而仿真中"免费且完美"的量。因此可以：

1. 让状态策略在仿真里跑（确定性推理）；
2. 每步同时记录 **相机图像** 和 **盒子 6-DoF 真值**；
3. 在另一台机器上监督训练 ``CNN: image -> (pos, quat / label_normal)``；
4. 部署时用 CNN 的输出替换状态向量的那三段，接回同一个状态策略即可。

本脚本只负责"采集与落盘"，不做训练。

输出目录结构
------------
::

    <out>/<run_name>/
        images/<cam>/000000.png          8-bit RGB，320x240
        depth/<cam>/000000.png           16-bit PNG，单位毫米（毫米！）
        frames.npz                       逐帧真值，见下
        dataset_info.json                schema / 相机内外参 / 观测排布（可移植）
        previews/check_*.png             带投影十字准星的可视化，用于人眼校验对齐
        checkpoint.pt                    本次采集用的策略（便于溯源/复现）

帧号全局递增（跨 env 连续），文件名即帧号；``dataset_info.json`` 里有完整说明。

为什么相机是 320x240（而不是策略用的 96x72）
--------------------------------------------
``OpenArmFlipStateEnvCfg(enable_cameras=True)`` 时 ``enable_image_obs=False``，
``__post_init__`` 会走 ``(CAM_WIDTH, CAM_HEIGHT) = (320, 240)`` 这支分支——采集
用高分辨率，将来要下采样随时可以；策略那条 96x72 的路不用动。

相机的坑（重要）
----------------
* ``AppLauncher(enable_cameras=True)`` 与 ``env_cfg.enable_cameras=True`` **必须同时**
  打开，否则 RTX 不渲染 / 场景里没有相机传感器。
* ``enable_cameras`` 是**构造期**参数，构造后改无效（``__post_init__`` 已经跑过）。
* 腕部相机当前朝下看指尖（link7 局部 +X 15 cm），**视野里常常没有盒子**（被夹爪
  占据）。脚本会逐帧统计每台相机里盒子的投影像素与可见性，采完可以据此判断哪台
  相机真正可用。前视相机是唯一"保证看得到盒子"的相机。

用法
----
::

    # 默认规格（单回合、最多 100 帧、达标后留 5 帧）：
    $ISAAC_ROOT/python.sh scripts/collect_vision_data.py
    $ISAAC_ROOT/python.sh scripts/collect_vision_data.py --frames 2000 --num_envs 16 \
        --out /home/blanc/isaacsim/perception_data --run_name full_run

默认采集规格（``--episodes 1 --frames 100 --window_steps 100 --after_success 15``）
-------------------------------------------------------------------------------
单回合、**最多 100 帧**；达标发生在第 S 帧时，再录 ``after_success``（默认 **15**）帧就结束
（录到第 min(S+15, 上限) 帧）。批量采集常用 ``--window_steps 200``：此时 S ≤ 185 都能录满
15 帧尾帧，S > 185 则由 200 帧上限截断。
把 ``--after_success`` 调大、或 ``--episodes`` 调成 >1（多回合连续录）即可换规格。

想得到"一条连续长片段"：``--episodes 0 --frames 100`` —— 每个 env 达标（或到步数上限）后
**自动重置并接着录下一回合**，直到帧数满；``frames.npz`` 里的 ``episode_index`` 标明每帧属于
第几回合，方便按回合切分训练/验证集。采集结束会自动调用 ``scripts/make_preview_video.py``
生成预览视频（WebP + GIF，叠加真值框、label_up 与 SUCCESS 标记）。

注意：同一时间只能跑一个 Isaac Sim 实例（12 GB 显存）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import shutil
import sys
import time

# ---------------------------------------------------------------------------
# Isaac Sim 的 pip_prebundle 前置（**必须在 import isaaclab / numpy / torch 之前**）
# ---------------------------------------------------------------------------
# isaaclab 的 experience 文件（apps/isaaclab.python.headless.kit）显式排除了
# ``omni.isaac.ml_archive`` 与 ``omni.kit.pip_archive`` 两个扩展，因此 torch 并不在
# python.sh 自动挂的 PYTHONPATH 上——必须像 scripts/eval_model.py、camera_check.py、
# smoke_test.py 一样手工把 prebundle 插到 sys.path 最前面，否则 AppLauncher 启动过程中
# 的 ``import torch`` 会 ModuleNotFoundError。
_ISAAC_ROOT = pathlib.Path(
    os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64")
)
for _rel in (
    "extsDeprecated/omni.isaac.ml_archive/pip_prebundle",  # torch / torchvision
    "exts/isaacsim.pip.newton/pip_prebundle",
):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

# ``import openarm_lab`` 需要仓库的父目录（/home/blanc/isaacsim）在 sys.path 上：
# 用 python.sh 跑脚本时 sys.path[0] 是 scripts/ 而不是项目根，故显式插入。
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# CLI（必须在 AppLauncher 之前解析完）
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT = _REPO_ROOT / "models" / "openarm_flip_state62_best.pt"
_DEFAULT_OUT = pathlib.Path("/home/blanc/isaacsim/perception_data")

parser = argparse.ArgumentParser(description="用状态策略采集视觉-位姿数据集（不训练）")
parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT),
                    help="状态策略 checkpoint（默认 62 维 mark 过的最优模型）")
parser.add_argument("--out", type=str, default=str(_DEFAULT_OUT), help="数据集根目录")
parser.add_argument("--run_name", type=str, default=None,
                    help="本次采集的子目录名；默认 sample<frames>_<时间戳>")
parser.add_argument("--frames", type=int, default=100,
                    help="总帧数上限（默认 100）。与 --after_success 联合决定何时结束："
                         "达标发生在第 S 帧时，录到第 min(S + after_success, frames) 帧为止")
parser.add_argument("--num_envs", type=int, default=1,
                    help="并行 env 数；>1 时每步记录所有 env（初始条件更丰富）")
parser.add_argument("--warmup", type=int, default=12,
                    help="开始记录前的空动作热身步数：让 RTX 降噪收敛（改光照后必须热身）")
parser.add_argument("--cameras", type=str, default="front,wrist_left,wrist_right",
                    help="要采集的相机，逗号分隔；可用 front / wrist_left / wrist_right")
parser.add_argument("--no_depth", action="store_true", help="不保存深度图（默认保存）")
parser.add_argument("--preview", type=int, default=6, help="保存多少张带准星的可视化图")
parser.add_argument("--device", type=str, default="cuda:0")
# --- 取景覆盖（默认用 env cfg 里的值；仅在需要实验新取景时传）---
parser.add_argument("--cam_focal", type=float, default=None,
                    help="前视相机焦距 mm（默认用 env cfg 的 FRONT_CAM_FOCAL=12）")
parser.add_argument("--cam_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="前视相机安装位置（env-local，米）")
parser.add_argument("--cam_target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="前视相机注视点（env-local，米）")
parser.add_argument("--cam_res", type=int, nargs=2, default=None, metavar=("W", "H"),
                    help="前视相机分辨率（默认 640x480；降到 320x240 只降像素密度不缩视野）")
# --- 采集窗口（强烈建议开；实测能把取景覆盖率从 85% 提到 97.5%）---
parser.add_argument("--window_steps", type=int, default=100,
                    help="每个 env 每个回合最多记录多少步（默认 100；0 = 不限，按 --frames 记满为止）。"
                         "实测『达标前 + 前 60 步』这段的取景覆盖率最高（97.5% vs 整回合 85%）")
parser.add_argument("--after_success", type=int, default=15,
                    help="盒子首次达标(cos15°)后再多记几帧就停该 env（默认 15：留足「已完成」的样子）。"
                         "0 = 达标即停。含义：达标发生在第 S 帧 -> 录到第 min(S+after_success, 上限) 帧结束；"
                         "例如 --window_steps 200 + 默认 15：S<=185 都能录满 15 帧尾帧，S>185 则被 200 截断")
parser.add_argument("--episodes", type=int, default=1,
                    help="每个 env 录几个回合。1 = 单回合（默认，录完这一回合就结束）。"
                         ">1 = 达标/到步数上限后**自动重置该 env 并接着录下一回合**，"
                         "得到一条跨多回合的连续片段；0 = 不设上限，一直录到 --frames 满")
parser.add_argument("--seed", type=int, default=None, help="设置后固定 python/numpy/torch 随机种子")
parser.add_argument("--no_preview_video", action="store_true",
                    help="不生成预览视频（默认生成 WebP + GIF，由 scripts/make_preview_video.py 合成）")
parser.add_argument("--no_rerecord", action="store_true",
                    help="关闭『录制中发生环境自动 reset（如盒子被推下桌）就丢弃本回合重录』（默认开启）")
parser.add_argument("--max_retries", type=int, default=5,
                    help="同一回合最多重录几次；超过就放弃该回合（记 0 帧），保证数据里不混入坏回合")
parser.add_argument("--require_cam_visible_at_start", type=str, default="",
                    help="要求这些相机在**每回合第一帧**就能看见盒子（逗号分隔，例如 "
                         "front,wrist_left,wrist_right,top）。不满足就打印警告并计入统计——"
                         "腕部相机的需求是『初始必须看得见、过程中不一定』，这条用于当场发现"
                         "安装/朝向没调好，而不是等训完才发现")
parser.add_argument("--min_start_visible", type=float, default=0.5,
                    help="『初始看得见』的判据：盒子轮廓在画面内的占比 >= 该值")
parser.add_argument("--drop_if_invisible_frames", type=int, default=20,
                    help="盒子**完全离开画面**连续超过这么多帧，就丢弃当前回合并重录（0=关闭）。"
                         "这类帧对位姿网络是废数据，且通常意味着该回合要失败；"
                         "注意遮挡（盒子被手臂挡住）不算离开画面，不会被误判")
parser.add_argument("--max_occlusion", type=float, default=0.5,
                    help="**逐帧剔除**：盒子被遮挡（手臂/夹爪挡在前面）的比例超过此值就不保存该帧（默认 0.5=50%%）。"
                         "1.0 = 不剔除。需要深度图，故与 --no_depth 互斥")
parser.add_argument("--visible_eps", type=float, default=0.02,
                    help="判定「完全离开画面」的可见面积占比阈值（默认 0.02，即几乎不可见）")
parser.add_argument("--debug_force_reset_at", type=int, default=0,
                    help="【仅用于自测】在第 N 步人为制造一次回合终止，用来验证重录逻辑；0=关闭")
parser.add_argument("--video_episode", type=int, default=None,
                    help="预览视频只取第几个回合（0 起）；不传则把所有帧串成一个长视频。"
                         "多回合采集时建议指定，否则影片会很长（WebP/GIF 体积也大）")
parser.add_argument("--video_fps", type=float, default=15.0,
                    help="预览视频帧率；策略是 30 Hz，默认 15 = 慢放 2 倍便于看清")
parser.add_argument("--no_copy_checkpoint", action="store_true",
                    help="不把 checkpoint 复制进数据集（默认复制，便于拷到别的机器）")
args = parser.parse_args()

# 相机名 -> 场景里的传感器名
_CAM_SENSOR = {
    "front": "front_cam",
    "wrist_left": "wrist_left_cam",
    "wrist_right": "wrist_right_cam",
}
camera_keys = [c.strip() for c in args.cameras.split(",") if c.strip()]
for _k in camera_keys:
    if _k not in _CAM_SENSOR:
        raise SystemExit(f"[COLLECT] 未知相机 '{_k}'，可选：{sorted(_CAM_SENSOR)}")

# ---------------------------------------------------------------------------
# 启动 Isaac Sim（采集必须开相机）
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": True})
simulation_app = app_launcher.app

# 注意：numpy / torch / PIL 必须**在 AppLauncher 之后**导入 —— 它们由 Isaac Sim 的
# pip_prebundle 提供（App 启动时才会挂到 sys.path 上），提前 import 会 ModuleNotFoundError。
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import openarm_lab  # noqa: F401,E402
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab.utils.string import string_to_callable  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from openarm_lab.openarm_flip.checkpoint_utils import make_env_cfg  # noqa: E402
from openarm_lab.openarm_flip.openarm_flip_env_cfg import (  # noqa: E402
    CAM_DATA_TYPES,
    CAM_HEIGHT,
    CAM_WIDTH,
    _camera_look_at_quat_xyzw,
    CONTROLLED_JOINTS,
    FRONT_CAM_POS,
    FRONT_CAM_ROT_XYZW,
    PARCEL_SIZE,
    TABLE_POS,
    TABLE_SIZE,
    TABLE_TOP,
    WRIST_CAM_LOCAL,
)

TASK = "OpenArm-Flip-State-v0"
UP_COS = math.cos(math.radians(15))  # "标签朝上"判定阈值，与训练/评估一致
PARCEL_SIZE_HALF = tuple(s / 2.0 for s in PARCEL_SIZE)  # 盒子半长 (x, y, z)，用于角点投影
DROP_TOL = 0.08  # 与 parcel_dropped / parcel_on_table 一致的垂直容差（m）
WRIST_BODY_NAMES = {"wrist_left": "openarm_left_link7", "wrist_right": "openarm_right_link7"}


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def sha256(path: pathlib.Path, chunk: int = 1 << 20) -> str:
    """文件 sha256（前 16 位十六进制即可满足溯源需求，这里返回完整值）。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def to_numpy(x) -> np.ndarray:
    """ProxyArray / torch.Tensor / ndarray 统统转成 numpy（CPU）。"""
    if hasattr(x, "torch"):
        x = x.torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def label_normal_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """世界系下盒子局部 +Z（标签法向）单位向量，与 mdp.observations 完全一致。

    与 ``observation/parcel_label_normal_w`` 用同一组公式，保证真值和策略看到的一致。
    """
    x, y, z, w = quat.unbind(dim=-1)
    nx = 2.0 * (x * z + w * y)
    ny = 2.0 * (y * z - w * x)
    nz = 1.0 - 2.0 * (x * x + y * y)
    return torch.stack([nx, ny, nz], dim=-1)


def save_rgb(path: pathlib.Path, arr: np.ndarray) -> None:
    """保存 8-bit RGB（H,W,3）。"""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def save_depth_mm(path: pathlib.Path, arr: np.ndarray) -> tuple[int, int]:
    """深度存成 16-bit PNG，**单位毫米**。返回 (合法像素数, 被裁剪像素数)。

    ``distance_to_image_plane`` 单位是米、float32；转毫米后 uint16 上限 65.535 m，
    对本场景（5 m 裁剪面）绰绰有余。无效值（0 / NaN / inf）写成 0 = "无回波"。
    """
    d = np.asarray(arr, dtype=np.float32)
    invalid = ~np.isfinite(d) | (d <= 1e-6)
    mm = np.where(invalid, 0.0, np.clip(d, 0.0, 65.535) * 1000.0)
    clipped = int((np.where(invalid, 0.0, d) > 65.535).sum())
    Image.fromarray(mm.astype(np.uint16), mode="I;16").save(path)
    return int((~invalid).sum()), clipped


def project_to_pixels(
    points_w: torch.Tensor, cam_pos_w: torch.Tensor, cam_quat_w_ros: torch.Tensor, K: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """把世界系点投到像素系。

    用 ``quat_w_ros``（相机 x 右 / y 下 / z 前）配合 ``intrinsic_matrices``
    （fx, 0, cx; 0, fy, cy）——这是针孔模型的标准约定。

    Returns:
        (uv, z_cam)：uv 形状 (N, 2) 像素坐标；z_cam 形状 (N,) 相机系前向距离（米）。
    """
    p_cam = quat_apply_inverse(cam_quat_w_ros, points_w - cam_pos_w)  # (N, 3)
    z = p_cam[:, 2]
    fx, fy = K[:, 0, 0], K[:, 1, 1]
    cx, cy = K[:, 0, 2], K[:, 1, 2]
    safe_z = torch.where(z.abs() < 1e-6, torch.full_like(z, 1e-6), z)
    u = fx * p_cam[:, 0] / safe_z + cx
    v = fy * p_cam[:, 1] / safe_z + cy
    return torch.stack([u, v], dim=-1), z


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """(N, 4) xyzw 四元数 -> (N, 3, 3) 旋转矩阵（避免依赖 isaaclab 的批处理工具）。"""
    x, y, z, w = quat.unbind(dim=-1)
    row0 = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], dim=-1)
    row1 = torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], dim=-1)
    row2 = torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _parcel_corners_w(
    parcel_pos_w: torch.Tensor, parcel_quat_w: torch.Tensor, half_extents: tuple[float, float, float]
) -> torch.Tensor:
    """盒子 8 个角点的世界坐标 (N, 8, 3)。

    角点顺序：``sx`` 最慢、``sy`` 次之、``sz`` 最快（每个分量取 -1/+1）。
    """
    dev = parcel_pos_w.device
    signs = torch.tensor(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        device=dev,
        dtype=torch.float32,
    )
    local = signs * torch.tensor(half_extents, device=dev, dtype=torch.float32)  # (8, 3)
    R = quat_to_matrix(parcel_quat_w)  # (N, 3, 3)
    return parcel_pos_w.unsqueeze(1) + torch.einsum("nij,kj->nki", R, local)


def parcel_visible_fraction(
    parcel_pos_w: torch.Tensor,
    parcel_quat_w: torch.Tensor,
    cam_pos_w: torch.Tensor,
    cam_quat_w_ros: torch.Tensor,
    K: torch.Tensor,
    half_extents: tuple[float, float, float],
    img_wh: tuple[int, int],
) -> torch.Tensor:
    """盒子轮廓落在画面内的面积占比 (N,)，1.0 = 完整可见。

    这是数据集**可用性**的核心指标：盒子被画面边缘截断时，位姿 CNN 拿不到完整
    证据。做法是把盒子的 8 个角点投到像素系，比较"完整包围盒面积"与"被画面裁剪
    后的面积"。不涉及遮挡（被机械臂挡住要另算）。
    """
    corners_w = _parcel_corners_w(parcel_pos_w, parcel_quat_w, half_extents)  # (N, 8, 3)
    n = corners_w.shape[0]
    uv, _ = project_to_pixels(
        corners_w.reshape(-1, 3),
        cam_pos_w.repeat(8, 1),
        cam_quat_w_ros.repeat(8, 1),
        K.repeat(8, 1, 1),
    )
    uv = uv.reshape(n, 8, 2)
    w, h = float(img_wh[0]), float(img_wh[1])
    u, v = uv[..., 0], uv[..., 1]
    full = (u.max(dim=1).values - u.min(dim=1).values) * (v.max(dim=1).values - v.min(dim=1).values)
    cu0, cu1 = u.min(dim=1).values.clamp(min=0.0), u.max(dim=1).values.clamp(max=w)
    cv0, cv1 = v.min(dim=1).values.clamp(min=0.0), v.max(dim=1).values.clamp(max=h)
    clipped = (cu1 - cu0).clamp(min=0.0) * (cv1 - cv0).clamp(min=0.0)
    return torch.where(full > 1e-6, (clipped / full).clamp(0.0, 1.0), torch.zeros_like(full))


def quat_z_to(n: torch.Tensor) -> torch.Tensor:
    """把世界 +Z 转到 ``n`` 的**最短弧**四元数（即"面内自转为 0"的参考朝向）。"""
    nx, ny, nz = n.unbind(-1)
    q = torch.stack([-ny, nx, torch.zeros_like(nx), 1.0 + nz], dim=-1)
    norm = q.norm(dim=-1, keepdim=True)
    degen = norm.squeeze(-1) < 1e-6  # n ≈ -Z：最短弧不唯一，约定绕 X 轴 180°
    q = q / norm.clamp_min(1e-8)
    if degen.any():
        q = torch.where(degen.unsqueeze(-1),
                        torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype).expand_as(q), q)
    return q


def twist_deg_about_label(quat: torch.Tensor, local_x_w: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """盒子绕自身标签轴的自转角（度，(-180,180]）。

    定义：以"把世界 +Z 用最短弧转到标签法向 n"得到的参考朝向为 0 度，量出盒子实际
    局部 +X 相对该参考 +X 绕 n 转过的角度。这一维正是仅凭 label_normal 无法确定的自由度，
    也是盒子的 90° 四重对称所在（正方形底面）。
    """
    x_ref = quat_to_matrix(quat_z_to(n))[..., 0]        # 参考朝向的局部 +X 在世界系
    cross = torch.cross(x_ref, local_x_w, dim=-1)
    sin = (cross * n).sum(dim=-1)
    cos = (x_ref * local_x_w).sum(dim=-1)
    return torch.rad2deg(torch.atan2(sin, cos))


def silhouette_corner_uv(
    parcel_pos_w: torch.Tensor, parcel_quat_w: torch.Tensor,
    cam_pos_w: torch.Tensor, cam_quat_w_ros: torch.Tensor, K: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """盒子 8 角点在相机像素系的坐标 (N,8,2) 与相机系深度 (N,8)。"""
    corners_w = _parcel_corners_w(parcel_pos_w, parcel_quat_w, PARCEL_SIZE_HALF)
    n = corners_w.shape[0]
    uv, z = project_to_pixels(
        corners_w.reshape(-1, 3), cam_pos_w.repeat(8, 1), cam_quat_w_ros.repeat(8, 1), K.repeat(8, 1, 1)
    )
    return uv.reshape(n, 8, 2), z.reshape(n, 8)


def occlusion_ratio_np(
    depth_m: np.ndarray, uv8: np.ndarray, z_center: float, half_diag: float,
    margin: float = 0.01,
) -> float:
    """遮挡比例：预测轮廓内"比盒子近表面还近"的像素占比（nan = 测不了）。

    与 ``scripts/verify_vision_data.py`` 的判据 4 **同一公式**，所以两边数字可比。
    盒子的最靠近相机的表面不会比 ``z_center - half_diag`` 更近；更近的像素说明
    有东西（手臂/夹爪）挡在盒子前面。
    """
    h, w = depth_m.shape
    u0, u1 = int(max(uv8[:, 0].min(), 0)), int(min(uv8[:, 0].max(), w - 1))
    v0, v1 = int(max(uv8[:, 1].min(), 0)), int(min(uv8[:, 1].max(), h - 1))
    if u1 - u0 < 2 or v1 - v0 < 2:
        return float("nan")
    patch = depth_m[v0:v1 + 1, u0:u1 + 1]
    valid = patch > 0
    if valid.sum() < 10:
        return float("nan")
    nearer = valid & (patch < z_center - half_diag - margin)
    return float(nearer.sum()) / float(valid.sum())


def draw_preview(
    rgb: np.ndarray, uv: np.ndarray, label_up: float, cam: str, frame: int, visible: float
) -> Image.Image:
    """在图上画盒子中心的准星 + 8 角点投影轮廓，用于人眼验证"图像与真值对齐"。

    关掉预览图的用处：轮廓与画面里盒子的实际边缘应重合；不重合就是内外参/约定有问题。
    """
    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB").copy()
    d = ImageDraw.Draw(img)
    w, h = img.width, img.height

    def on_screen(u: float, v: float) -> tuple[float, float]:
        return float(u), float(v)

    pts = [on_screen(u, v) for u, v in uv]
    if all(np.isfinite([p[0], p[1]]).all() for p in pts):
        # 12 条棱（角点顺序：sx 慢、sy 中、sz 快）
        edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                diff = sum(1 for a, b in zip(
                    [(i >> 2) & 1, (i >> 1) & 1, i & 1], [(j >> 2) & 1, (j >> 1) & 1, j & 1]
                ) if a != b)
                if diff == 1:
                    edges.append((pts[i], pts[j]))
        for a, b in edges:
            d.line([a, b], fill=(0, 255, 0), width=1)
        cu = sum(p[0] for p in pts) / 8.0
        cv = sum(p[1] for p in pts) / 8.0
        if 0 <= cu < w and 0 <= cv < h:
            d.line([(cu - 7, cv), (cu + 7, cv)], fill=(255, 0, 0), width=2)
            d.line([(cu, cv - 7), (cu, cv + 7)], fill=(255, 0, 0), width=2)
    d.rectangle([0, 0, w - 1, 13], fill=(0, 0, 0))
    d.text((3, 2), f"{cam} #{frame} up={label_up:+.2f} vis={visible * 100:.0f}%", fill=(255, 255, 0))
    return img


def policy_obs(obs) -> torch.Tensor:
    """从 rsl-rl 的 ``TensorDict`` 观测里取出 ``policy`` 组（纯 62 维张量）。

    ``RslRlVecEnvWrapper`` 的 reset/step 返回的是批大小 [num_envs] 的 ``TensorDict``——
    直接对整体取 ``.shape[-1]`` 只会得到 num_envs（这就是"观测维度 = 1"那个坑）。
    """
    if hasattr(obs, "keys"):
        return obs["policy"] if "policy" in obs.keys() else next(iter(obs.values()))
    return obs


def main() -> None:
    import importlib.metadata as metadata

    if args.seed is not None:
        import random

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.exists():
        raise SystemExit(f"[COLLECT] checkpoint 不存在：{checkpoint}")

    run_name = args.run_name or f"sample{args.frames}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = pathlib.Path(args.out).expanduser().resolve() / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"[COLLECT] 目录已存在且非空，换个 --run_name：{run_dir}")

    # ------------------------------------------------------------------ 环境
    from openarm_lab.openarm_flip.openarm_flip_env_cfg import OpenArmFlipStateEnvCfg  # noqa: F401

    spec = gym.spec(TASK)
    # enable_cameras=True 必须在构造期传入；lean_cameras=False 才保留腕部相机与 depth。
    # --cameras front 时用 lean_cameras=True 让腕部相机**根本不创建**（省 2/3 渲染量）；
    # 但 lean 模式会顺手把前视相机的 depth 也去掉，而深度图是校验判据（凸起点测/遮挡）
    # 的必要输入，所以随后显式补回 depth。
    _front_only = camera_keys == ["front"]
    env_cfg = make_env_cfg(
        string_to_callable(spec.kwargs["env_cfg_entry_point"]),
        checkpoint,
        enable_cameras=True,
        lean_cameras=_front_only,
    )
    if _front_only and not args.no_depth:
        env_cfg.scene.front_cam.data_types = list(CAM_DATA_TYPES)
    if args.max_occlusion < 1.0 and args.no_depth:
        print("[COLLECT] 警告：--max_occlusion 需要深度图，但已指定 --no_depth -> 遮挡剔除将不生效", flush=True)
    agent_cfg = string_to_callable(spec.kwargs["rsl_rl_cfg_entry_point"])()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, str(metadata.version("rsl-rl-lib")))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    # 取景覆盖：CameraCfg 在 gym.make 时才被用来创建传感器，所以此时改仍然生效。
    # 焦距只影响视场，分辨率只影响像素密度 —— 想"既看得全又看得清"要同时调这两个。
    front = env_cfg.scene.front_cam
    if front is None:
        raise SystemExit("[COLLECT] 场景里没有前视相机（front_cam 为 None）")
    if args.cam_focal is not None:
        front.spawn.focal_length = float(args.cam_focal)
    if args.cam_pos is not None:
        front.offset.pos = tuple(float(v) for v in args.cam_pos)
    if args.cam_target is not None:
        front.offset.rot = _camera_look_at_quat_xyzw(tuple(front.offset.pos), tuple(args.cam_target))
    if args.cam_res is not None:
        front.width, front.height = int(args.cam_res[0]), int(args.cam_res[1])

    # 把**实际生效**的相机参数记下来（不是模块常量 —— CLI 可能覆盖过），写进 dataset_info
    def _cam_effective(key: str) -> dict:
        cfg = env_cfg.scene.front_cam if key == "front" else getattr(env_cfg.scene, _CAM_SENSOR[key])
        focal = float(cfg.spawn.focal_length)
        aperture = float(cfg.spawn.horizontal_aperture)
        fov_h = 2.0 * math.degrees(math.atan(aperture / (2.0 * focal)))
        fov_v = 2.0 * math.degrees(math.atan((cfg.height / cfg.width) * aperture / (2.0 * focal)))
        return {
            "sensor_name": _CAM_SENSOR[key],
            "resolution_hw": [int(cfg.height), int(cfg.width)],
            "focal_length_mm": focal,
            "horizontal_aperture_mm": aperture,
            "fov_horizontal_deg": round(fov_h, 2),
            "fov_vertical_deg": round(fov_v, 2),
            "data_types": list(cfg.data_types),
            "offset_pos_env_local": [float(v) for v in cfg.offset.pos],
            "offset_rot_xyzw": [float(v) for v in cfg.offset.rot],
            "offset_convention": str(cfg.offset.convention),
            "mount": (
                "static camera in front of the robot, env-local offset; looks at the parcel work area"
                if key == "front"
                else f"eye-in-hand: parented to {WRIST_BODY_NAMES[key]} with a fixed local offset"
            ),
            "extrinsics_source": f"per-frame pose in frames.npz: {key}_pos_w / {key}_quat_w_ros",
            "intrinsics_source": f"per-frame matrix in frames.npz: {key}_K",
        }

    cam_info = {key: _cam_effective(key) for key in camera_keys}
    if _front_only:
        print(f"[COLLECT] 仅前视相机（腕部相机不创建）| data_types = "
              f"{cam_info['front']['data_types']}", flush=True)
    print("[COLLECT] 生效相机配置：", flush=True)
    for key, ci in cam_info.items():
        print(f"    {key:12s} {ci['resolution_hw'][1]}x{ci['resolution_hw'][0]} @ {ci['focal_length_mm']:.1f}mm "
              f"(fov {ci['fov_horizontal_deg']:.1f}x{ci['fov_vertical_deg']:.1f} deg) "
              f"pos {[round(v, 3) for v in ci['offset_pos_env_local']]}", flush=True)

    if not env_cfg.enable_cameras:
        raise SystemExit("[COLLECT] env_cfg.enable_cameras 仍为 False —— 相机不会被创建")
    if env_cfg.enable_image_obs:
        raise SystemExit("[COLLECT] 期望状态观测（enable_image_obs=False），但走出了图像观测分支")

    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
    print(f"[COLLECT] 加载策略 {checkpoint}", flush=True)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    scene = unwrapped.scene
    device = unwrapped.device
    n_envs = unwrapped.num_envs
    max_steps = int(unwrapped.max_episode_length)
    dt = unwrapped.step_dt

    # 检查观测维度（62 维才和 best model 匹配）
    obs_probe, _ = env.reset()
    obs_dim = int(policy_obs(obs_probe).shape[-1])
    print(f"[COLLECT] 观测维度 = {obs_dim}（期望 62）| {n_envs} envs | 回合 {max_steps} 步 ({max_steps * dt:.1f} s)",
          flush=True)
    if obs_dim != 62:
        print(f"[COLLECT] 警告：观测维度不是 62，本脚本的真值排布假设 62 维", flush=True)

    # 场景实体
    parcel = scene["parcel"]
    robot = scene["robot"]
    origins = scene.env_origins
    cams = {k: scene.sensors[_CAM_SENSOR[k]] for k in camera_keys}
    print(f"[COLLECT] 场景相机 {list(cams)} | 传感器总数 {len(scene.sensors)}", flush=True)

    # 受控关节在 articulation 里的索引（与观测里的 18 维同序）
    ctrl_ids, ctrl_names = robot.find_joints(CONTROLLED_JOINTS, preserve_order=False)
    all_joint_names = list(robot.data.joint_names)
    print(f"[COLLECT] 受控关节 {len(ctrl_ids)} 个（articulation 共 {len(all_joint_names)} 个）", flush=True)

    # link7 的 body 索引（腕部相机外参可由物理状态重算，不依赖 USD prim 是否同步）
    wrist_body_ids = {}
    for key, body_name in WRIST_BODY_NAMES.items():
        ids, _ = robot.find_bodies([body_name])
        wrist_body_ids[key] = ids[0] if ids else None

    # 静态相机外参（env-local）：前视相机挂在本 env 根下，配置里就是局部量
    front_pos_w_static = np.asarray(FRONT_CAM_POS, dtype=np.float32)

    # ------------------------------------------------------------------ 热身
    action_dim = unwrapped.action_manager.total_action_dim
    zero_action = torch.zeros(n_envs, action_dim, device=device)
    print(f"[COLLECT] 空动作热身 {args.warmup} 步（RTX 降噪收敛）...", flush=True)
    for _ in range(args.warmup):
        env.step(zero_action)

    # ------------------------------------------------------------------ 目录
    for key in camera_keys:
        (run_dir / "images" / key).mkdir(parents=True, exist_ok=True)
        if not args.no_depth:
            (run_dir / "depth" / key).mkdir(parents=True, exist_ok=True)
    (run_dir / "previews").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 采集
    frame_idx = 0
    records: dict[str, list] = {}

    def push(name: str, value) -> None:
        records.setdefault(name, []).append(value)

    def read_cam(key: str) -> dict:
        cam = cams[key]
        out = cam.data.output
        rgb = out["rgb"]
        depth = out.get("distance_to_image_plane", None) if not args.no_depth else None
        return {
            "rgb": to_numpy(rgb),  # (N, H, W, 3)
            "depth": to_numpy(depth) if depth is not None else None,  # (N, H, W, 1)
            "pos_w": to_numpy(cam.data.pos_w),
            "quat_w_ros": to_numpy(cam.data.quat_w_ros),
            "quat_w_world": to_numpy(cam.data.quat_w_world),
            "K": to_numpy(cam.data.intrinsic_matrices),
        }

    # 采集窗口状态（逐 env）：达标即停 + 每回合步数上限
    # 为什么要窗口：奖励是"成功后不终止"，回合内盒子会被反复推走，既产生大量
    # "已翻正且静止"的重复帧，又把盒子推出取景范围。实测覆盖率 85% -> 97.5%。
    rec_done = np.zeros(n_envs, dtype=bool)
    rec_frames = np.zeros(n_envs, dtype=np.int64)
    success_step = np.full(n_envs, -1, dtype=np.int64)
    # 达标时的**已保存帧数**（含达标那一帧）。尾帧按"实际落盘的帧"计，
    # 否则被遮挡剔除掉的帧会让尾帧数量不足。
    success_rec = np.full(n_envs, -1, dtype=np.int64)
    ep_index = np.zeros(n_envs, dtype=np.int64)      # 该 env 正在录第几个回合（从 0 起）
    ep_finished = np.zeros(n_envs, dtype=np.int64)   # 该 env 已录完几个回合
    policy_step = 0
    ep_frames: list[list[int]] = [[] for _ in range(n_envs)]  # 当前回合已落盘的帧号（用于丢弃）
    ep_retries = np.zeros(n_envs, dtype=np.int64)             # 当前回合已重录次数
    n_rerecord = 0        # 触发重录的次数
    n_abandoned = 0       # 重录次数超限被放弃的回合数
    n_discarded = 0       # 被丢弃的帧数
    n_invisible_drop = 0  # 因"盒子离开画面"而触发的重录次数
    n_invisible_frames = 0  # 因此丢弃的帧数
    n_occl_skipped = 0      # 因遮挡超阈值而未保存的帧数
    # 每台相机"回合第一帧看得见"的统计（需求：腕部相机初始必须看见盒子）
    start_vis_seen = {c: 0 for c in camera_keys}
    start_vis_total = {c: 0 for c in camera_keys}
    inv_streak = np.zeros(n_envs, dtype=np.int64)  # 盒子连续不可见的帧数
    track_cam = "front" if "front" in camera_keys else camera_keys[0]

    def discard_frames(frames: list[int]) -> None:
        """丢弃若干帧：删掉已落盘的 png，并从 records 里剔掉对应条目。

        frame_index 不回退（留空洞没关系 —— 校验/出视频都按 frames.npz 里的
        frame_index 取值，不假设连续）。
        """
        nonlocal n_discarded
        if not frames:
            return
        for fn in frames:
            for key in camera_keys:
                (run_dir / "images" / key / f"{fn:06d}.png").unlink(missing_ok=True)
                if not args.no_depth:
                    (run_dir / "depth" / key / f"{fn:06d}.png").unlink(missing_ok=True)
                (run_dir / "previews" / f"check_{fn:06d}_{key}.png").unlink(missing_ok=True)
        keep = ~np.isin(np.asarray(records["frame_index"]), np.asarray(frames))
        for k in list(records.keys()):
            records[k] = [v for v, kp in zip(records[k], keep) if kp]
        n_discarded += len(frames)
    # 注意：policy_step 是**全局**计数器，而 max_policy_steps 原本是"单回合"上限，
    # 多回合时必须按回合数放大，否则 --episodes 100 会在第二个回合左右就退出。
    _per_ep = args.window_steps + args.after_success + 2 if args.window_steps else None
    if _per_ep is None:
        max_policy_steps = 10**9
    else:
        max_policy_steps = _per_ep * (args.episodes if args.episodes > 0 else 1) + 10

    def env_all_done() -> bool:
        """所有 env 都已录满 --episodes 个回合（--episodes=0 表示不设上限）。"""
        return args.episodes > 0 and bool((ep_finished >= args.episodes).all())


    print(f"[COLLECT] 开始采集 {args.frames} 帧（每帧 {len(camera_keys)} 台相机）"
          f"{f' | 窗口: 每 env 最多 {args.window_steps} 步, 达标后 +{args.after_success} 帧' if args.window_steps else ''}...",
          flush=True)
    t0 = time.time()
    obs, _ = env.reset()

    while (frame_idx < args.frames and policy_step < max_policy_steps
           and not env_all_done() and not rec_done.all()):
        with torch.no_grad():
            actions = policy(obs)

        # 记录"策略实际看到的观测"（含观测噪声）——将来做行为克隆时要用；
        # 真值另有来源（物理状态），不要混用。
        obs_now = policy_obs(obs).detach().clone()
        obs, _, done, _ = env.step(actions)  # rsl-rl 5.x: (TensorDict, rew, dones, extras)
        policy_step += 1

        if args.debug_force_reset_at and policy_step == args.debug_force_reset_at:
            # 把回合计时器顶满 -> 下一步 time_out 终止并自动 reset，用来验证重录路径
            unwrapped.episode_length_buf[:] = int(unwrapped.max_episode_length)
            print(f"  [自测] 第 {policy_step} 步人为制造回合终止", flush=True)

        # 步后统一读取：图像与真值都对应同一物理时刻
        cam_data = {k: read_cam(k) for k in camera_keys}
        parcel_pos_w = parcel.data.root_pos_w.torch
        parcel_quat_w = parcel.data.root_quat_w.torch
        label_normal_w = label_normal_from_quat(parcel_quat_w)
        # 盒子朝向的完整刻画：局部 +X（正面）/ +Y（左侧）/ 绕标签轴自转角
        # （label_normal_w 已给出局部 +Z；有了 +X 就唯一确定整个 3x3 旋转：+Y = +Z x +X）
        _Rq = quat_to_matrix(parcel_quat_w)
        parcel_local_x_w = _Rq[..., 0]
        parcel_local_y_w = _Rq[..., 1]
        parcel_twist_deg = twist_deg_about_label(parcel_quat_w, parcel_local_x_w, label_normal_w)
        parcel_pos_env = parcel_pos_w - origins
        on_table = (
            ((parcel_pos_env[:, 0] - TABLE_POS[0]).abs() < TABLE_SIZE[0] / 2.0 - 0.03)
            & ((parcel_pos_env[:, 1] - TABLE_POS[1]).abs() < TABLE_SIZE[1] / 2.0 - 0.03)
            & (parcel_pos_env[:, 2] > TABLE_TOP - DROP_TOL)
        )
        joint_pos_all = robot.data.joint_pos.torch
        joint_vel_all = robot.data.joint_vel.torch
        body_pos_w = robot.data.body_pos_w.torch
        body_quat_w = robot.data.body_quat_w.torch
        step_in_episode = unwrapped.episode_length_buf

        done_mask = done if done.dtype == torch.bool else done.bool()

        # 录制中却发生了**环境自动 reset**（盒子被推下桌 -> parcel_out_of_bound、关节异常、
        # NaN 等）：本回合已录的帧全部作废，从新回合重录 —— 否则一个 episode_index 里会混进
        # 两段环境回合（实测 batch200 的 100 个回合里有 11 个这种脏回合）。
        skipped = np.zeros(n_envs, dtype=bool)
        for e in range(n_envs):
            if rec_done[e] or not bool(done_mask[e].item()):
                continue
            skipped[e] = True
            n_bad = len(ep_frames[e])
            discard_frames(ep_frames[e])
            ep_frames[e] = []
            rec_frames[e] = 0
            success_step[e] = -1
            success_rec[e] = -1
            if args.no_rerecord:
                rec_done[e] = True
                ep_finished[e] += 1
                print(f"  [警告] env{e} 第 {int(ep_index[e]) + 1} 回合中途被 reset，"
                      f"本回合仅保留 {n_bad} 帧（--no_rerecord 已关闭重录）", flush=True)
                continue
            ep_retries[e] += 1
            if ep_retries[e] > args.max_retries:
                # 反复被 reset（策略总是把盒子推飞）：放弃该回合，保证数据集里没有脏回合
                rec_done[e] = True
                ep_finished[e] += 1
                n_abandoned += 1
                print(f"  [放弃] env{e} 第 {int(ep_index[e]) + 1} 回合连续 {args.max_retries} 次"
                      f"被 reset，已丢弃 {n_bad} 帧并跳过该回合", flush=True)
            else:
                n_rerecord += 1
                print(f"  [重录] env{e} 第 {int(ep_index[e]) + 1} 回合中途被 reset，"
                      f"丢弃 {n_bad} 帧，从新回合重录（第 {int(ep_retries[e])}/{args.max_retries} 次）",
                      flush=True)

        # (B) 盒子**完全离开画面**且连续超过 N 帧 -> 丢弃本回合重录。
        #     与上面的情况不同：这里环境不会自己 reset（盒子只是被推出了视野），
        #     所以要主动强制重置该 env，才能从新回合重录。
        # 跟踪相机的逐 env 度量：可见度（判断是否离画）、遮挡比（逐帧剔除）共用一次投影
        occl_all = None
        if camera_keys and (args.drop_if_invisible_frames > 0 or args.max_occlusion < 1.0):
            _ct = cam_data[track_cam]
            _h, _w = _ct["rgb"].shape[1], _ct["rgb"].shape[2]
            _cpos = torch.as_tensor(_ct["pos_w"], device=device)
            _cquat = torch.as_tensor(_ct["quat_w_ros"], device=device)
            _K = torch.as_tensor(_ct["K"], device=device)
            _uv8, _z8 = silhouette_corner_uv(parcel_pos_w, parcel_quat_w, _cpos, _cquat, _K)
            _u, _v = _uv8[..., 0], _uv8[..., 1]
            _full = ((_u.max(1).values - _u.min(1).values) * (_v.max(1).values - _v.min(1).values)).clamp_min(1e-9)
            _cu0, _cu1 = _u.min(1).values.clamp(min=0.0), _u.max(1).values.clamp(max=float(_w))
            _cv0, _cv1 = _v.min(1).values.clamp(min=0.0), _v.max(1).values.clamp(max=float(_h))
            _vis = (((_cu1 - _cu0).clamp(min=0) * (_cv1 - _cv0).clamp(min=0)) / _full).clamp(0, 1)
            _zc = _z8[:, 0]  # 角点深度的量级足够用来判断"是否在相机后方"
            out_of_view = (_vis < args.visible_eps) | (_zc <= 0.02)
            if args.max_occlusion < 1.0 and _ct["depth"] is not None:
                # 用**中心**的相机系深度作参考（与校验脚本一致）
                _, _zc_center = project_to_pixels(parcel_pos_w, _cpos, _cquat, _K)
                _half_diag = float(np.linalg.norm(PARCEL_SIZE_HALF))
                _dep = _ct["depth"]
                occl_all = np.full(n_envs, np.nan, dtype=np.float32)
                _uv8_np = to_numpy(_uv8)
                for e in range(n_envs):
                    _d = _dep[e]
                    _d = _d[..., 0] if _d.ndim == 3 else _d
                    occl_all[e] = occlusion_ratio_np(_d, _uv8_np[e], float(_zc_center[e]), _half_diag)
            to_force_reset = []
            # 只有显式启用（>0）才做"离开画面"丢弃；否则把计数清零即可
            # （0 表示关闭 —— 若不判这一句，0 会让 inv_streak < 0 恒不成立、变成每帧都丢）
            for e in range(n_envs):
                if args.drop_if_invisible_frames <= 0:
                    inv_streak[e] = 0
                    continue
                if rec_done[e] or skipped[e]:
                    inv_streak[e] = 0
                    continue
                inv_streak[e] = inv_streak[e] + 1 if bool(out_of_view[e]) else 0
                if inv_streak[e] < args.drop_if_invisible_frames:
                    continue
                # 触发丢弃
                n_bad = len(ep_frames[e])
                discard_frames(ep_frames[e])
                n_invisible_drop += 1
                n_invisible_frames += n_bad
                ep_frames[e] = []
                rec_frames[e] = 0
                success_step[e] = -1
                success_rec[e] = -1
                inv_streak[e] = 0
                skipped[e] = True
                ep_retries[e] += 1
                if ep_retries[e] > args.max_retries:
                    rec_done[e] = True
                    ep_finished[e] += 1
                    n_abandoned += 1
                    print(f"  [放弃] env{e} 第 {int(ep_index[e]) + 1} 回合因盒子连续 "
                          f"{args.drop_if_invisible_frames} 帧离开画面，已丢弃 {n_bad} 帧并跳过", flush=True)
                else:
                    to_force_reset.append(e)
                    print(f"  [重录] env{e} 第 {int(ep_index[e]) + 1} 回合盒子连续 "
                          f"{args.drop_if_invisible_frames} 帧离开画面，丢弃 {n_bad} 帧，强制重置后重录"
                          f"（第 {int(ep_retries[e])}/{args.max_retries} 次）", flush=True)
            if to_force_reset:
                unwrapped._reset_idx(torch.tensor(to_force_reset, dtype=torch.long, device=device))
                obs = env.get_observations()
                for e in to_force_reset:
                    ep_retries[e] = ep_retries[e]  # 保持重试计数
                    inv_streak[e] = 0

        # 先更新窗口状态，再决定记哪些 env
        for e in range(n_envs):
            if rec_done[e] or skipped[e]:
                continue
            if success_step[e] < 0 and float(label_normal_w[e, 2].item()) > UP_COS:
                success_step[e] = policy_step
                success_rec[e] = rec_frames[e] + 1  # +1 = 本步将要保存的这一帧
            if success_step[e] >= 0 and rec_frames[e] >= success_rec[e] + args.after_success:
                rec_done[e] = True  # 达标（+ after_success 个**已保存**尾帧）即停
            elif args.window_steps and rec_frames[e] >= args.window_steps:
                rec_done[e] = True  # 本回合步数用尽
            if rec_done[e]:
                ep_finished[e] += 1  # 本回合录制结束（含最后一回合，供"全部录完"判据用）
                ep_frames[e] = []
                ep_retries[e] = 0
                inv_streak[e] = 0
                inv_streak[e] = 0

        for e in range(n_envs):
            if rec_done[e] or skipped[e] or frame_idx >= args.frames:
                continue
            # 逐帧剔除：遮挡超过阈值就不保存这一帧（计入统计，但不算进 rec_frames）
            if occl_all is not None and np.isfinite(occl_all[e]) and occl_all[e] > args.max_occlusion:
                n_occl_skipped += 1
                continue
            f = frame_idx
            ep_frames[e].append(f)
            step_e = int(step_in_episode[e].item())
            push("frame_index", f)
            push("env_id", e)
            push("episode_index", int(ep_index[e]))
            push("step_in_episode", step_e)
            push("sim_time", float(step_e) * dt)  # 本回合内的仿真时间
            push("reset_flag", step_e <= 1)  # 该帧是刚 reset 后的第一帧（回合边界）
            push("parcel_pos_env", to_numpy(parcel_pos_env[e]))
            push("parcel_pos_w", to_numpy(parcel_pos_w[e]))
            push("env_origin", to_numpy(origins[e]))
            push("parcel_quat_w", to_numpy(parcel_quat_w[e]))
            push("label_normal_w", to_numpy(label_normal_w[e]))
            push("parcel_local_x_w", to_numpy(parcel_local_x_w[e]))
            push("parcel_local_y_w", to_numpy(parcel_local_y_w[e]))
            push("parcel_twist_deg", float(parcel_twist_deg[e].item()))
            push("label_up", float(label_normal_w[e, 2].item()))
            push("pre_success", success_step[e] < 0 or policy_step <= success_step[e])
            push("on_table", bool(on_table[e].item()))
            push("joint_pos_all", to_numpy(joint_pos_all[e]))
            push("joint_vel_all", to_numpy(joint_vel_all[e]))
            push("joint_pos_ctrl", to_numpy(joint_pos_all[e, ctrl_ids]))
            push("joint_vel_ctrl", to_numpy(joint_vel_all[e, ctrl_ids]))
            push("action", to_numpy(actions[e]))
            push("obs", to_numpy(obs_now[e]))

            for key in camera_keys:
                cd = cam_data[key]
                rgb_e = cd["rgb"][e]
                save_rgb(run_dir / "images" / key / f"{f:06d}.png", rgb_e)
                push(f"{key}_rgb_mean", float(rgb_e.mean()))

                if cd["depth"] is not None:
                    dep_e = cd["depth"][e]
                    dep2 = dep_e[..., 0] if dep_e.ndim == 3 else dep_e
                    valid, clipped = save_depth_mm(run_dir / "depth" / key / f"{f:06d}.png", dep2)
                    push(f"{key}_depth_valid_px", valid)
                    push(f"{key}_depth_clipped_px", clipped)

                pos_w = torch.as_tensor(cd["pos_w"][e], device=device).unsqueeze(0)
                quat_ros = torch.as_tensor(cd["quat_w_ros"][e], device=device).unsqueeze(0)
                K = torch.as_tensor(cd["K"][e], device=device).unsqueeze(0)
                uv, z = project_to_pixels(parcel_pos_w[e].unsqueeze(0), pos_w, quat_ros, K)
                push(f"{key}_pos_w", cd["pos_w"][e])
                push(f"{key}_quat_w_ros", cd["quat_w_ros"][e])
                push(f"{key}_quat_w_world", cd["quat_w_world"][e])
                push(f"{key}_K", cd["K"][e])
                push(f"{key}_parcel_uv", to_numpy(uv[0]))
                push(f"{key}_parcel_depth", float(z[0].item()))
                u, v = float(uv[0, 0]), float(uv[0, 1])
                push(f"{key}_parcel_in_frame", bool(0 <= u < cd["rgb"].shape[2] and 0 <= v < cd["rgb"].shape[1]))

                # 盒子轮廓可见度（0..1）：数据集可用性的核心指标，被画面截断就说明取景不对
                vis = parcel_visible_fraction(
                    parcel_pos_w[e].unsqueeze(0),
                    parcel_quat_w[e].unsqueeze(0),
                    pos_w,
                    quat_ros,
                    K,
                    PARCEL_SIZE_HALF,
                    (cd["rgb"].shape[2], cd["rgb"].shape[1]),
                )
                push(f"{key}_parcel_visible_frac", float(vis[0].item()))
                if step_e <= 1:                      # 本回合第一帧
                    start_vis_total[key] += 1
                    if float(vis[0].item()) >= args.min_start_visible:
                        start_vis_seen[key] += 1
                    elif key in [c.strip() for c in args.require_cam_visible_at_start.split(",") if c.strip()]:
                        print(f"  [警告] env{e} ep{int(ep_index[e])}: 相机 {key} 在回合起始"
                              f"看不见盒子（取景 {float(vis[0].item()):.2f} < {args.min_start_visible}）"
                              f" —— 见 env cfg 里 WRIST_CAM_* 注释", flush=True)

                if f < args.preview:
                    uv8, _ = project_to_pixels(
                        _parcel_corners_w(
                            parcel_pos_w[e].unsqueeze(0), parcel_quat_w[e].unsqueeze(0), PARCEL_SIZE_HALF
                        )[0],
                        pos_w.repeat(8, 1),
                        quat_ros.repeat(8, 1),
                        K.repeat(8, 1, 1),
                    )
                    draw_preview(
                        rgb_e, to_numpy(uv8), float(label_normal_w[e, 2].item()), key, f, float(vis[0].item())
                    ).save(run_dir / "previews" / f"check_{f:06d}_{key}.png")

            frame_idx += 1
            rec_frames[e] += 1

        # 窗口结束的 env：强制重置、接着录下一回合（--episodes>1 或 =0 时）
        to_reset = [
            e for e in range(n_envs)
            # 用"当前已开始第几个回合"(ep_index，0 起) 判断还要不要开新回合：
            # 录完第 k 个回合时 ep_index=k-1，只要 (k-1)+1 < episodes 就还该继续。
            # （原先用 ep_finished 会少录最后一个回合：录完第 2 个就判停。）
            if rec_done[e] and (args.episodes == 0 or ep_index[e] + 1 < args.episodes)
            and frame_idx < args.frames
        ]
        if to_reset:
            ids = torch.tensor(to_reset, dtype=torch.long, device=device)
            # 走框架自身的重置入口（环境终止时也调它）：重置场景、重抽 reset 模式事件
            # （盒子位姿/摩擦等）并刷新 obs_buf。
            unwrapped._reset_idx(ids)
            obs = env.get_observations()  # 重建 TensorDict，保证下一步的观测是重置后的
            for e in to_reset:
                ep_index[e] += 1
                rec_done[e] = False
                rec_frames[e] = 0
                success_step[e] = -1
                success_rec[e] = -1
                ep_frames[e] = []
                ep_retries[e] = 0
                inv_streak[e] = 0
            print(f"  ... 回合切换：env {to_reset} 开始录第 {[int(ep_index[e]) + 1 for e in to_reset]} 回合"
                  f"（累计 {frame_idx} 帧）", flush=True)

        if frame_idx % 20 == 0 or frame_idx >= args.frames or env_all_done():
            print(f"  ... {frame_idx}/{args.frames} 帧  ({time.time() - t0:.1f} s)", flush=True)

    elapsed = time.time() - t0
    n_saved = len(records.get("frame_index", []))  # frame_idx 把被丢弃的帧也算进去了
    if n_saved == 0:
        raise SystemExit(
            "[COLLECT] 一帧都没录到（所有回合都被丢弃/放弃）。\n"
            f"          环境reset重录 {n_rerecord} 次 / 盒子离开画面重录 {n_invisible_drop} 次 / "
            f"放弃回合 {n_abandoned} 个。\n"
            "          若是 --drop_if_invisible_frames / --visible_eps 设得太严，请放宽；"
            "若是策略真的无法完成任务，可加大 --max_retries。")
    print(f"[COLLECT] 采集完成：保留 {n_saved} 帧"
          f"{f'（录制 {frame_idx} 帧，丢弃 {frame_idx - n_saved} 帧）' if n_saved != frame_idx else ''}"
          f"，用时 {elapsed:.1f} s（{frame_idx / max(elapsed, 1e-6):.1f} 帧/s）", flush=True)

    # ------------------------------------------------------------------ 起始可见性汇总
    # 需求：腕部相机在**回合初始**必须能看见盒子（运动过程中允许看不见），
    # 所以这里按相机逐一报告"回合第一帧盒子在画面内"的比例；低于 ~0.9 就该调
    # WRIST_CAM_LOOKAT_FORWARD / WRIST_CAM_FOCAL / WRIST_CAM_LOCAL 了。
    print("[COLLECT] 各相机『回合起始可见率』（多相机训练要求腕部相机初始可见）:", flush=True)
    for key in camera_keys:
        tot = start_vis_total[key]
        seen = start_vis_seen[key]
        rate = seen / tot if tot else float("nan")
        flag = "OK  " if (tot and rate >= 0.9) else "WARN"
        print(f"  [{flag}] {key:12s} {seen}/{tot} 回合起始看得见（取景 >= {args.min_start_visible}）", flush=True)

    # ------------------------------------------------------------------ 落盘
    arrays = {k: np.stack(v) if isinstance(v[0], (np.ndarray, list)) else np.asarray(v) for k, v in records.items()}
    np.savez_compressed(run_dir / "frames.npz", **arrays)
    print(f"[COLLECT] frames.npz 已写出（{len(arrays)} 个数组）", flush=True)

    if not args.no_copy_checkpoint:
        shutil.copy2(checkpoint, run_dir / "checkpoint.pt")

    # 逐回合统计（直接从已记录的数组推，不额外记账）：帧数、起手姿态、达标帧
    ep_summary = []
    for ep in np.unique(arrays["episode_index"]):
        m = arrays["episode_index"] == ep
        up = arrays["label_up"][m]
        hit = up > UP_COS
        first_up = float(up[0])
        ep_summary.append({
            "episode": int(ep),
            "frames": int(m.sum()),
            "initial_label_up": first_up,
            "initial": ("down(180deg)" if first_up < -0.5
                        else ("side(90deg)" if abs(first_up) < 0.5 else "up")),
            "reached_goal": bool(hit.any()),
            "success_frame": (int(np.argmax(hit)) + 1) if hit.any() else None,
        })

    info = {
        "schema_version": 1,
        "run_name": run_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "state-policy rollout with camera images + parcel 6-DoF ground truth, for perception CNN training",
        "frames": int(frame_idx),
        "num_envs": int(n_envs),
        "policy_rate_hz": float(1.0 / dt),
        "sim_dt": float(unwrapped.cfg.sim.dt),
        "decimation": int(unwrapped.cfg.decimation),
        "episode_length_steps": int(max_steps),
        "warmup_steps": int(args.warmup),
        "collection_window": {
            "episodes_per_env": int(args.episodes),
            "episodes_completed": {f"env{e}": int(ep_finished[e]) for e in range(n_envs)},
            "window_steps_per_env": int(args.window_steps),
            "frames_after_first_success": int(args.after_success),
            "note": "0/0 = 无窗口（记满 --frames）。开了窗口就只记『达标前 + 上限步数』那段，"
                    "实测取景覆盖率 97.5% vs 整回合 85%，且避免大量已翻正后静止的重复帧。",
        },
        "seed": args.seed,
        "task_id": TASK,
        "source_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "copied_into_run": not args.no_copy_checkpoint,
            "copied_name": None if args.no_copy_checkpoint else "checkpoint.pt",
            "note": "62-dim state policy, deterministic inference (act_inference = distribution mean)",
        },
        "observation": {
            "dim": int(obs_dim),
            "layout": (
                "0-17 joint_pos(18, limit-normalized) | 18-35 joint_vel(18, x0.1) | 36-38 parcel_pos(3, env-local) | "
                "39-42 parcel_quat(4, xyzw, world) | 43-45 label_normal(3, world) | 46-61 last_action(16)"
            ),
            "note": (
                "'obs' in frames.npz is the noisy observation the policy actually saw. "
                "The ground-truth fields (parcel_pos_env / parcel_quat_w / label_normal_w) come from physics state "
                "and carry NO observation noise & NO action noise."
            ),
        },
        "conventions": {
            "quaternion_order": "xyzw",
            "parcel_pos_env": "world-axis-aligned position minus env origin [m]; identical to what the policy consumes",
            "parcel_quat_w": "world-frame orientation of the parcel root",
            "label_normal_w": "unit vector = R(parcel_quat_w) @ (0,0,1); the task goal is label_normal_w[2] -> +1",
            "parcel_local_x_w": "unit vector = R(parcel_quat_w) @ (1,0,0) — the box's FRONT (local +X) in world",
            "parcel_local_y_w": "unit vector = R(parcel_quat_w) @ (0,1,0) — the box's LEFT (local +Y) in world; "
                                "with local_x_w and label_normal_w the full 3x3 rotation is determined (+Y = +Z x +X)",
            "parcel_twist_deg": "rotation about the box's own label axis, in degrees (-180,180], measured against "
                                "the zero-twist reference (shortest-arc rotation taking world +Z to label_normal_w). "
                                "This is the one DOF that label_normal alone cannot determine.",
            "label_up": "label_normal_w[2] (cos of the tilt between label normal and world +Z)",
            "on_table": f"parcel center within table footprint (x={TABLE_POS[0]}, y={TABLE_POS[1]}, half={TABLE_SIZE[0]/2},{TABLE_SIZE[1]/2}) and z > {TABLE_TOP - DROP_TOL:.3f} m",
            "action": "16-dim: 14 arm joints (rad, relative to home) + 2 coupled gripper (m, relative to home); NOT clipped",
            "image_color_order": "RGB, uint8",
            "depth_units": "millimetres, uint16 PNG (distance_to_image_plane); 0 = invalid/no return",
            "depth_sensor": "distance_to_image_plane = distance along the camera optical axis (not radial)",
        },
        "cameras": cam_info,
        "frame_arrays": sorted(arrays.keys()),
        "key_arrays_note": (
            "parcel_pos_env = world-axis position minus env origin; parcel_pos_w = world position; "
            "env_origin = per-env origin; {cam}_parcel_visible_frac = fraction of the box silhouette "
            "inside the image (1.0 = fully visible) — use it to filter frames where the box is cut off."
        ),
        "parcel_size_m": [float(v) for v in PARCEL_SIZE],
        "episodes": ep_summary,
        "frame_filter": {
            "max_occlusion": float(args.max_occlusion),
            "occlusion_skipped_frames": int(n_occl_skipped),
            "note": "遮挡比超过 max_occlusion 的帧不保存（公式与 verify_vision_data 判据 4 相同）",
        },
        "rerecord": {
            "enabled": not args.no_rerecord,
            "drop_if_invisible_frames": int(args.drop_if_invisible_frames),
            "invisible_events": int(n_invisible_drop),
            "invisible_discarded_frames": int(n_invisible_frames),
            "events": int(n_rerecord),
            "discarded_frames": int(n_discarded),
            "abandoned_episodes": int(n_abandoned),
            "max_retries": int(args.max_retries),
            "note": "录制中发生环境自动 reset 时丢弃本回合已录帧并重录；超过 max_retries 则放弃该回合",
        },
        "preview_video": None if args.no_preview_video else {
            "fps": float(args.video_fps),
            "formats": ["webp", "gif"],
            "note": "由 scripts/make_preview_video.py 合成；叠加了盒子真值投影与 label_up/可见度",
        },
        "files": {
            "image_pattern": "images/<cam>/{frame_index:06d}.png",
            "depth_pattern": None if args.no_depth else "depth/<cam>/{frame_index:06d}.png",
            "frames": "frames.npz",
            "previews": "previews/check_{frame:06d}_{cam}.png (crosshair = projected parcel center; green = inside frame)",
        },
        "known_limitations": [
            "Wrist cameras point at the fingertips (link7 local +X 15 cm) and frequently do NOT contain the parcel; "
            "check {cam}_parcel_in_frame / {cam}_rgb_mean before using them.",
            "FRAMING (measured on sample100): the front camera is aim-narrow (fx~366 px, fov_v~36 deg, 0.35 m away). "
            "During a flip the policy pushes the box ~16 cm (+x/+y); from about frame 40 the box rides off the TOP edge "
            "and only 44% of its silhouette stays in frame (62/100 frames below 50% visible). Reframe or record only "
            "the pre-flip window before collecting a real dataset.",
            "Frames are consecutive policy steps from few initial conditions; this is a pipeline-validation sample, "
            "not a training set. Cover initial orientations, lighting and textures deliberately for real training.",
            "reset_flag marks frames right after an episode reset (mixed episode boundary); filter them out if you "
            "assume temporal continuity.",
        ],
    }
    (run_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[COLLECT] dataset_info.json 已写出 -> {run_dir}", flush=True)

    # 简洁汇总
    print("\n" + "=" * 78)
    print(f"数据集: {run_dir}")
    print(f"  帧数 {frame_idx} | 相机 {camera_keys} | 观测 {obs_dim} 维 | 用时 {elapsed:.1f} s")
    for key in camera_keys:
        rgb_mean = arrays[f"{key}_rgb_mean"]
        in_frame = arrays[f"{key}_parcel_in_frame"]
        dep = arrays.get(f"{key}_parcel_depth", None)
        line = (f"  {key:12s} rgb均值 {rgb_mean.mean():6.1f} | 盒子在画面内 {100.0 * in_frame.mean():5.1f}%"
                f" | 盒子投影深度中位数 {np.median(dep):.3f} m" if dep is not None else
                f"  {key:12s} rgb均值 {rgb_mean.mean():6.1f} | 盒子在画面内 {100.0 * in_frame.mean():5.1f}%")
        print(line)
        if rgb_mean.mean() < 5.0:
            print(f"    !! {key} 图像接近全黑（均值 {rgb_mean.mean():.1f}），检查相机朝向/光照")
        # 可见度自检：这是数据集能否用于位姿学习的关键指标
        vis = arrays[f"{key}_parcel_visible_frac"]
        full = float((vis > 0.95).mean()) * 100.0
        cut = float((vis <= 0.5).mean()) * 100.0
        print(f"      盒子完整可见(>95%) {full:5.1f}% | 严重截断(<50%) {cut:5.1f}% | 平均可见 {100.0 * vis.mean():5.1f}%")
        if cut > 20.0:
            print(f"    !! {key} 有 {cut:.0f}% 的帧盒子被画面截断 —— 取景过窄，位姿 CNN 拿不到完整证据；"
                  "建议后移/抬高相机、改用更广的焦距，或缩短采集窗口")
    print(f"  盒子 label_up 范围 [{arrays['label_up'].min():+.3f}, {arrays['label_up'].max():+.3f}]")
    if n_occl_skipped:
        print(f"  逐帧剔除: 遮挡 >{100 * args.max_occlusion:.0f}% 的帧 {n_occl_skipped} 帧未保存")
    if n_rerecord or n_abandoned or n_invisible_drop:
        print(f"  重录: 环境reset触发 {n_rerecord} 次 | 盒子离开画面触发 {n_invisible_drop} 次"
              f" | 共丢弃 {n_discarded} 帧 | 放弃回合 {n_abandoned} 个")

    if ep_summary and (len(ep_summary) > 1 or ep_summary[0]["frames"] != n_saved):
        reached = sum(e["reached_goal"] for e in ep_summary)
        fr = np.array([e["frames"] for e in ep_summary])
        n_down = sum(e["initial"].startswith("down") for e in ep_summary)
        print(f"\n【逐回合统计】共 {len(ep_summary)} 个回合 | 达成目标 {reached} 个"
              f"（{100.0 * reached / len(ep_summary):.1f}%）")
        print(f"  每回合帧数: 中位 {int(np.median(fr))} | 最少 {int(fr.min())} | 最多 {int(fr.max())} | 合计 {int(fr.sum())}")
        print(f"  起手姿态: 标签朝下(180°) {n_down} 个 / 其它 {len(ep_summary) - n_down} 个")
        show = ep_summary if len(ep_summary) <= 12 else ep_summary[:6] + [None] + ep_summary[-5:]
        print(f"  {'回合':<6}{'帧数':>6}{'起手':>14}{'达标帧':>8}")
        for e in show:
            if e is None:
                print(f"  {'...':<6}")
                continue
            print(f"  {e['episode'] + 1:<6}{e['frames']:>6}{e['initial']:>14}"
                  f"{(e['success_frame'] if e['success_frame'] else '—'):>8}")
    print("=" * 78)

    # ------------------------------------------------------------------ 预览视频
    if not args.no_preview_video:
        # 独立子进程调用，工具本身只依赖 PIL/numpy（宿主 python3 也能跑），
        # 失败不影响已落盘的数据集，因此 check=False 只告警。
        import subprocess

        tool = pathlib.Path(__file__).resolve().parent / "make_preview_video.py"
        cmd = [sys.executable, str(tool), str(run_dir), "--cam", "front",
               "--fps", str(args.video_fps)]
        if args.video_episode is not None:
            cmd += ["--episode", str(args.video_episode)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            for line in (r.stdout or "").strip().splitlines():
                print(line, flush=True)
            if r.returncode != 0:
                print(f"[COLLECT] !! 预览视频生成失败（数据已保存）：{(r.stderr or '').strip()[-400:]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[COLLECT] !! 预览视频生成异常（数据已保存）：{type(e).__name__}: {e}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
