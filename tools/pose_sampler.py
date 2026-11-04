"""脚本化位姿采样的**采样计划**（纯 Python，不 import Isaac / torch，可单测）。

为什么要脚本化采样，而不是继续跑策略录数据
------------------------------------------
策略 rollout 只能覆盖"策略自己会走到的状态"，于是数据集的覆盖分布由策略决定：
实测 100 回合里初始位置只在 ±1.8 cm 内抖动、位置覆盖是桌面上的一条"走廊"、
"已翻正"状态只出现在每个回合的最后几帧（而且静止重复）。而位姿估计网络要的恰恰是
**每个状态在各种位置/遮挡/视角下都出现过**。

脚本化采样直接把盒子摆到指定状态、渲染、记真值：
* 覆盖可控：按桶配额生成（面类型 / 位置网格 / 遮挡 / 光照），缺什么补什么；
* 快：不用跑策略、不用等物理收敛到"成功"，每个状态几帧即可；
* 干净：不会出现"盒子被推到画面外"这种无解帧（采样时就按取景过滤）。

分布设计（依据实测缺口）
------------------------
1. **面类型**：``up``（标签朝上，已翻正）/ ``side``（标签在侧面）/ ``down``（标签朝下）。
   现有数据里 up 只占 6%（batch200_fix）或 25%（batch_redesign 但去冗余后≈0 增量），
   而任务的验收判据正是 label_up > cos15°，所以 up 必须刻意加权重。
2. **位置**：桌面工作区内分层网格采样。现有数据 5×5 网格里 3~5 格是空的、单格占 20% 以上。
3. **yaw**：绕世界 Z 均匀随机。盒子绕标签轴有 90° 对称（四个侧面同色），
   所以 yaw 本身不可从图像辨识 —— 但要均匀采样，避免模型把某个固定 yaw 记成先验。
4. **遮挡**：单独一档，用"重放已录到的机械臂关节构型"来制造真实遮挡（见采集脚本）。

> 说明：这里只采**静置**姿态（盒子贴桌面）。翻动过程中的中间态（``label_up`` 在 0.3~0.9）
> 不作为本模块的目标 —— 那类帧仍由策略 rollout 采集。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

#: 三种可静置的"面"（标签法向朝哪）
FACES = ("up", "side", "down")


@dataclass
class PoseSpec:
    """一个待拍摄的**静置**状态（盒子贴着桌面）。"""

    face: str                      # "up" | "side" | "down"
    xy: tuple[float, float]        # 环境系 (x, y)，米
    yaw_deg: float                 # 绕世界 +Z 的偏航角（度）

    def __post_init__(self) -> None:
        if self.face not in FACES:
            raise ValueError(f"face 必须是 {FACES}，收到 {self.face!r}")


@dataclass
class PlanConfig:
    #: 三种面的配额（会自动归一化）
    face_probs: dict = field(default_factory=lambda: {"up": 0.35, "side": 0.45, "down": 0.20})
    #: 桌面工作区（环境系，米）。默认取桌面中心 ±(0.10, 0.12)：
    #: 盒子半边长 0.045，留出安全边界，避免一开始就贴边/出画。
    x_range: tuple = (0.20, 0.40)
    y_range: tuple = (-0.02, 0.22)
    #: 位置分层网格（None = 纯均匀）。给 (nx, ny) 时按网格逐格配额，保证不留空格。
    grid: tuple | None = (5, 5)
    #: yaw 是否均匀随机
    uniform_yaw: bool = True


def _normalize(probs: dict) -> dict:
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("face_probs 之和必须 > 0")
    return {k: v / total for k, v in probs.items()}


def plan_states(n: int, cfg: PlanConfig, seed: int = 0) -> list[PoseSpec]:
    """生成 ``n`` 个待拍摄状态：按面类型配额、按网格分层、yaw 均匀。"""
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)
    probs = _normalize(cfg.face_probs)

    # ---- 面类型：按配额分配数量（保证每种面的实际占比接近配额）----
    counts = {f: int(round(p * n)) for f, p in probs.items()}
    # 舍入误差修正到最大的一项上
    diff = n - sum(counts.values())
    if diff:
        biggest = max(probs, key=probs.get)
        counts[biggest] += diff
    faces = [f for f, c in counts.items() for _ in range(c)]
    rng.shuffle(faces)

    # ---- 位置：分层网格（每格再均匀抖动），保证不留空格 ----
    xs = np.linspace(cfg.x_range[0], cfg.x_range[1], 400)
    ys = np.linspace(cfg.y_range[0], cfg.y_range[1], 400)
    if cfg.grid:
        nx, ny = cfg.grid
        cells = [(ix, iy) for ix in range(nx) for iy in range(ny)]
        rng.shuffle(cells)
        cell_of = [cells[i % len(cells)] for i in range(n)]
    else:
        nx = ny = 0
        cell_of = [None] * n

    specs: list[PoseSpec] = []
    for i in range(n):
        if cell_of[i] is None:
            xy = (float(rng.uniform(*cfg.x_range)), float(rng.uniform(*cfg.y_range)))
        else:
            ix, iy = cell_of[i]
            x0 = cfg.x_range[0] + (cfg.x_range[1] - cfg.x_range[0]) * ix / nx
            x1 = cfg.x_range[0] + (cfg.x_range[1] - cfg.x_range[0]) * (ix + 1) / nx
            y0 = cfg.y_range[0] + (cfg.y_range[1] - cfg.y_range[0]) * iy / ny
            y1 = cfg.y_range[0] + (cfg.y_range[1] - cfg.y_range[0]) * (iy + 1) / ny
            xy = (float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        yaw = float(rng.uniform(0.0, 360.0)) if cfg.uniform_yaw else 0.0
        specs.append(PoseSpec(face=faces[i], xy=xy, yaw_deg=yaw))
    _ = (xs, ys)  # 仅用于文档化范围
    return specs


def plan_summary(specs: list[PoseSpec], cfg: PlanConfig) -> dict:
    """统计计划分布（打印/写进 dataset_info 用）。"""
    if not specs:
        return {}
    faces = {f: sum(1 for s in specs if s.face == f) for f in FACES}
    xy = np.array([s.xy for s in specs])
    yaw = np.array([s.yaw_deg for s in specs])
    out = {
        "n": len(specs),
        "faces": faces,
        "xy_range": [[float(xy[:, 0].min()), float(xy[:, 0].max())],
                     [float(xy[:, 1].min()), float(xy[:, 1].max())]],
        "yaw_deg": {"min": float(yaw.min()), "max": float(yaw.max())},
    }
    if cfg.grid:
        nx, ny = cfg.grid
        H, _, _ = np.histogram2d(
            xy[:, 0], xy[:, 1], bins=[nx, ny],
            range=[cfg.x_range, cfg.y_range],
        )
        out["grid_cells"] = int((H > 0).sum())
        out["grid_empty"] = int((H == 0).sum())
    return out


def rest_height(face: str, size: tuple[float, float, float]) -> float:
    """该面静置时盒子中心离桌面的高度（米）。

    与 ``openarm_flip/mdp/events.py::reset_parcel_root_state`` 的 rest 计算必须一致，
    否则盒子会被放到错误的初始高度（落下来会产生非静止的 GT 帧）。
    """
    sx, sy, sz = size
    if face == "side":
        return sx / 2.0        # 侧立：绕水平轴转 90°，中心高度 = sx/2（与 sy/2 相同量级）
    if face in ("up", "down"):
        return sz / 2.0        # 立/倒：局部 Z 轴竖直，中心高度 = sz/2
    raise ValueError(f"未知 face: {face!r}")


def label_up_of(face: str) -> float:
    """该状态下 ``label_up``（标签法向的 z 分量）的期望值，用于自检。"""
    return {"up": 1.0, "side": 0.0, "down": -1.0}[face]
