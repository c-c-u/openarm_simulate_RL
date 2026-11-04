#!/usr/bin/env python3
# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""脚本化位姿采样：把盒子**主动摆到指定状态**、渲染、记真值，直接产出可训练数据集。

为什么需要这个（而不是继续跑策略录数据）
------------------------------------
``collect_vision_data.py`` 用状态策略 rollout，只能覆盖"策略自己会走到的状态"。实测
100 回合的产物：初始位置只在 ±1.8 cm 内抖动、位置覆盖是桌面上的一条走廊（5×5 网格里
3~5 格是空的、单格占 20%+）、"已翻正"状态只出现在每个回合末尾几帧且静止重复。

位姿网络要的是**同一状态在不同位置/遮挡下都出现过**，所以这里反过来：
按采样计划主动摆位 -> 渲染 -> 记真值。好处：
* 覆盖可控（按面类型/位置网格/遮挡分层配额），缺什么补什么；
* 快（不跑策略、不等"翻到成功"），每个状态 1 帧即可；
* 干净（采样时就按取景/遮挡桶过滤，不会产出"盒子出画"的无解帧）。

产物与 ``collect_vision_data.py`` **同一 schema**，所以 ``parcel_pose_cnn`` 的
``prepare_index.py`` / ``train.py`` / ``audit_dataset.py`` 零改动即可用。

每个状态是一个独立回合
----------------------
``episode_index`` = 状态序号、``step_in_episode`` = 0，因此
* 按回合划分 train/val/test 天然等价于"按状态划分"，不会有相邻帧泄漏；
* 时序堆叠（``data.stack``）不会跨状态拼接。

用法::

    # 1) 冒烟：12 个状态、不出手臂，先确认能跑通并看图
    $ISAAC_ROOT/python.sh scripts/collect_posed_data.py --smoke --run_name posed_smoke

    # 2) 正式：3000 个状态，面类型配额 up/side/down = 0.35/0.45/0.20
    $ISAAC_ROOT/python.sh scripts/collect_posed_data.py \
        --states 3000 --face-probs up:side:down=0.35:0.45:0.20 --run_name posed_v1

    # 3) 带遮挡：从已录数据集里重放真实机械臂构型制造遮挡（目标 20~60% 遮挡占比各半）
    $ISAAC_ROOT/python.sh scripts/collect_posed_data.py \
        --states 1500 --occlusion-prob 0.5 --occlusion-source ../perception_data/batch_redesign \
        --occl-min 0.15 --run_name posed_occl_v1

采集完照常体检::

    python3 scripts/../parcel_pose_cnn/scripts/audit_dataset.py <run_dir>   # 无 torch 也能跑
    python parcel_pose_cnn/scripts/prepare_index.py --runs <run_dir>
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# 参数（必须在 AppLauncher 之前解析：headless 等由它决定）
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="脚本化位姿采样（把盒子摆到指定状态并拍摄）")
parser.add_argument("--out", type=str, default=str(pathlib.Path("/home/blanc/isaacsim/perception_data")),
                    help="数据集根目录")
parser.add_argument("--run_name", type=str, default=None, help="输出子目录名（默认时间戳）")
parser.add_argument("--states", type=int, default=500, help="采样多少个状态（每个状态 1 帧）")
parser.add_argument("--frames-per-state", type=int, default=1,
                    help="每个状态存几帧。默认 1：同一状态多帧几乎完全重复，去冗余时会被丢掉")
parser.add_argument("--face-probs", type=str, default="up:side:down=0.35:0.45:0.20",
                    help="三种面类型的配额，例 up:side:down=0.35:0.45:0.20")
parser.add_argument("--x-range", type=float, nargs=2, default=(0.20, 0.40),
                    help="采样位置 env-local x 范围（米）")
parser.add_argument("--y-range", type=float, nargs=2, default=(-0.02, 0.22),
                    help="采样位置 env-local y 范围（米）")
parser.add_argument("--grid", type=int, nargs=2, default=(5, 5),
                    help="位置分层网格 (nx, ny)，保证不出现空格子；0 0 = 纯均匀采样")
parser.add_argument("--settle-steps", type=int, default=10, help="摆位后空跑多少物理步让场景稳定/Render 收敛")
parser.add_argument("--warmup", type=int, default=12, help="开场热身步数（RTX 降噪收敛）")
parser.add_argument("--max-tries", type=int, default=3, help="单个状态校验失败后的重试次数（0 = 不校验）")
parser.add_argument("--min-in-frame", type=float, default=0.5,
                    help="接受该状态要求的最小取景占比（盒子轮廓在画面内的比例）")
parser.add_argument("--occlusion-prob", type=float, default=0.0,
                    help="以多大概率摆一个机械臂构型来制造遮挡（0 = 不出手臂）")
parser.add_argument("--occlusion-source", type=str, default=None,
                    help="从哪个已录数据集重放机械臂关节构型（读它的 frames.npz: joint_pos_all）")
parser.add_argument("--occl-min", type=float, default=0.0, help="接受该状态要求的最小遮挡率")
parser.add_argument("--occl-max", type=float, default=1.0, help="接受该状态要求的最大遮挡率")
parser.add_argument("--cameras", type=str, default="front", help="相机列表，逗号分隔")
parser.add_argument("--cam-focal", type=float, default=None, help="前视相机焦距（mm），改视场")
parser.add_argument("--cam-pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--cam-target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
parser.add_argument("--cam-res", type=int, nargs=2, default=None, metavar=("W", "H"))
parser.add_argument("--no-depth", action="store_true", help="不保存深度图")
parser.add_argument("--smoke", action="store_true", help="冒烟模式：12 个状态、不出手臂、少量打印")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()
if args.smoke:
    args.states = min(args.states, 12)
    args.occlusion_prob = 0.0
    args.settle_steps = max(6, args.settle_steps)

_CAM_SENSOR = {"front": "front_cam", "wrist_left": "wrist_left_cam", "wrist_right": "wrist_right_cam"}
camera_keys = [c.strip() for c in args.cameras.split(",") if c.strip()]
for _k in camera_keys:
    if _k not in _CAM_SENSOR:
        raise SystemExit(f"[POSED] 未知相机 '{_k}'，可选：{sorted(_CAM_SENSOR)}")


def _parse_face_probs(spec: str) -> dict:
    """``up:side:down=0.35:0.45:0.20`` -> dict（也接受 ``up=0.5,side=0.3,down=0.2``）。"""
    body = spec.replace(",", ":").split("=")
    if len(body) != 2:
        raise SystemExit(f"[POSED] --face-probs 格式应为 up:side:down=0.35:0.45:0.20，收到 {spec!r}")
    keys, vals = body[0].split(":"), body[1].split(":")
    out = {k.strip(): float(v) for k, v in zip(keys, vals)}
    if abs(sum(out.values())) < 1e-9:
        raise SystemExit("[POSED] --face-probs 之和必须 > 0")
    return out


_face_probs = _parse_face_probs(args.face_probs)

# ---------------------------------------------------------------------------
# 启动 Isaac Sim（采集必须开相机）
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": True})
simulation_app = app_launcher.app

# numpy / torch / PIL 必须在 AppLauncher 之后导入（由 Isaac Sim 的 pip_prebundle 提供）
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from PIL import Image  # noqa: E402

import openarm_lab  # noqa: F401,E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.string import string_to_callable  # noqa: E402
from openarm_lab.openarm_flip import mdp  # noqa: E402
from openarm_lab.openarm_flip.checkpoint_utils import make_env_cfg  # noqa: E402
from openarm_lab.openarm_flip.openarm_flip_env_cfg import (  # noqa: E402
    CAM_DATA_TYPES,
    _camera_look_at_quat_xyzw,
    PARCEL_SIZE,
    TABLE_POS,
    TABLE_SIZE,
    TABLE_TOP,
)
from openarm_lab.tools.pose_sampler import (  # noqa: E402
    PlanConfig,
    label_up_of,
    plan_states,
    plan_summary,
)

TASK = "OpenArm-Flip-State-v0"
UP_COS = math.cos(math.radians(15))
_HORIZON = 5.0  # 深度裁剪面（米），与场景 far 面一致时才谈得上"无效像素"


# ---------------------------------------------------------------------------
# 小工具（与 collect_vision_data.py 同一套公式，便于两边数字可比）
# ---------------------------------------------------------------------------
def to_numpy(x) -> np.ndarray:
    if hasattr(x, "torch"):
        x = x.torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def label_normal_from_quat(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dim=-1)


def save_rgb(path: pathlib.Path, arr: np.ndarray) -> None:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def save_depth_mm(path: pathlib.Path, arr: np.ndarray) -> None:
    d = np.asarray(arr, dtype=np.float32)
    invalid = ~np.isfinite(d) | (d <= 1e-6)
    mm = np.where(invalid, 0.0, np.clip(d, 0.0, 65.535) * 1000.0)
    Image.fromarray(mm.astype(np.uint16), mode="I;16").save(path)


def project_to_pixels(points_w, cam_pos_w, cam_quat_w_ros, K):
    """世界点 (N,3) -> 像素 (N,2) 与相机系深度 (N,)。用四元数把点转到 ROS 光学系。"""
    x, y, z, w = cam_quat_w_ros.unbind(dim=-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    dp = points_w - cam_pos_w.reshape(-1, 3)
    pc = torch.bmm(R.transpose(1, 2), dp.unsqueeze(-1)).squeeze(-1)  # (N,3) ROS 光学系
    zz = pc[:, 2].clamp_min(1e-6)
    u = K[:, 0, 0] * pc[:, 0] / zz + K[:, 0, 2]
    v = K[:, 1, 1] * pc[:, 1] / zz + K[:, 1, 2]
    return torch.stack([u, v], dim=-1), pc[:, 2]


def parcel_corners_w(pos_w, quat_w):
    """盒子 8 角点世界坐标 (N,8,3)。角点顺序与 parcel_pose 的 CORNER_SIGNS 一致。"""
    half = torch.tensor([[s / 2.0 for s in PARCEL_SIZE]], dtype=pos_w.dtype, device=pos_w.device)
    signs = torch.tensor([[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
                         dtype=pos_w.dtype, device=pos_w.device)
    local = signs.unsqueeze(0) * half.unsqueeze(1)              # (1,8,3)
    R = quat_to_matrix(quat_w)                                  # (N,3,3)
    return torch.einsum("nij,nkj->nki", R, local) + pos_w.unsqueeze(1)


def visible_fraction(uv8, width: int, height: int) -> float:
    u0, u1 = float(uv8[:, 0].min()), float(uv8[:, 0].max())
    v0, v1 = float(uv8[:, 1].min()), float(uv8[:, 1].max())
    full = (u1 - u0) * (v1 - v0)
    if full <= 1e-6:
        return 0.0
    cu = max(0.0, min(u1, width) - max(u0, 0.0))
    cv = max(0.0, min(v1, height) - max(v0, 0.0))
    return float(cu * cv / full)


def occlusion_ratio(depth_m: np.ndarray, uv8: np.ndarray, z_center: float, half_diag: float,
                    margin: float = 0.01) -> float:
    """轮廓内"比盒子近表面还近"的像素占比（与 verify_vision_data 判据 4 同公式）。"""
    h, w = depth_m.shape
    u0, u1 = int(max(uv8[:, 0].min(), 0)), int(min(uv8[:, 0].max(), w - 1))
    v0, v1 = int(max(uv8[:, 1].min(), 0)), int(min(uv8[:, 1].max(), h - 1))
    if u1 - u0 < 2 or v1 - v0 < 2:
        return float("nan")
    patch = depth_m[v0:v1 + 1, u0:u1 + 1]
    valid = patch > 0
    if valid.sum() < 10:
        return float("nan")
    return float((valid & (patch < z_center - half_diag - margin)).sum()) / float(valid.sum())


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    import importlib.metadata as metadata

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_name = args.run_name or f"posed_{args.states}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = pathlib.Path(args.out).expanduser().resolve() / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"[POSED] 目录已存在且非空，换个 --run_name：{run_dir}")

    # ------------------------------------------------------------------ 环境
    spec = gym.spec(TASK)
    _front_only = camera_keys == ["front"]
    env_cfg = make_env_cfg(
        string_to_callable(spec.kwargs["env_cfg_entry_point"]),
        pathlib.Path(_REPO_ROOT / "models" / "openarm_flip_state62_best.pt"),
        enable_cameras=True,
        lean_cameras=_front_only,
    )
    if _front_only and not args.no_depth:
        env_cfg.scene.front_cam.data_types = list(CAM_DATA_TYPES)
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device
    # 不需要策略：关掉"自动 reset"相关的回合逻辑干扰（这里只用手动 teleport）
    env_cfg.episode_length_s = 1e6

    front = env_cfg.scene.front_cam
    if front is None:
        raise SystemExit("[POSED] 场景里没有前视相机")
    if args.cam_focal is not None:
        front.spawn.focal_length = float(args.cam_focal)
    if args.cam_pos is not None:
        front.offset.pos = tuple(float(v) for v in args.cam_pos)
    if args.cam_target is not None:
        front.offset.rot = _camera_look_at_quat_xyzw(tuple(front.offset.pos), tuple(args.cam_target))
    if args.cam_res is not None:
        front.width, front.height = int(args.cam_res[0]), int(args.cam_res[1])
    cam_info = {
        "front": {
            "sensor_name": "front_cam",
            "resolution_hw": [int(front.height), int(front.width)],
            "focal_length_mm": float(front.spawn.focal_length),
            "offset_pos_env_local": [float(v) for v in front.offset.pos],
            "offset_rot_xyzw": [float(v) for v in front.offset.rot],
            "offset_convention": "ros" if getattr(front.offset, "convention", "ros") == "ros" else str(
                getattr(front.offset, "convention", "ros")),
            "mount": "static camera in front of the robot (脚本化位姿采样)",
            "data_types": list(front.data_types),
        }
    }

    print(f"[POSED] 启动环境：{args.states} 个状态 × {args.frames_per_state} 帧", flush=True)
    env = gym.make(TASK, cfg=env_cfg)
    unwrapped = env.unwrapped
    scene = unwrapped.scene
    sim = unwrapped.sim
    dt = unwrapped.step_dt
    decimation = int(unwrapped.cfg.decimation)

    parcel = scene["parcel"]
    robot = scene["robot"]
    origins = scene.env_origins
    cams = {k: scene.sensors[_CAM_SENSOR[k]] for k in camera_keys}
    all_joint_names = list(robot.data.joint_names)
    n_joints = len(all_joint_names)

    # 机械臂构型库（用于制造遮挡）：直接从已录数据集重放真实关节角
    arm_lib = None
    if args.occlusion_prob > 0:
        if not args.occlusion_source:
            raise SystemExit("[POSED] --occlusion-prob>0 时必须给 --occlusion-source <已录数据集目录>")
        src = pathlib.Path(args.occlusion_source).expanduser().resolve()
        npz = src / "frames.npz" if src.is_dir() else src
        if not npz.is_file():
            raise SystemExit(f"[POSED] 找不到 {npz}")
        with np.load(npz) as z:
            key = "joint_pos_all" if "joint_pos_all" in z.files else "joint_pos_ctrl"
            arm_lib = np.asarray(z[key], dtype=np.float32)
        print(f"[POSED] 机械臂构型库 {arm_lib.shape}（来自 {npz}，键 {key}）", flush=True)
        if arm_lib.shape[1] != n_joints and arm_lib.shape[1] != 18:
            raise SystemExit(f"[POSED] 关节维数不匹配：库里 {arm_lib.shape[1]}，场景 {n_joints}")

    # ------------------------------------------------------------------ 目录
    for key in camera_keys:
        (run_dir / "images" / key).mkdir(parents=True, exist_ok=True)
        if not args.no_depth:
            (run_dir / "depth" / key).mkdir(parents=True, exist_ok=True)
    (run_dir / "previews").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 热身
    no_action = torch.zeros(1, unwrapped.action_manager.total_action_dim, device=args.device)
    print(f"[POSED] 热身 {args.warmup} 步（RTX 降噪收敛）...", flush=True)
    for _ in range(args.warmup):
        env.step(no_action)

    # 相机尺寸（取景占比要用）：必须在至少一步之后读，否则 output 可能还是空缓冲
    cam_hw = {}
    for k, cam in cams.items():
        rgb = to_numpy(cam.data.output["rgb"])
        cam_hw[k] = (int(rgb.shape[1]), int(rgb.shape[2]))
    # 手臂的"中立"姿态 = 环境 reset 后的默认姿态。别用构型库的均值：
    # 均值姿态可能正好压在盒子上，teleport 会把盒子顶飞。
    arm_home = to_numpy(robot.data.joint_pos.torch)[0].copy()
    print(f"[POSED] 相机 {[(k, cam_hw[k]) for k in camera_keys]} | 关节 {n_joints} 个", flush=True)

    # ------------------------------------------------------------------ 采样计划
    plan_cfg = PlanConfig(
        face_probs=_face_probs,
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        grid=tuple(args.grid) if args.grid and args.grid[0] > 0 else None,
    )
    specs = plan_states(args.states, plan_cfg, seed=args.seed)
    summary = plan_summary(specs, plan_cfg)
    print(f"[POSED] 采样计划: {json.dumps(summary, ensure_ascii=False)}", flush=True)

    # ------------------------------------------------------------------ 采集
    records: dict[str, list] = {}
    frame_idx = 0
    t_sim = 0.0
    stats = {"states_done": 0, "face_reject": 0, "frame_reject": 0, "occl_reject": 0, "tries": 0}
    half_diag = float(np.linalg.norm(np.asarray(PARCEL_SIZE) / 2.0))
    rng = np.random.default_rng(args.seed)

    def push(name: str, value) -> None:
        records.setdefault(name, []).append(value)

    arm_hold_default = arm_home

    def _write_joints(pose: np.ndarray) -> None:
        hold = torch.as_tensor(pose, dtype=torch.float32, device=args.device).reshape(1, -1)
        if hold.shape[1] != n_joints:
            raise SystemExit(
                f"[POSED] 关节维数不匹配：构型库 {hold.shape[1]} vs 场景 {n_joints}。"
                "请用带 joint_pos_all 的数据集作为 --occlusion-source（不要用 joint_pos_ctrl）。"
            )
        robot.write_joint_state_to_sim(hold, torch.zeros_like(hold))

    for si, spec in enumerate(specs):
        acquired = False
        for attempt in range(max(1, args.max_tries)):
            stats["tries"] += 1
            want_arm = arm_lib is not None and rng.random() < args.occlusion_prob
            arm_pose = arm_lib[int(rng.integers(len(arm_lib)))] if want_arm else arm_hold_default

            # 顺序很重要：**先摆手臂、再放盒子** —— 否则手臂扫过去会把盒子推离目标位姿
            _write_joints(arm_pose)

            mdp.reset_parcel_root_state(
                unwrapped, torch.tensor([0], device=args.device),
                asset_cfg=SceneEntityCfg("parcel"),
                table_pos=TABLE_POS, table_size=TABLE_SIZE, table_top=TABLE_TOP,
                parcel_size=PARCEL_SIZE,
                parcel_xy=(float(spec.xy[0]), float(spec.xy[1])),
                pos_jitter=0.0,
                side_prob=1.0 if spec.face == "side" else 0.0,
                velocity_range=(0.0, 0.0),
                label_face=spec.face,
            )
            expect_up = label_up_of(spec.face)

            # 稳定 + 渲染
            for _ in range(args.settle_steps):
                _write_joints(arm_pose)   # 每步回写手臂关节，等效运动学固定，避免重力把它拉走
                sim.step()
            scene.update(dt)

            # ---- 读取真值与相机 ----
            parcel_pos_w = parcel.data.root_pos_w.torch
            parcel_quat_w = parcel.data.root_quat_w.torch
            label_normal_w = label_normal_from_quat(parcel_quat_w)
            up_now = float(label_normal_w[0, 2])
            Rq = quat_to_matrix(parcel_quat_w)
            local_x_w = Rq[..., 0]

            # ---- 校验 1：面类型没被推翻 ----
            if args.max_tries > 0:
                # side 的面法向水平（label_up≈0），up/down 是 ±1；留出落在边缘/滑动导致的容差
                ok_face = (abs(up_now) < 0.35) if spec.face == "side" else (abs(up_now - expect_up) < 0.15)
                if not ok_face:
                    stats["face_reject"] += 1
                    continue

            cam_data = {k: _read_cam(cams[k], not args.no_depth) for k in camera_keys}
            key0 = "front" if "front" in camera_keys else camera_keys[0]
            cd = cam_data[key0]
            w, h = cam_hw[key0]
            corners = parcel_corners_w(parcel_pos_w, parcel_quat_w)[0]
            uv8, z8 = project_to_pixels(corners, cd["pos_w"][0], cd["quat_w_ros"][0], cd["K"][0])
            uv8n = to_numpy(uv8)
            vis = visible_fraction(uv8n, w, h)
            center_uv, center_z = project_to_pixels(parcel_pos_w, cd["pos_w"][0], cd["quat_w_ros"][0], cd["K"][0])

            # ---- 校验 2：取景 ----
            if args.max_tries > 0 and vis < args.min_in_frame:
                stats["frame_reject"] += 1
                continue

            # ---- 校验 3：遮挡桶 ----
            occ = float("nan")
            if cd["depth"] is not None:
                occ = occlusion_ratio(cd["depth"][0, :, :, 0], uv8n, float(center_z[0]), half_diag)
                if args.max_tries > 0 and args.occlusion_prob > 0:
                    good = (occ == occ) and (args.occl_min <= occ <= args.occl_max)
                    if not good:
                        stats["occl_reject"] += 1
                        continue

            # ---- 落盘 ----
            for k in camera_keys:
                for rep in range(args.frames_per_state):
                    fn = frame_idx + rep
                    save_rgb(run_dir / "images" / k / f"{fn:06d}.png", cam_data[k]["rgb"][0])
                    if cam_data[k]["depth"] is not None:
                        save_depth_mm(run_dir / "depth" / k / f"{fn:06d}.png", cam_data[k]["depth"][0, :, :, 0])
            for rep in range(args.frames_per_state):
                push("frame_index", frame_idx + rep)
                push("env_id", 0)
                push("episode_index", si)          # 每个状态一个回合
                push("step_in_episode", rep)
                push("sim_time", float(t_sim))
                push("reset_flag", rep == 0)
                push("parcel_pos_w", to_numpy(parcel_pos_w)[0].astype(np.float32))
                push("parcel_pos_env", to_numpy(parcel_pos_w - origins)[0].astype(np.float32))
                push("env_origin", to_numpy(origins)[0].astype(np.float32))
                push("parcel_quat_w", to_numpy(parcel_quat_w)[0].astype(np.float32))
                push("label_normal_w", to_numpy(label_normal_w)[0].astype(np.float32))
                push("label_up", float(up_now))
                push("parcel_local_x_w", to_numpy(local_x_w)[0].astype(np.float32))
                push("parcel_local_y_w", to_numpy(Rq[..., 1])[0].astype(np.float32))
                push("on_table", bool(abs(float(to_numpy(parcel_pos_w - origins)[0][2]) - TABLE_TOP) < 0.10))
                push("joint_pos_all", to_numpy(robot.data.joint_pos.torch)[0].astype(np.float32))
                push("joint_vel_all", to_numpy(robot.data.joint_vel.torch)[0].astype(np.float32))
                push("sampled_face", spec.face)
                push("sampled_xy", np.asarray(spec.xy, dtype=np.float32))
                push("sampled_yaw_deg", float(spec.yaw_deg))
                push("arm_replayed", bool(want_arm))
                for k in camera_keys:
                    cd = cam_data[k]
                    push(f"{k}_pos_w", to_numpy(cd["pos_w"])[0].astype(np.float32))
                    push(f"{k}_quat_w_ros", to_numpy(cd["quat_w_ros"])[0].astype(np.float32))
                    push(f"{k}_K", to_numpy(cd["K"])[0].astype(np.float32))
                    c_uv, c_z = project_to_pixels(
                        parcel_pos_w, cd["pos_w"][0], cd["quat_w_ros"][0], cd["K"][0])
                    c_uv_n = to_numpy(c_uv)[0]
                    push(f"{k}_parcel_uv", c_uv_n.astype(np.float32))
                    push(f"{k}_parcel_depth", float(c_z[0]))
                    push(f"{k}_parcel_in_frame", bool(0 <= c_uv_n[0] < cam_hw[k][0] and 0 <= c_uv_n[1] < cam_hw[k][1]))
                    cc = parcel_corners_w(parcel_pos_w, parcel_quat_w)[0]
                    uvk, _ = project_to_pixels(cc, cd["pos_w"][0], cd["quat_w_ros"][0], cd["K"][0])
                    push(f"{k}_parcel_visible_frac", visible_fraction(to_numpy(uvk), *cam_hw[k]))
                    push(f"{k}_rendered_occluded_frac", float(occ))
                    if cd["depth"] is not None:
                        push(f"{k}_depth_valid_px", int((cd["depth"][0, :, :, 0] > 0).sum()))
            frame_idx += args.frames_per_state
            t_sim += args.settle_steps * dt
            stats["states_done"] += 1
            acquired = True
            if args.smoke or (si + 1) % 100 == 0:
                print(f"  [{si + 1}/{len(specs)}] face={spec.face} xy=({spec.xy[0]:.2f},{spec.xy[1]:.2f}) "
                      f"yaw={spec.yaw_deg:.0f}° label_up={up_now:+.2f} 取景={vis:.2f} 遮挡={occ:.2f}"
                      f"{' | 手臂遮挡' if want_arm else ''}", flush=True)
            break
        if not acquired:
            print(f"  [跳过] 状态 {si}（face={spec.face}）重试 {args.max_tries} 次仍不满足约束", flush=True)

    # ------------------------------------------------------------------ 写盘
    arrays = {k: np.asarray(v) for k, v in records.items()}
    if not arrays:
        raise SystemExit("[POSED] 没有采集到任何帧（检查采样范围/取景约束是否过严）")
    np.savez_compressed(run_dir / "frames.npz", **arrays)
    print(f"[POSED] frames.npz 已写出：{len(arrays)} 个数组 / {frame_idx} 帧", flush=True)

    info = {
        "schema_version": 1,
        "run_name": run_name,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": "SCRIPTED POSE SAMPLING (not a policy rollout): parcel teleported to sampled "
                   "poses, rendered, ground truth recorded — for perception CNN training",
        "frames": int(frame_idx),
        "num_envs": 1,
        "parcel_size_m": [float(v) for v in PARCEL_SIZE],
        "task_id": TASK,
        "sampler": {
            "plan": summary,
            "settle_steps": int(args.settle_steps),
            "frames_per_state": int(args.frames_per_state),
            "min_in_frame": float(args.min_in_frame),
            "occlusion_prob": float(args.occlusion_prob),
            "occlusion_source": str(args.occlusion_source),
            "occl_range": [float(args.occl_min), float(args.occl_max)],
            "stats": stats,
            "note": "每个状态是一个独立回合（episode_index=状态序号），所以按回合划分 == 按状态划分",
        },
        "conventions": {
            "quaternion_order": "xyzw",
            "parcel_pos_env": "world position minus env origin",
            "parcel_quat_w": "world-frame orientation of the parcel root",
            "label_normal_w": "unit vector = R(parcel_quat_w) @ (0,0,1); label_up = its z component",
            "label_up": "label_normal_w[2] (cos of tilt between label normal and world +Z)",
            "{cam}_parcel_visible_frac": "取景占比：盒子轮廓落在画面内的比例（不是遮挡）",
            "{cam}_rendered_occluded_frac": "采集时用同一深度公式算的遮挡率（供核对，训练侧会重算）",
            "image_color_order": "RGB, uint8",
            "depth_units": "millimetres, uint16 PNG (distance_to_image_plane); 0 = invalid",
        },
        "cameras": cam_info,
        "frame_arrays": sorted(arrays.keys()),
        "files": {
            "image_pattern": "images/<cam>/{frame_index:06d}.png",
            "depth_pattern": "depth/<cam>/{frame_index:06d}.png",
            "frames": "frames.npz",
        },
        "known_limitations": [
            "脚本化采样没有物理过程：盒子是被 teleport 到静止姿态的，不存在『翻起中』的中间态。"
            "中间态（label_up 在 0.3~0.9）仍需策略 rollout 补充。",
            "手臂遮挡靠重放已录构型：若重放的构型与盒子相交，盒子会被推开——"
            "采集后请用 audit_dataset.py 与遮挡分桶核对。",
        ],
    }
    (run_dir / "dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[POSED] dataset_info.json 已写出 -> {run_dir}", flush=True)
    print(f"[POSED] 统计: {json.dumps(stats, ensure_ascii=False)}", flush=True)
    print("[POSED] 下一步: python3 parcel_pose_cnn/scripts/audit_dataset.py " + str(run_dir), flush=True)


def _read_cam(cam, want_depth: bool) -> dict:
    out = cam.data.output
    rgb = to_numpy(out["rgb"])
    depth = out.get("distance_to_image_plane", None) if want_depth else None
    return {
        "rgb": rgb,
        "depth": to_numpy(depth) if depth is not None else None,
        "pos_w": to_numpy(cam.data.pos_w),
        "quat_w_ros": to_numpy(cam.data.quat_w_ros),
        "K": to_numpy(cam.data.intrinsic_matrices),
    }


if __name__ == "__main__":
    main()
