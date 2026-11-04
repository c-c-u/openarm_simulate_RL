# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""用真实回合**实测**盒子的工作包络，并据此评估候选相机的取景覆盖率。

为什么需要它
------------
相机取景必须覆盖盒子的实际活动范围，而凭感觉设定包络会出错：曾按"x[0.24,0.52]
y[0.02,0.30]"设计相机，实测某局盒子被推到 **y=0.40**，直接越界、42% 的帧被截断。

本脚本用状态策略跑若干完整回合，记录**每个 env 每一步**的盒子位姿（长 0.09×0.09×0.07 m、
8 个角点都算上），然后：

1. 按不同**采集窗口**给出实测包络（整回合 / 未达标 / 回合前 N 步 / 达标前）；
2. 对每个**候选相机**，在这些样本上直接投影，算出"盒子完整落在画面内"的比例，
   以及盒子在画面上的像素宽度。

这样"相机怎么装""采集窗口怎么定"这两个耦合的决策都由实测数据支撑，不必再猜。

不开相机（纯状态策略），64 envs × 600 步约 35 s。

用法::

    $ISAAC_ROOT/python.sh scripts/probe_workspace.py --num_envs 64 --steps 600
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT = _REPO_ROOT / "models" / "openarm_flip_state62_best.pt"

parser = argparse.ArgumentParser(description="实测工作包络 + 评估候选相机取景")
parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=600, help="总策略步数（600 = 2 个完整回合）")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--warmup", type=int, default=8, help="开始记录前的空动作步数")
parser.add_argument("--early", type=int, default=60, help="『回合前 N 步』窗口的 N")
parser.add_argument("--out", type=str, default="/home/blanc/isaacsim/perception_data/workspace_envelope.json")
args = parser.parse_args()

# Isaac Sim 的 pip_prebundle 前置（必须在 import isaaclab / numpy / torch 之前）
_ISAAC_ROOT = pathlib.Path(
    os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64")
)
for _rel in ("extsDeprecated/omni.isaac.ml_archive/pip_prebundle", "exts/isaacsim.pip.newton/pip_prebundle"):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": False})
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import openarm_lab  # noqa: F401,E402
from isaaclab.utils.string import string_to_callable  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from openarm_lab.openarm_flip.checkpoint_utils import make_env_cfg  # noqa: E402
from openarm_lab.openarm_flip.openarm_flip_env_cfg import PARCEL_SIZE  # noqa: E402

TASK = "OpenArm-Flip-State-v0"
UP_COS = math.cos(math.radians(15))
APERTURE = 20.955  # 水平像幅（mm），与场景相机一致
SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
HALF = np.array(PARCEL_SIZE) / 2.0
CORNERS = SIGNS * HALF  # (8, 3) 盒子局部角点

# 候选相机：(标签, 位置, 注视点, 焦距mm, 宽, 高)
CANDIDATES = [
    ("A 现状 yaw31  12mm", (0.15, 0.00, 0.72), (0.35, 0.12, 0.42), 12.0, 640, 480),
    ("B yaw0       12mm", (0.10, 0.14, 0.72), (0.35, 0.14, 0.42), 12.0, 640, 480),
    ("C 中线 yaw0   12mm", (0.15, 0.00, 0.72), (0.35, 0.00, 0.42), 12.0, 640, 480),
    ("D yaw0      8.5mm", (0.10, 0.14, 0.72), (0.35, 0.14, 0.42), 8.5, 640, 480),
    ("E yaw0        6mm", (0.10, 0.14, 0.72), (0.30, 0.20, 0.42), 6.0, 640, 480),
    ("F 高俯视 yaw0 25mm", (0.10, 0.18, 1.15), (0.30, 0.18, 0.42), 25.0, 640, 480),
    ("G 高俯视 yaw0 16mm", (0.10, 0.18, 0.95), (0.30, 0.18, 0.42), 16.0, 640, 480),
    ("H yaw0 320x240 12mm", (0.10, 0.14, 0.72), (0.35, 0.14, 0.42), 12.0, 320, 240),
]


def rot_of_quat_np(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def look_at_ros(eye: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """返回相机->世界的旋转，列向量为 (右, 下, 前)（ROS 光学约定）。"""
    f_view = eye - tgt
    f_view = f_view / np.linalg.norm(f_view)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, f_view)
    right = right / np.linalg.norm(right)
    up2 = np.cross(f_view, right)
    return np.column_stack([right, -up2, -f_view])


def eval_camera(pos, tgt, focal, w, h, pos_samples, quat_samples) -> dict:
    """在实测样本上评估：盒子 8 角点是否完整落在画面内。"""
    pos = np.asarray(pos, dtype=np.float64)
    R = look_at_ros(pos, np.asarray(tgt, dtype=np.float64))
    fx = focal / APERTURE * w
    cx, cy = w / 2.0, h / 2.0
    Rq = np.array([rot_of_quat_np(q) for q in quat_samples])  # (N,3,3)
    corners_w = pos_samples[:, None, :] + np.einsum("nij,kj->nki", Rq, CORNERS)  # (N,8,3)
    pc = (corners_w - pos) @ R                   # (N,8,3) 相机系
    z = pc[..., 2]
    behind = (z <= 0.02).any(axis=1)
    u = fx * pc[..., 0] / np.where(np.abs(z) < 1e-6, 1e-6, z) + cx
    v = fx * pc[..., 1] / np.where(np.abs(z) < 1e-6, 1e-6, z) + cy
    inside = (u.min(axis=1) >= 0) & (u.max(axis=1) <= w) & (v.min(axis=1) >= 0) & (v.max(axis=1) <= h)
    full = inside & ~behind
    width_px = np.where(behind, np.nan, u.max(axis=1) - u.min(axis=1))
    dist = float(np.linalg.norm(np.asarray(tgt, dtype=np.float64) - pos))
    dv = (np.asarray(tgt, dtype=np.float64) - pos) / dist
    return {
        "full_visible_rate": float(full.mean()),
        "box_width_px_median": float(np.nanmedian(width_px)),
        "fov_h_deg": round(2 * math.degrees(math.atan(APERTURE / (2 * focal))), 1),
        "fov_v_deg": round(2 * math.degrees(math.atan((h / w) * APERTURE / (2 * focal))), 1),
        "pitch_deg": round(math.degrees(math.asin(np.clip(-dv[2], -1, 1))), 1),
        "yaw_deg": round(math.degrees(math.atan2(dv[1], dv[0])), 1),
        "distance_m": round(dist, 3),
    }


def main() -> None:
    import importlib.metadata as metadata

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    spec = gym.spec(TASK)
    env_cfg = make_env_cfg(string_to_callable(spec.kwargs["env_cfg_entry_point"]), checkpoint, enable_cameras=False)
    agent_cfg = string_to_callable(spec.kwargs["rsl_rl_cfg_entry_point"])()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, str(metadata.version("rsl-rl-lib")))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    env = gym.make(TASK, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    parcel = unwrapped.scene["parcel"]
    origins = unwrapped.scene.env_origins
    device = unwrapped.device
    n = unwrapped.num_envs

    obs, _ = env.reset()
    zero = torch.zeros(n, unwrapped.action_manager.total_action_dim, device=device)
    for _ in range(args.warmup):
        env.step(zero)

    print(f"[PROBE] {n} envs × {args.steps} 步（不开相机）...", flush=True)
    t0 = time.time()
    P, Q, LU, EP, ST = [], [], [], [], []
    first_success = np.full(n, -1, dtype=np.int64)
    step = 0
    while step < args.steps:
        with torch.no_grad():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)
        step += 1
        p = (parcel.data.root_pos_w.torch - origins).detach().cpu().numpy()
        q = parcel.data.root_quat_w.torch.detach().cpu().numpy()
        lu = 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)
        first_success[(lu > UP_COS) & (first_success < 0)] = step
        P.append(p)
        Q.append(q)
        LU.append(lu)
        EP.append(unwrapped.episode_length_buf.detach().cpu().numpy().copy())
        ST.append(np.full(n, step, dtype=np.int64))
    elapsed = time.time() - t0

    P = np.stack(P).reshape(-1, 3)
    Q = np.stack(Q).reshape(-1, 4)
    LU = np.stack(LU).reshape(-1)
    EP = np.stack(EP).reshape(-1)
    ST = np.stack(ST).reshape(-1)
    FS = np.repeat(first_success, args.steps)  # 每个样本对应的"该 env 首次达标步"
    print(f"[PROBE] 完成：{len(P)} 个样本，用时 {elapsed:.1f} s（{args.steps / elapsed:.0f} 步/s）", flush=True)

    # 有效样本：盒子还在桌面上方（掉下桌的极端值不该决定相机取景）+ 跳过热身
    valid = (P[:, 2] > 0.35) & (EP > args.warmup)
    conditions = {
        "整回合": valid,
        f"回合前 {args.early} 步": valid & (ST <= args.early),
        "未达标(label_up<cos15)": valid & (LU < UP_COS),
        f"未达标 且 前 {args.early} 步": valid & (LU < UP_COS) & (ST <= args.early),
        "达标前(含达标步)": valid & ((FS < 0) | (ST <= FS)),
        f"达标前 且 前 {args.early} 步": valid & (((FS < 0) | (ST <= FS))) & (ST <= args.early),
    }

    print("\n【实测工作包络】盒子中心 1–99 百分位（括号内为含盒子体积后的范围）")
    envelopes: dict[str, dict] = {}
    for name, m in conditions.items():
        pts = P[m]
        if pts.shape[0] < 50:
            print(f"  {name:26s} 样本过少（{pts.shape[0]}），跳过")
            continue
        lo = np.percentile(pts, 1, axis=0)
        hi = np.percentile(pts, 99, axis=0)
        d = float(np.linalg.norm(np.percentile(pts, 99, axis=0) - np.percentile(pts, 1, axis=0)))
        hd = float(np.linalg.norm(HALF))
        print(f"  {name:26s} {pts.shape[0]:6d} 样本 | x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] "
              f"z[{lo[2]:.3f},{hi[2]:.3f}]")
        print(f"  {'':26s} 含体积: x[{lo[0] - hd:.3f},{hi[0] + hd:.3f}] y[{lo[1] - hd:.3f},{hi[1] + hd:.3f}]")
        envelopes[name] = {
            "n_samples": int(pts.shape[0]),
            "center_p1_p99": {"x": [float(lo[0]), float(hi[0])], "y": [float(lo[1]), float(hi[1])],
                              "z": [float(lo[2]), float(hi[2])]},
            "with_box_extent": {"x": [float(lo[0] - hd), float(hi[0] + hd)],
                                "y": [float(lo[1] - hd), float(hi[1] + hd)],
                                "z": [float(lo[2] - hd), float(hi[2] + hd)]},
        }

    print("\n【候选相机 × 采集窗口】盒子『完整落在画面内』的比例（+ 盒子像素宽度中位）")
    header = "  " + f"{'候选相机':<22}" + "".join(f"{k[:12]:>14}" for k in conditions)
    print(header)
    cam_results: dict[str, dict] = {}
    for label, pos, tgt, focal, w, h in CANDIDATES:
        row = []
        res = {}
        for name, m in conditions.items():
            if m.sum() < 50:
                row.append("—")
                continue
            r = eval_camera(pos, tgt, focal, w, h, P[m], Q[m])
            res[name] = r
            row.append(f"{100 * r['full_visible_rate']:5.1f}%/{r['box_width_px_median']:.0f}px")
        cam_results[label] = {"config": {"pos": list(pos), "target": list(tgt), "focal_mm": focal,
                                         "res": [w, h]}, "per_window": res}
        print(f"  {label:<22}" + "".join(f"{c:>14}" for c in row))
    print("\n  格式 = 完整可见率 / 盒子像素宽度中位")

    for label, _, tgt, focal, w, h in CANDIDATES:
        c = cam_results[label]["config"]
        d = float(np.linalg.norm(np.asarray(tgt) - np.asarray(c["pos"])))
        dv = (np.asarray(tgt) - np.asarray(c["pos"])) / d
        print(f"  {label:<22} fov {2 * math.degrees(math.atan(APERTURE / (2 * focal))):5.1f}° "
              f"pitch {math.degrees(math.asin(-dv[2])):5.1f}° yaw {math.degrees(math.atan2(dv[1], dv[0])):5.1f}° "
              f"距离 {d:.2f} m")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(checkpoint), "num_envs": int(n), "steps": int(args.steps),
        "parcel_size_m": list(PARCEL_SIZE), "envelopes": envelopes, "cameras": cam_results,
        "note": "完整可见率 = 盒子 8 角点全部落在画面内的样本比例；按采集窗口分别统计",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[PROBE] 已写出 {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
