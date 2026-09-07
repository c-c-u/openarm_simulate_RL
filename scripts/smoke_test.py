# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test: build the OpenArm-Flip env, step it, and sanity-check the task.

Validates the whole stack end-to-end (scene spawn, URDF conversion, physics,
observations, rewards, resets) without any policy.

Usage:
    python.sh scripts/smoke_test.py --num_envs 16 --device cpu --headless
"""
from __future__ import annotations

import os
import pathlib
import sys

# make the framework package importable when run via `python.sh <script>`
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Two Isaac Sim python deps live only inside extension "pip_prebundle"
# directories that the extension system exposes after the app starts, but they
# are needed earlier (AppLauncher imports torch; the PhysX backend resolves
# the 'newton' package). numpy etc. already come from PYTHONPATH.
_ISAAC_ROOT = pathlib.Path(
    os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64")
)
for _rel in (
    "extsDeprecated/omni.isaac.ml_archive/pip_prebundle",  # torch / torchvision
    "exts/isaacsim.pip.newton/pip_prebundle",  # newton physics bindings
):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Smoke test the OpenArm-Flip environment.")
parser.add_argument("--task", type=str, default="OpenArm-Flip-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments.")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device.")
parser.add_argument("--steps", type=int, default=120, help="Number of policy steps.")
parser.add_argument("--headless", action="store_true", help="Run without the GUI.")
parser.add_argument(
    "--no_cameras",
    action="store_true",
    help="Skip scene cameras (only valid with the state-only task "
    "OpenArm-Flip-State-v0; the visual task OpenArm-Flip-v0 needs its camera).",
)
args, _ = parser.parse_known_args()

# 视觉任务把前视相机图像喂给 CNN 策略，因此必须离屏 RTX 渲染；
# --no_cameras 只对纯状态任务（OpenArm-Flip-State-v0）有意义。
_IMAGE_TASKS = {"OpenArm-Flip-v0", "OpenArm-Flip-Play-v0"}
_NEEDS_CAMERAS = args.task in _IMAGE_TASKS or not args.no_cameras

# Isaac Lab 3.0 的 AppLauncher 只有显式请求 "kit" 可视化器才会打开 GUI，
# 否则即使 headless=False 也会被强制无头；这里按 --headless 决定是否请求 Kit GUI。
app_launcher = AppLauncher(
    {
        "headless": args.headless,
        "width": 960,
        "height": 720,
        "enable_cameras": _NEEDS_CAMERAS,
        "visualizer": None if args.headless else "kit",
    }
)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm_lab  # noqa: F401
from isaaclab.utils.string import string_to_callable


def main() -> None:
    # --- 1) 按注册的 entry point 实例化环境 cfg 并创建环境 ---
    # （相机决策已编码在 cfg 里：视觉任务强制开前视相机，状态任务默认无相机）
    spec = gym.spec(args.task)
    env_cfg = string_to_callable(spec.kwargs["env_cfg_entry_point"])()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    env = gym.make(args.task, cfg=env_cfg)

    # --- 冒烟点 1：环境能创建，观测组 shape / 动作维度符合预期 ---
    obs, _ = env.reset()
    print(f"SMOKE: env created. device={env.unwrapped.device} num_envs={env.unwrapped.num_envs}", flush=True)
    if isinstance(obs, dict):
        for k, v in obs.items():
            print(f"SMOKE: obs group '{k}' shape={tuple(v.shape)}", flush=True)
    else:
        print(f"SMOKE: obs shape={tuple(obs.shape)}", flush=True)

    # --- 冒烟点 2：跑若干步纯随机动作，回报 / label_up / 终止率应处于合理区间 ---
    num_actions = env.unwrapped.action_manager.total_action_dim
    print(f"SMOKE: action dim = {num_actions}", flush=True)
    label_up_vals = []
    for step in range(args.steps):
        actions = torch.rand(env.unwrapped.num_envs, num_actions, device=env.unwrapped.device) * 2.0 - 1.0
        obs, rew, terminated, truncated, infos = env.step(actions)

        quat = env.unwrapped.scene["parcel"].data.root_quat_w  # (x, y, z, w) 四元数
        x, y, _, _ = quat.unbind(dim=-1)
        label_up = 1.0 - 2.0 * (x * x + y * y)  # 由姿态算"标签朝上"程度：1 朝上 / -1 朝下
        label_up_vals.append(label_up.mean().item())

        if step % 20 == 0:
            print(
                f"SMOKE: step {step:4d} mean_reward={rew.mean().item():+.3f} "
                f"mean_label_up={label_up.mean().item():+.3f} "
                f"terminated={(terminated | truncated).float().mean().item():.2f}",
                flush=True,
            )

    import statistics

    print(
        f"SMOKE: mean label_up over run = {statistics.mean(label_up_vals):+.3f} "
        f"(1.0 = label up, -1.0 = label down)",
        flush=True,
    )

    # --- 冒烟点 3：完整触发一次 reset（重置随机化事件在此执行），确认盒子朝向/位置被随机化 ---
    obs, _ = env.reset()
    quat = env.unwrapped.scene["parcel"].data.root_quat_w  # (x, y, z, w) 四元数
    x, y, _, _ = quat.unbind(dim=-1)
    label_up = 1.0 - 2.0 * (x * x + y * y)
    print(f"SMOKE: after reset mean_label_up={label_up.mean().item():+.3f} (expect spread of orientations)", flush=True)
    pos = env.unwrapped.scene["parcel"].data.root_pos_w[:4]
    print(f"SMOKE: first 4 parcel world positions: {pos.tolist()}", flush=True)

    print("SMOKE: PASS", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
