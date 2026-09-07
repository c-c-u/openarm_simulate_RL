# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained checkpoint of the OpenArm parcel-flip policy.

Usage:
    python.sh scripts/play.py --checkpoint logs/rsl_rl/openarm_flip/<run>/model_5000.pt \
        --task OpenArm-Flip-Play-v0 --num_envs 8
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

parser = argparse.ArgumentParser(description="Play a trained OpenArm parcel-flip policy (RSL-RL).")
parser.add_argument("--task", type=str, default="OpenArm-Flip-Play-v0", help="Gym task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a model_*.pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments.")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device (cpu or cuda:N).")
parser.add_argument("--steps", type=int, default=600, help="Number of policy steps to run.")
parser.add_argument("--seed", type=int, default=0, help="Seed for the env.")
parser.add_argument("--headless", action="store_true", help="Run without the GUI.")
parser.add_argument(
    "--cam",
    type=str,
    default="",
    help="Camera preset for the GUI viewport: 'top' (straight above the parcel, "
    "label clearly visible), 'side' (classic 3/4 view), 'follow' (track the "
    "parcel). Default: no override (Kit default).",
)
args, _ = parser.parse_known_args()

# Isaac Lab 3.0 的 AppLauncher 只有显式请求 "kit" 可视化器才会打开 GUI，
# 否则即使 headless=False 也会被强制无头；这里按 --headless 决定是否请求 Kit GUI。
app_launcher = AppLauncher(
    {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "enable_cameras": True,
        "visualizer": None if args.headless else "kit",
    }
)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import openarm_lab  # noqa: F401
from isaaclab.utils.string import string_to_callable
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main() -> None:
    # --- 1) 解析任务：从 gym spec 拿到注册时绑定的 env/agent cfg 类并实例化 ---
    spec = gym.spec(args.task)
    env_cfg = string_to_callable(spec.kwargs["env_cfg_entry_point"])()
    agent_cfg = string_to_callable(spec.kwargs["rsl_rl_cfg_entry_point"])()
    # --- 2) 兼容清理：去掉本机安装的 rsl-rl 版本已弃用的模型字段 ---
    import importlib.metadata as metadata
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    # 传入纯版本号字符串即可，解析由 handle_deprecated_rsl_rl_cfg 内部完成
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, str(metadata.version("rsl-rl-lib")))
    # 命令行覆盖：环境数 / 设备 / 种子
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    # --- 3) 创建环境并包上 rsl-rl 向量化包装 ---
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- 4) （可选）GUI 相机预设：把 Kit 视口对准 env0 的盒子（无头或未指定则跳过） ---
    if not args.headless and args.cam:
        try:
            sim = env.unwrapped.sim
            origin = env.unwrapped.scene.env_origins[0].cpu().numpy()  # env0 原点
            # 盒子在 env-local (0.30, 0.10)，高约 0.40 m：以它为注视点
            target = (float(origin[0] + 0.30), float(origin[1] + 0.10), 0.42)
            if args.cam == "top":
                eye = (float(origin[0] + 0.30), float(origin[1] + 0.10), 1.25)  # 正上方俯视，标签最清楚
            elif args.cam == "side":
                eye = (float(origin[0] + 0.65), float(origin[1] - 0.45), 0.85)  # 经典 3/4 视角
            elif args.cam == "follow":
                # 较高机位 3/4 视角，仍能看到盒顶标签
                eye = (float(origin[0] + 0.45), float(origin[1] + 0.35), 1.1)
            else:
                raise SystemExit(f"Unknown --cam '{args.cam}' (choose 'top', 'side', 'follow')")
            sim.set_camera_view(eye, target)
            print(f"[PLAY] camera preset '{args.cam}': eye={tuple(round(v,2) for v in eye)} "
                  f"target={tuple(round(v,2) for v in target)}", flush=True)
        except Exception as e:  # 仅 GUI 增强功能，失败不影响回放
            print(f"[PLAY] camera preset failed (ignored): {e}", flush=True)

    # --- 5) 通过 runner 加载 checkpoint（兼容 rsl-rl 存档格式）并取推理策略 ---
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
    print(f"[PLAY] loading checkpoint {args.checkpoint}")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- 6) 无梯度回放：每步统计"标签朝上"的环境步数 ---
    obs = env.reset()[0]
    successes = 0
    total_steps = 0
    with torch.no_grad():
        for _ in range(args.steps):
            actions = policy(obs)
            # RslRlVecEnvWrapper 返回 rsl-rl 的四元组 (obs, rew, done, infos)
            obs, rew, done, infos = env.step(actions)
            total_steps += 1
            # 直接从盒子姿态算 label_up（四元数 (x,y,z,w)，z 分量 → 标签朝上程度）
            quat = env.unwrapped.scene["parcel"].data.root_quat_w.torch
            x, y, _, _ = quat.unbind(dim=-1)
            label_up = 1.0 - 2.0 * (x * x + y * y)
            successes += int((label_up > 0.966).sum().item())  # >cos(15°) 视为"朝上"
            if total_steps % 50 == 0:
                print(f"[PLAY] step {total_steps:4d}  cumulative label-up env-steps: {successes}", flush=True)

    print(f"[PLAY] done after {total_steps} steps, {successes} label-up env-steps", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
