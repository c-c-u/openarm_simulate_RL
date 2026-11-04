"""测量视觉任务（CNN 策略）的仿真训练吞吐 —— 用于评估"CNN 能不能训得动"。

输出 env-steps/s、每 PPO 迭代耗时、显存峰值，并推算 20000 次迭代的总时长。
参数：
  * ``--optimize 0``：原始配置（前视 96x72 rgb+depth + 左右腕 320x240 rgb+depth）
  * ``--optimize 1``：精简配置（仅前视 rgb，等价于 env cfg 的 ``lean_cameras``）
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
_ISAAC_ROOT = pathlib.Path(os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64"))
for _rel in ("extsDeprecated/omni.isaac.ml_archive/pip_prebundle", "exts/isaacsim.pip.newton/pip_prebundle"):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=200, help="测量用的策略步数")
parser.add_argument("--optimize", type=int, default=0, help="1=去掉腕部相机与 depth")
args = parser.parse_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": True})
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm_lab  # noqa: F401
from isaaclab.utils.string import string_to_callable

spec = gym.spec("OpenArm-Flip-v0")  # 视觉任务（CNN）
env_cfg = string_to_callable(spec.kwargs["env_cfg_entry_point"])(enable_cameras=True)
env_cfg.scene.num_envs = args.num_envs
env_cfg.sim.device = "cuda:0"

if args.optimize:
    # 策略只吃 front_cam 的 rgb：去掉两个腕部相机（不喂策略）与前视的 depth 渲染
    env_cfg.scene.wrist_left_cam = None
    env_cfg.scene.wrist_right_cam = None
    env_cfg.scene.front_cam.data_types = ["rgb"]
    print("[PERF] 优化配置：仅 front_cam(rgb)，无腕部相机、无 depth", flush=True)
else:
    print("[PERF] 当前配置：front_cam(rgb+depth) + 左右腕(320x240, rgb+depth)", flush=True)

env = gym.make("OpenArm-Flip-v0", cfg=env_cfg)
unwrapped = env.unwrapped
n = env_cfg.scene.num_envs

obs = env.reset()[0]
obs_t = obs["policy"] if hasattr(obs, "keys") else obs
print(f"[PERF] obs 组: {list(obs.keys()) if hasattr(obs, 'keys') else 'tensor'}", flush=True)
if hasattr(obs, "keys") and "image" in obs.keys():
    print(f"[PERF] image 张量形状: {tuple(obs['image'].shape)}", flush=True)

torch.cuda.reset_peak_memory_stats()
# 预热（前 20 步不计时，避开编译/分配开销）
for _ in range(20):
    a = torch.randn(n, 16, device=unwrapped.device).clamp(-1, 1)
    obs, _, _, _, _ = env.step(a)

t0 = time.perf_counter()
for _ in range(args.steps):
    a = torch.randn(n, 16, device=unwrapped.device).clamp(-1, 1)
    obs, _, _, _, _ = env.step(a)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

fps = args.steps * n / dt  # env-steps/s（Isaac Lab 口径）
peak = torch.cuda.max_memory_allocated() / 1e9
reserved = torch.cuda.max_memory_reserved() / 1e9
iter_s = 24 * n / fps  # 一次 PPO 迭代 = num_steps_per_env(24) x num_envs
print("\n" + "=" * 72)
print(f"[PERF] 配置            : {'优化(仅前视rgb)' if args.optimize else '当前(含腕部相机+depth)'}")
print(f"[PERF] num_envs        : {n}")
print(f"[PERF] 实测吞吐        : {fps:,.0f} env-steps/s   ({args.steps} 步用时 {dt:.2f} s)")
print(f"[PERF] 每 PPO 迭代     : {iter_s:.2f} s   (24 步/env)")
print(f"[PERF] 显存峰值        : allocated {peak:.2f} GB / reserved {reserved:.2f} GB")
for iters in (2000, 20000):
    hours = iters * iter_s / 3600
    print(f"[PERF] 推算 {iters:5d} 迭代: {hours:5.2f} 小时")
print("=" * 72, flush=True)
env.close()
