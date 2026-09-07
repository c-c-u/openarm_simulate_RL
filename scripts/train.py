# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the OpenArm parcel-flip policy with RSL-RL (PPO).

Usage:
    python.sh scripts/train.py --task OpenArm-Flip-v0 \
        --num_envs 256 --device cpu --headless

    # GPU (once the PhysX GPU tensor pipeline is available on the machine):
    python.sh scripts/train.py --num_envs 2048 --device cuda:0

Resume:
    python.sh scripts/train.py --checkpoint logs/rsl_rl/openarm_flip/<run>/model_2000.pt
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
import datetime

from isaaclab.app import AppLauncher

# launch Isaac Sim (must happen before any isaacsim imports)
parser = argparse.ArgumentParser(description="Train the OpenArm parcel-flip policy (RSL-RL PPO).")
parser.add_argument("--task", type=str, default="OpenArm-Flip-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments.")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device (cpu or cuda:N).")
parser.add_argument("--max_iterations", type=int, default=None, help="Override max PPO iterations.")
parser.add_argument("--seed", type=int, default=None, help="Seed for env and agent.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model_*.pt to resume from.")
parser.add_argument("--headless", action="store_true", help="Run without the GUI.")
parser.add_argument(
    "--disable_cameras",
    action="store_true",
    help="Skip the scene cameras. Only the state-only task OpenArm-Flip-State-v0 "
    "runs without cameras; the visual task OpenArm-Flip-v0 needs its front camera "
    "for the image observation (this flag is ignored there).",
)
args, _ = parser.parse_known_args()

# 视觉任务把前视相机图像喂给 CNN 策略，因此必须离屏 RTX 渲染；
# 纯状态任务（OpenArm-Flip-State-v0）本来就默认无相机，--disable_cameras 只是那里的
# 无操作选项（视觉任务上会被忽略并告警）。
_IMAGE_TASKS = {"OpenArm-Flip-v0", "OpenArm-Flip-Play-v0"}
if args.disable_cameras and args.task in _IMAGE_TASKS:
    print(f"[WARN] --disable_cameras is ignored for task '{args.task}': it needs its camera.", flush=True)
_NEEDS_CAMERAS = args.task in _IMAGE_TASKS

# Isaac Lab 3.0 的 AppLauncher 只有显式请求 "kit" 可视化器（visualizer="kit"）才会打开
# Kit GUI，否则即使 headless=False 也会被强制无头。这里：非 --headless 时请求 Kit GUI。
app_launcher = AppLauncher(
    {
        "headless": args.headless,
        "width": 960,
        "height": 720,
        # offscreen RTX rendering when the scene cameras are enabled
        "enable_cameras": _NEEDS_CAMERAS,
        "visualizer": None if args.headless else "kit",
    }
)
simulation_app = app_launcher.app

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import openarm_lab  # noqa: F401  (registers OpenArm-Flip-* tasks)
from isaaclab.utils.string import string_to_callable
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def main() -> None:
    # --- 1) 解析任务：从 gym spec 拿到注册时绑定的 env/agent cfg 类并实例化 ---
    spec = gym.spec(args.task)
    # cfg 类本身已编码相机决策（视觉任务强制开前视相机；状态任务默认无相机）
    env_cfg = string_to_callable(spec.kwargs["env_cfg_entry_point"])()
    agent_cfg = string_to_callable(spec.kwargs["rsl_rl_cfg_entry_point"])()
    # --- 2) 兼容清理：去掉本机安装的 rsl-rl 版本中已弃用的模型字段 ---
    import importlib.metadata as metadata
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    # 传入纯版本号字符串即可，解析由 handle_deprecated_rsl_rl_cfg 内部完成
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, str(metadata.version("rsl-rl-lib")))

    # --- 3) 命令行覆盖默认配置（环境数 / 设备 / 种子 / 最大迭代数） ---
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    if args.seed is not None:
        env_cfg.seed = args.seed
        agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    # --- 4) 生成日志目录 logs/rsl_rl/<experiment_name>/<时间戳>（tensorboard + checkpoint 落这里） ---
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = os.path.join(log_root, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # --- 5) 创建环境并包上 rsl-rl 向量化包装（clip_actions 限制动作幅度） ---
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --- 6) 创建 PPO runner；给了 --checkpoint 则加载该模型（用于续训） ---
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args.checkpoint:
        print(f"[INFO] Loading model checkpoint from: {args.checkpoint}")
        runner.load(args.checkpoint)

    # --- 7) 开始训练；捕获 NaN 报错时打印物理量诊断（定位是哪个关节/盒子发散） ---
    try:
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=agent_cfg.init_at_random_ep_len,
        )
    except ValueError as exc:
        # diagnostic aid: on NaN, dump the physical state to find the culprit
        if "NaN" in str(exc):
            scene = env.unwrapped.scene
            jp = scene["robot"].data.joint_pos.torch
            jv = scene["robot"].data.joint_vel.torch
            pq = scene["parcel"].data.root_quat_w.torch
            pp = scene["parcel"].data.root_pos_w.torch
            print(f"[DIAG] NaN dump: joint_pos NaN={torch.isnan(jp).any().item()} "
                  f"joint_vel NaN={torch.isnan(jv).any().item()} "
                  f"parcel_quat NaN={torch.isnan(pq).any().item()} "
                  f"parcel_pos NaN={torch.isnan(pp).any().item()}", flush=True)
            bad = torch.isnan(jp).any(dim=0)
            if bad.any():
                names = scene["robot"].dof_names
                print(f"[DIAG] NaN joint pos cols: {[names[i] for i in torch.nonzero(bad).flatten().tolist()]}", flush=True)
            print(f"[DIAG] joint_pos range (finite): {jp[torch.isfinite(jp)].min().item():.3f}..{jp[torch.isfinite(jp)].max().item():.3f}", flush=True)
            print(f"[DIAG] joint_vel range (finite): {jv[torch.isfinite(jv)].min().item():.3f}..{jv[torch.isfinite(jv)].max().item():.3f}", flush=True)
        raise
    # 训练完 / 异常都要关闭环境，释放仿真资源
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
