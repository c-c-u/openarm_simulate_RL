"""回放训练好的 checkpoint 并测量真实任务表现（成功率等指标，全程无头）。

成功判据 = 盒子标签朝上（标签法向与 +Z 夹角 < 15°）**且**盒子仍在桌上；
盒子掉下桌即使标签朝上也不算成功。适合批量评估 / 对比 checkpoint。
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

_ISAAC_ROOT = pathlib.Path(
    os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64")
)
for _rel in (
    "extsDeprecated/omni.isaac.ml_archive/pip_prebundle",
    "exts/isaacsim.pip.newton/pip_prebundle",
):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="OpenArm-Flip-v0",
                    help="Gym task id (OpenArm-Flip-v0 = visual/CNN; OpenArm-Flip-State-v0 = state-only/MLP).")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=10, help="episodes per env")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--cameras", action="store_true", help="Enable scene cameras (default off: state-only eval).")
args = parser.parse_known_args()[0]

# 视觉任务把前视相机图像喂给 CNN 策略，因此必须离屏 RTX 渲染；
# --cameras 只对纯状态任务有意义（默认关）。
_IMAGE_TASKS = {"OpenArm-Flip-v0", "OpenArm-Flip-Play-v0"}
_NEEDS_CAMERAS = args.task in _IMAGE_TASKS or args.cameras

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": _NEEDS_CAMERAS})
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import openarm_lab  # noqa: F401
from isaaclab.utils.string import string_to_callable
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from openarm_lab.openarm_flip.openarm_flip_env_cfg import (
    TABLE_POS,
    TABLE_SIZE,
    TABLE_TOP,
)

UP_COS = float(__import__("math").cos(__import__("math").radians(15)))  # cos(15°)：判定"标签朝上"的角度阈值


def _on_table(pos: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
    """盒子中心是否仍在桌面上方 [N]（与训练时的 parcel_on_table 判据一致）。"""
    p = pos - origins  # env-local
    inside = (
        (p[:, 0] - TABLE_POS[0]).abs() < TABLE_SIZE[0] / 2.0 - 0.03
    ) & ((p[:, 1] - TABLE_POS[1]).abs() < TABLE_SIZE[1] / 2.0 - 0.03)
    above = p[:, 2] > TABLE_TOP - 0.08
    return inside & above


def main() -> None:
    import importlib.metadata as metadata

    # --- 1) 解析任务注册的 env/agent cfg 并实例化（相机按 _NEEDS_CAMERAS 决定） ---
    spec = gym.spec(args.task)
    env_cfg = string_to_callable(spec.kwargs["env_cfg_entry_point"])(enable_cameras=_NEEDS_CAMERAS)
    agent_cfg = string_to_callable(spec.kwargs["rsl_rl_cfg_entry_point"])()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, str(metadata.version("rsl-rl-lib")))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    # 关掉基于时间的终止，让每个回合跑满完整预算（否则超时重置会提前截断统计）
    env_cfg.episode_length_s = 12.0

    # --- 2) 创建环境 + rsl-rl 向量化包装，加载 checkpoint 取推理策略 ---
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
    print(f"loading {args.checkpoint}")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --- 3) 统计量：成功率（标签朝上且在桌上）、最大 label_up、臂动作占比、盒子被推动占比 ---
    max_steps = int(env_cfg.episode_length_s / env.unwrapped.step_dt)
    successes = 0
    solved_once = 0
    max_label_up = -1.0
    episodes = args.num_envs * args.episodes
    moved_parcel_eps = 0
    arm_moved = 0.0

    # --- 4) 逐 env 跑 args.episodes 个回合，全程无梯度回放并累计统计量 ---
    for ep in range(args.episodes):
        obs = env.reset()[0]
        parcel_start = env.unwrapped.scene["parcel"].data.root_pos_w.torch.clone()
        ep_success = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
        for step in range(max_steps):
            actions = policy(obs).detach()
            obs, rew, done, infos = env.step(actions)
            quat = env.unwrapped.scene["parcel"].data.root_quat_w.torch
            ppos = env.unwrapped.scene["parcel"].data.root_pos_w.torch
            x, y, _, _ = quat.unbind(dim=-1)
            label_up = 1.0 - 2.0 * (x * x + y * y)  # 直接由盒子姿态算"标签朝上"程度
            max_label_up = max(max_label_up, label_up.max().item())
            # 成功判据：标签朝上（>cos15°）**且**盒子仍在桌上——掉下桌不算成功
            on_table = _on_table(ppos, env.unwrapped.scene.env_origins)
            ep_success |= (label_up > UP_COS) & on_table
            # 取回合中点的盒子位移统计"被推动"（避免回合结束重置造成的假象）
            if step == max_steps // 2:
                mid = env.unwrapped.scene["parcel"].data.root_pos_w.torch
                moved = (mid - parcel_start).norm(dim=-1) > 0.01
                moved_parcel_eps += int(moved.sum().item())
        successes += int(ep_success.sum().item())
        # 统计该回合"机械臂有明显动作"的环境占比（相对 home 位姿位移 > 0.1 rad）
        jp = env.unwrapped.scene["robot"].data.joint_pos.torch
        home = env.unwrapped.scene["robot"].data.default_joint_pos.torch
        arm_moved += float(((jp - home).abs() > 0.1).any(dim=1).float().mean().item())

    print(f"\n=== 模型回放结果 ({episodes} episodes, {args.num_envs} envs x {args.episodes}) ===")
    print(f"成功翻到标签朝上且盒子在桌上的 episode 数 : {successes} / {episodes}  ({100.0*successes/episodes:.1f}%)")
    print(f"回放中达到的最大 label_up     : {max_label_up:+.3f}  (1.0 = 完全朝上)")
    print(f"机械臂有明显动作的占比         : {100.0*arm_moved/args.episodes:.1f}%")
    print(f"盒子被推动超过 1cm 的占比      : {100.0*moved_parcel_eps/episodes:.1f}%")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
