# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gym registration for the OpenArm parcel-flip task.

Registers the manager-based RL environments:

* ``OpenArm-Flip-v0`` — **visual** task: normalised joint positions +
  front-camera rgb image as the policy input (CNN policy). This is the
  default / intended training task.
* ``OpenArm-Flip-State-v0`` — state-only baseline (68-dim state, MLP
  policy, no cameras) for fast iteration / debugging.
* ``OpenArm-Flip-Play-v0`` — play-mode (visual) config of the task.

Each registration follows the Isaac Lab convention of
``env_cfg_entry_point`` / ``rsl_rl_cfg_entry_point`` / ``default_agent``.
"""
from __future__ import annotations

import gymnasium as gym

from . import agents  # noqa: F401  (importing registers the agents package)
from .openarm_flip_env_cfg import (
    OpenArmFlipEnvCfg,
    OpenArmFlipEnvCfg_PLAY,
    OpenArmFlipStateEnvCfg,
)
from .openarm_teleop_env_cfg import OpenArmTeleopEnvCfg

##
# Register Gym environments.
##

# 视觉任务：关节位置 + 前视相机图像（CNN 策略），即默认训练任务；
# agent 用 OpenArmFlipImgPPORunnerCfg（CNN，日志目录 openarm_flip_img）。
gym.register(
    id="OpenArm-Flip-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm_flip_env_cfg:OpenArmFlipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmFlipImgPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 纯状态基线：68 维状态、MLP 策略、无相机；
# agent 用 OpenArmFlipPPORunnerCfg（MLP，日志目录 openarm_flip）。
gym.register(
    id="OpenArm-Flip-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm_flip_env_cfg:OpenArmFlipStateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmFlipPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 视觉任务的播放配置（OpenArmFlipEnvCfg_PLAY：16 envs、15 s、只出 90° 侧面起手）
gym.register(
    id="OpenArm-Flip-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm_flip_env_cfg:OpenArmFlipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmFlipImgPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 遥操作任务（手动操作机械臂，不训练）：IK 动作 + 键盘/SpaceMouse 设备。
# 用 scripts/teleop_openarm.py 运行；后续接 XR 头显（PICO/Quest via CloudXR）时
# 沿用同一套 IK 动作项，只把输入源换成头显手柄/手部追踪。
gym.register(
    id="OpenArm-Teleop-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.openarm_teleop_env_cfg:OpenArmTeleopEnvCfg",
        # 遥操作不使用策略，这里沿用 MLP cfg 仅为满足注册惯例
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OpenArmFlipPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)

# 本框架注册的全部任务 id（供脚本判断任务类型 / 相机需求用）
OPENARM_FLIP_ENVS = {"OpenArm-Flip-v0", "OpenArm-Flip-State-v0", "OpenArm-Flip-Play-v0"}

# 遥操作任务 id（无策略、无相机、动作是 IK）
OPENARM_TELEOP_ENVS = {"OpenArm-Teleop-v0"}

# 训练日志目录名 -> 该目录下 checkpoint 对应的任务 id。
# PPO 的日志目录由 runner cfg 的 experiment_name 决定，因此从 checkpoint 路径
# （logs/rsl_rl/<experiment>/<run>/model_*.pt）即可反推它该用哪个任务加载 ——
# MLP 状态模型与 CNN 视觉模型的网络结构不同，用错任务会报 state_dict 不匹配。
# play.py / play_latest.py 共用这张表来避免手动指定 --task 出错。
EXPERIMENT_TASKS = {
    "openarm_flip": "OpenArm-Flip-State-v0",  # OpenArmFlipPPORunnerCfg（MLP，68 维状态）
    "openarm_flip_img": "OpenArm-Flip-Play-v0",  # OpenArmFlipImgPPORunnerCfg（CNN，图像观测）
}

__all__ = [
    "OpenArmFlipEnvCfg",
    "OpenArmFlipEnvCfg_PLAY",
    "OpenArmFlipStateEnvCfg",
    "OpenArmTeleopEnvCfg",
    "OPENARM_FLIP_ENVS",
    "OPENARM_TELEOP_ENVS",
    "EXPERIMENT_TASKS",
]
