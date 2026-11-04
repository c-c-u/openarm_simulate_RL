# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""checkpoint 辅助工具：读取策略的观测维度，并据此配置环境。

背景
----
状态任务的观测有两套排布（见 ``openarm_flip_env_cfg.py``）：

* **62 维**（默认）：``joint_pos(18) + joint_vel(18) + parcel_pos(3) +
  parcel_quat(4) + label_normal(3) + last_action(16)`` —— 为 sim-to-real 设计，
  不含盒子速度（真机最难测的量）。
* **68 维**（旧）：额外含 ``parcel_lin_vel(3) + parcel_ang_vel(3)``，
  2026-09 及之前训练的所有 checkpoint 都用这一版。

用错排布会让 rsl-rl 报 state_dict 尺寸不匹配，或更糟——静默地把权重对上错误的
观测位置。因此各脚本在加载 checkpoint 前，先用 :func:`obs_dim_from_checkpoint`
读出它需要的维度，再用 :func:`configure_env_cfg_for_checkpoint` 设置
``env_cfg.use_parcel_vel``。

本模块只依赖 torch（不 import isaaclab），可在 AppLauncher 之前或之后调用。
"""
from __future__ import annotations

import pathlib

# 状态观测两种维度的约定
OBS_DIM_62 = 62  # 默认：不含盒子速度
OBS_DIM_68 = 68  # 旧版：含盒子线速度+角速度

# 视觉任务的 policy 组只有关节位置（18 维），用于识别"这是视觉 checkpoint"
OBS_DIM_VISION = 18


def obs_dim_from_checkpoint(checkpoint: str | pathlib.Path, *, key: str = "actor_state_dict") -> int:
    """读取 checkpoint 里策略第一层的输入维度（即观测维度）。

    Args:
        checkpoint: ``model_*.pt`` 路径。
        key: 权重字典的键名（actor 或 critic）。

    Returns:
        观测维度（62 / 68 / 视觉任务的 18）。
    """
    import torch

    sd = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if key not in sd:
        raise KeyError(f"checkpoint {checkpoint} has no '{key}'; keys = {sorted(sd.keys())}")
    weight = sd[key]["mlp.0.weight"]
    return int(weight.shape[1])


def configure_env_cfg_for_checkpoint(env_cfg, checkpoint: str | pathlib.Path) -> int:
    """按 checkpoint 的观测维度**就地修正**已实例化的 env_cfg。

    .. warning::
       ``env_cfg.__post_init__`` 在**实例化时**就已经决定了观测配置，所以事后只改
       ``use_parcel_vel`` 标志是**无效**的（观测维度不会变）。本函数因此还会重新
       赋值 ``env_cfg.observations``。新代码建议直接用 :func:`make_env_cfg`，在构造
       时就传入该标志。

    Returns:
        检测到的观测维度。
    """
    dim = obs_dim_from_checkpoint(checkpoint)
    if getattr(env_cfg, "enable_image_obs", False):
        # 视觉任务的 policy 组只含关节位置，与 use_parcel_vel 无关
        print(f"[CKPT] checkpoint obs dim = {dim}（视觉策略）-> 不改观测配置", flush=True)
        return dim

    from .openarm_flip_env_cfg import StateObservationsCfg, StateObservationsFullCfg

    if dim == OBS_DIM_68:
        env_cfg.use_parcel_vel = True
        env_cfg.observations = StateObservationsFullCfg()
        print("[CKPT] checkpoint obs dim = 68（含盒子速度）-> 使用 68 维观测", flush=True)
    else:
        env_cfg.use_parcel_vel = False
        env_cfg.observations = StateObservationsCfg()
        print(f"[CKPT] checkpoint obs dim = {dim} -> 使用 62 维观测", flush=True)
    return dim


def make_env_cfg(cfg_factory, checkpoint: str | pathlib.Path | None = None, **kwargs):
    """按 checkpoint 的观测维度构造 env cfg（推荐入口）。

    ``use_parcel_vel`` 必须在**构造时**传入，才能让 ``__post_init__`` 选中正确的
    观测组 —— 事后改标志不会生效。

    Args:
        cfg_factory: 任务注册里的 ``env_cfg_entry_point`` 解析出的 cfg 类。
        checkpoint: checkpoint 路径；为 None 时用默认（62 维）。
        **kwargs: 传给 cfg 构造函数的其它参数（如 ``enable_cameras``）。

    Returns:
        实例化好的 env cfg。
    """
    if checkpoint is not None:
        dim = obs_dim_from_checkpoint(checkpoint)
        if dim == OBS_DIM_68:
            kwargs["use_parcel_vel"] = True
        elif dim == OBS_DIM_62:
            kwargs["use_parcel_vel"] = False
        # 其它维度（如视觉任务的 18）保持默认，由 enable_image_obs 决定观测组
        print(
            f"[CKPT] checkpoint obs dim = {dim} -> 构造 env cfg 时 use_parcel_vel="
            f"{kwargs.get('use_parcel_vel', '默认(False)')}",
            flush=True,
        )
    return cfg_factory(**kwargs)
