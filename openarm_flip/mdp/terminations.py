# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the parcel-flip task."""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def parcel_out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("parcel"),
    in_bound_range: dict[str, tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Terminate when the parcel leaves the allowed box (fell off the table etc.).

    ``in_bound_range`` maps ``"x"``/``"y"``/``"z"`` to (min, max) in the
    env-local frame; axes not listed are unchecked.
    """
    if in_bound_range is None:
        in_bound_range = {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 1.0)}
    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w.torch - env.scene.env_origins  # env-local [N, 3]
    out = torch.zeros(pos.shape[0], dtype=torch.bool, device=pos.device)
    for i, key in enumerate(("x", "y", "z")):
        if key not in in_bound_range:
            continue
        lo, hi = in_bound_range[key]
        out = out | (pos[:, i] < lo) | (pos[:, i] > hi)
    return out


def nan_state_guard(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    parcel_cfg: SceneEntityCfg = SceneEntityCfg("parcel"),
) -> torch.Tensor:
    """Terminate environments whose physics state contains NaN.

    The bimanual arm occasionally drives a joint (or the parcel) into a
    divergent contact state during exploration; without this guard the NaN
    propagates into the observation buffer and kills training. Because the RL
    env resets terminated environments *before* computing the returned
    observations, terminating here yields a clean post-reset observation.
    """
    robot = env.scene[robot_cfg.name]
    parcel = env.scene[parcel_cfg.name]
    bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    bad |= torch.isnan(robot.data.joint_pos.torch).any(dim=1)
    bad |= torch.isnan(robot.data.joint_vel.torch).any(dim=1)
    bad |= torch.isnan(parcel.data.root_pos_w.torch).any(dim=1)
    bad |= torch.isnan(parcel.data.root_quat_w.torch).any(dim=1)
    return bad
