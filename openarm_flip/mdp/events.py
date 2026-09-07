# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset / randomization events for the parcel-flip task."""
from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_matrix


def _align_rotation(label_dir: torch.Tensor, side: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Rotation (N, 3, 3) mapping the parcel's local +Z onto ``label_dir``.

    ``side`` selects a horizontal label direction (label on a side face), ``k``
    picks which of the four sides. The box is left resting on a face: for side
    labels the box tips 90 deg (rest height = sx/2), for a down label it is
    rotated 180 deg (rest height = sz/2).
    """
    n = label_dir.shape[0]
    device = label_dir.device
    eye = torch.eye(3, device=device).expand(n, 3, 3)

    def rot_x(a: torch.Tensor) -> torch.Tensor:
        r = eye.clone()
        c, s = torch.cos(a), torch.sin(a)
        r[:, 1, 1] = c
        r[:, 1, 2] = -s
        r[:, 2, 1] = s
        r[:, 2, 2] = c
        return r

    def rot_y(a: torch.Tensor) -> torch.Tensor:
        r = eye.clone()
        c, s = torch.cos(a), torch.sin(a)
        r[:, 0, 0] = c
        r[:, 0, 2] = s
        r[:, 2, 0] = -s
        r[:, 2, 2] = c
        return r

    # k: 0 -> label +X (Ry(+90)), 1 -> label -X (Ry(-90)),
    #    2 -> label +Y (Rx(-90)), 3 -> label -Y (Rx(+90))
    half = torch.full((n,), math.pi / 2.0, device=device)
    ax = torch.where(
        side & (k >= 2),
        torch.where(k == 2, -half, half),
        torch.where(side, torch.zeros(n, device=device), torch.full((n,), math.pi, device=device)),
    )
    ay = torch.where(
        side & (k < 2),
        torch.where(k == 0, half, -half),
        torch.zeros(n, device=device),
    )
    return rot_x(ax) @ rot_y(ay)


def reset_parcel_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    table_pos: tuple[float, float, float],
    table_size: tuple[float, float, float],
    table_top: float,
    parcel_size: tuple[float, float, float],
    parcel_xy: tuple[float, float] | None = None,
    pos_jitter: float = 0.03,
    side_prob: float = 0.7,
    velocity_range: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Reset the parcel to rest on the table with a random label face and yaw.

    The label (local +Z) is placed on one of the four side faces with
    probability ``side_prob``, or facing down with probability ``1 - side_prob``.
    The yaw is uniform in [0, 2*pi). The parcel is dropped with zero velocity at
    the correct rest height for the chosen face.

    ``parcel_xy`` is the env-local (x, y) the parcel is centred on (defaults to
    the table centre); small jitter is added around it.
    """
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = asset.device

    sx, sy, sz = parcel_size
    base_x = parcel_xy[0] if parcel_xy is not None else table_pos[0]
    base_y = parcel_xy[1] if parcel_xy is not None else table_pos[1]

    side = torch.rand(n, device=device) < side_prob
    k = torch.randint(0, 4, (n,), device=device)
    yaw = torch.rand(n, device=device) * 2.0 * math.pi

    # label direction in world (pre-yaw): horizontal for side faces, -Z for down
    label_dir = torch.zeros(n, 3, device=device)
    label_dir[:, 0] = torch.where(side & (k == 0), 1.0, torch.where(side & (k == 1), -1.0, 0.0))
    label_dir[:, 1] = torch.where(side & (k == 2), 1.0, torch.where(side & (k == 3), -1.0, 0.0))
    label_dir[:, 2] = torch.where(side, 0.0, -1.0)

    align = _align_rotation(label_dir, side, k)  # (N, 3, 3)

    # yaw about world Z applied after the align rotation
    cy, sy_ = torch.cos(yaw), torch.sin(yaw)
    rz = torch.zeros(n, 3, 3, device=device)
    rz[:, 0, 0] = cy
    rz[:, 0, 1] = -sy_
    rz[:, 1, 0] = sy_
    rz[:, 1, 1] = cy
    rz[:, 2, 2] = 1.0
    rot = rz @ align  # (N, 3, 3)
    # Isaac Lab sim interface uses (x, y, z, w) quaternions throughout.
    quat = quat_from_matrix(rot)  # (N, 4) in (x, y, z, w) order

    # rest height: down -> rests on label face (sz/2); side +-X -> sx/2; +-Y -> sy/2
    rest = torch.where(side, torch.where(k < 2, sx / 2.0, sy / 2.0), sz / 2.0)
    jitter = torch.rand(n, 2, device=device) * 2.0 * pos_jitter - pos_jitter
    # write_root_pose_to_sim_index expects WORLD positions: env origin + offset
    pos = env.scene.env_origins[env_ids].clone()
    pos[:, 0] += base_x + jitter[:, 0]
    pos[:, 1] += base_y + jitter[:, 1]
    pos[:, 2] += table_top + rest

    root_pose = torch.cat([pos, quat], dim=-1)
    asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(n, 6, device=device),
        env_ids=env_ids,
    )
