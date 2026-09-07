# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the parcel-flip task."""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def parcel_label_normal_w(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("parcel")
) -> torch.Tensor:
    """World-frame unit vector of the parcel's local +Z (label) axis, shape [N, 3].

    The parcel USD is authored so that the express label sits on its local +Z
    face; rotating this vector to align with world +Z is the task goal.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w.torch  # (N, 4) in (x, y, z, w) order
    x, y, z, w = quat.unbind(dim=-1)
    nx = 2.0 * (x * z + w * y)
    ny = 2.0 * (y * z - w * x)
    nz = 1.0 - 2.0 * (x * x + y * y)
    return torch.stack([nx, ny, nz], dim=-1)
