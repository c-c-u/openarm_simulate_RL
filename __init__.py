# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab training framework for the OpenArm parcel-flip task.

This package registers the parcel-flip tasks (manager-based RL envs) and
provides the PPO training / play entry points under :mod:`openarm_lab.scripts`.

Task: with the bimanual OpenArm, flip the labeled parcel that rests on a table so
that its express-label face (+Z in the parcel frame) ends up pointing up.

Registered tasks:

* ``OpenArm-Flip-v0``       — visual policy input: normalised joint positions +
  front-camera rgb image (CNN policy). This is the default training task.
* ``OpenArm-Flip-State-v0`` — fast state-only baseline (68-dim state, MLP,
  no cameras).
* ``OpenArm-Flip-Play-v0``  — play-mode config of the visual task.

Layout::

    openarm_lab/
      openarm_flip/        task definitions (env cfg + mdp terms) + gym registration
      scripts/             train.py / play.py / eval_model.py / smoke_test.py / camera_check.py
      configs/             rsl_rl PPO hyperparameters (reference yaml)
      assets/              parcel USD, label texture, robot URDF/USD
      logs/                training output (checkpoints + tensorboard events)
"""
from __future__ import annotations

from .openarm_flip import (
    OPENARM_FLIP_ENVS,
    OpenArmFlipEnvCfg,
    OpenArmFlipEnvCfg_PLAY,
    OpenArmFlipStateEnvCfg,
)

__all__ = [
    "OpenArmFlipEnvCfg",
    "OpenArmFlipEnvCfg_PLAY",
    "OpenArmFlipStateEnvCfg",
    "OPENARM_FLIP_ENVS",
]
