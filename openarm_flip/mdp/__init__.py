# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the OpenArm parcel-flip task."""
from __future__ import annotations

from isaaclab.envs.mdp.events import (
    randomize_actuator_gains,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_offset,
)

from .actions import CoupledJointPositionActionCfg, CoupledJointPositionAction
from .events import reset_parcel_root_state
from .observations import parcel_label_normal_w
from .rewards import (
    arm_body_contact_penalty,
    label_up,
    label_up_reward,
    label_up_success_reward,
    parcel_contact_stage_bonus,
    parcel_dropped_penalty,
    time_penalty,
)
from .terminations import nan_state_guard, parcel_out_of_bound

__all__ = [
    # actions
    "CoupledJointPositionActionCfg",
    "CoupledJointPositionAction",
    # observations
    "parcel_label_normal_w",
    # rewards
    "label_up",
    "label_up_reward",
    "label_up_success_reward",
    "parcel_dropped_penalty",
    "time_penalty",
    "parcel_contact_stage_bonus",
    "arm_body_contact_penalty",
    # events
    "reset_parcel_root_state",
    "reset_joints_by_offset",
    "randomize_rigid_body_material",
    "randomize_rigid_body_mass",
    "randomize_actuator_gains",
    # terminations
    "parcel_out_of_bound",
    "nan_state_guard",
]
