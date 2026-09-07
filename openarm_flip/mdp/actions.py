# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom action term: coupled (synchronised) joint-position gripper action.

Isaac Lab's stock :class:`~isaaclab.envs.mdp.actions.JointPositionAction`
maps **one action dimension per joint**. For the parallel gripper of the
OpenArm bimanual each side has two finger joints that move together (the two
fingers open/close symmetrically). This term couples the two finger joints of
each side into a *single* action dimension, so the action space becomes:

* 14 arm joints (one dimension each), plus
* 1 ``left`` gripper dimension and 1 ``right`` gripper dimension.

A raw value ``a`` in [-1, 1] maps to both finger joints of that side as
``target = default_offset + a * scale`` (identical for both fingers), i.e. the
finger pair is always driven to the same position target.

The URDF-based build and the official USD build differ in finger convention:
* URDF: 0 = open, 0.044 = closed
* official USD: 0 = closed, 0.044 = open
The offsets below (defaults from ``ArticulationCfg.init_state``) handle that
automatically because we read the articulation's ``default_joint_pos``.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor


@configclass.configclass
class CoupledJointPositionActionCfg(ActionTermCfg):
    """Configuration for the coupled (synchronised) joint position action term."""

    class_type: type[CoupledJointPositionAction] | str = f"{__name__}:CoupledJointPositionAction"

    joint_names: list[str] = configclass.MISSING
    """Joint names (regex allowed) that get ONE action dimension each (e.g. the arm)."""

    coupled_joints: dict[str, list[str]] = configclass.MISSING
    """Mapping ``action_name -> list of joint names/regex`` whose members share a
    single action dimension and are driven to the same position target."""

    scale: float | dict[str, float] = 1.0
    """Scale applied to raw action values (per joint name or float). Defaults to 1.0."""

    use_default_offset: bool = True
    """Whether to offset by the articulation's default joint positions. Defaults to True."""

    preserve_order: bool = False
    """Whether to preserve the order of the joint names. Defaults to False."""


class CoupledJointPositionAction(ActionTerm):
    """Coupled joint-position action: independent arm joints + synced gripper groups.

    The action buffer layout is ``[joints..., coupled_group_0, coupled_group_1, ...]``:
    every joint listed in :attr:`CoupledJointPositionActionCfg.joint_names` gets one
    dimension, then each entry of :attr:`coupled_joints` adds one shared dimension
    that drives all of its members to the same target.
    """

    cfg: CoupledJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: CoupledJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        # call the base constructor
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]

        # --- resolve the independent joints ---
        joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order, as_proxy=True
        )
        self._joint_ids = joint_ids.torch

        # --- resolve the coupled groups ---
        self._group_names: list[str] = []
        self._group_joint_ids: list[torch.Tensor] = []
        self._group_joint_names: list[list[str]] = []
        for group_name, group_exprs in self.cfg.coupled_joints.items():
            g_ids, g_names = self._asset.find_joints(
                group_exprs, preserve_order=self.cfg.preserve_order, as_proxy=True
            )
            self._group_names.append(group_name)
            self._group_joint_ids.append(g_ids.torch)
            self._group_joint_names.append(list(g_names))
        if len(self._group_joint_ids) == 0:
            raise ValueError(
                "CoupledJointPositionAction requires at least one entry in 'coupled_joints'."
            )

        # total number of driven joints
        self._all_joint_ids = torch.cat([self._joint_ids, *self._group_joint_ids])
        # action layout: independent joints first, then one dim per coupled group
        self._action_dim = len(self._joint_ids) + len(self._group_joint_ids)

        # buffers
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # --- scale (per driven joint, later expanded to the action layout) ---
        # gather scale for every driven joint (independent + all group members)
        all_joint_names = list(self._joint_names) + [
            n for grp in self._group_joint_names for n in grp
        ]
        if isinstance(self.cfg.scale, (float, int)):
            joint_scale = torch.full((len(all_joint_names),), float(self.cfg.scale), device=self.device)
        elif isinstance(self.cfg.scale, dict):
            joint_scale = torch.ones(len(all_joint_names), device=self.device)
            idx_list, _, val_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, all_joint_names
            )
            joint_scale[idx_list] = torch.tensor(val_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(self.cfg.scale)}.")

        # action-dim scale: independent dims take their own joint scale;
        # each coupled group takes the scale of its FIRST member (all members of a
        # group are driven to the same target, so their scales must be consistent).
        group_first_scale = []
        cursor = len(self._joint_ids)
        for g in range(len(self._group_joint_ids)):
            group_first_scale.append(joint_scale[cursor])
            cursor += len(self._group_joint_ids[g])
        self._scale = torch.cat(
            [joint_scale[: len(self._joint_ids)], torch.tensor(group_first_scale, device=self.device)]
        )

        # --- offset (default joint pos) ---
        if self.cfg.use_default_offset:
            defaults = self._asset.data.default_joint_pos.torch
            # independent dims: their own default
            offset_parts = [defaults[:, self._joint_ids]]
            # coupled dims: use the first member's default (group members share a target)
            for g in range(len(self._group_joint_ids)):
                first_id = self._group_joint_ids[g][0]
                offset_parts.append(defaults[:, first_id].unsqueeze(1))
            self._offset = torch.cat(offset_parts, dim=1)
        else:
            self._offset = 0.0

        # per-driven-joint offset/scale lookup is not needed after the above.

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Action-space size = independent joints + coupled groups (16 here)."""
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw (unprocessed) action buffer in [-1, 1] before scaling/offset."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Action buffer after the affine transform: processed = raw * scale + offset."""
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term."""
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "CoupledJointPositionAction"
        self._IO_descriptor.joint_names = list(self._joint_names) + [
            n for grp in self._group_joint_names for n in grp
        ]
        if isinstance(self._offset, torch.Tensor):
            self._IO_descriptor.offset = self._offset[0].detach().cpu().numpy().tolist()
        else:
            self._IO_descriptor.offset = self._offset
        self._IO_descriptor.scale = self._scale.tolist() if isinstance(self._scale, torch.Tensor) else self._scale
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor) -> None:
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformation: processed = raw * scale + offset
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def apply_actions(self) -> None:
        # build the per-joint targets: expand the coupled dims to all their members
        targets = []
        # independent joints
        targets.append(self._processed_actions[:, : len(self._joint_ids)])
        # coupled groups: repeat the group value for every member joint
        for g in range(len(self._group_joint_ids)):
            g_value = self._processed_actions[:, len(self._joint_ids) + g]
            targets.append(g_value.unsqueeze(1).expand(-1, len(self._group_joint_ids[g])))
        full_targets = torch.cat(targets, dim=1)
        # apply as position targets
        self._asset.set_joint_position_target_index(target=full_targets, joint_ids=self._all_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
