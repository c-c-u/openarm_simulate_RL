# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the parcel-flip task."""
from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg


def label_up(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("parcel")) -> torch.Tensor:
    """How "label up" the parcel is: z-component of the label normal [N].

    1.0 = label fully up, 0.0 = sideways, -1.0 = label down.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w.torch  # (N, 4) in (x, y, z, w) order
    x, y, _, _ = quat.unbind(dim=-1)
    return 1.0 - 2.0 * (x * x + y * y)


def label_up_reward(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("parcel"),
    table_pos: tuple[float, float, float] = (0.30, 0.15, 0.35),
    table_size: tuple[float, float, float] = (0.5, 0.7, 0.06),
    table_top: float = 0.38,
) -> torch.Tensor:
    """Dense shaping reward: larger when the label is closer to pointing up.

    Gated on the parcel still being **on the table**: a parcel that has fallen
    off (or been pushed off) earns **no** label-up reward even when its label
    happens to face up — otherwise the policy could learn to "flip and dump".
    """
    up = label_up(env, asset_cfg)
    on_table = parcel_on_table(env, asset_cfg, table_pos, table_size, table_top)
    return up * on_table.float()


def parcel_on_table(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    table_pos: tuple[float, float, float],
    table_size: tuple[float, float, float],
    table_top: float,
    margin: float = 0.03,
) -> torch.Tensor:
    """Boolean [N]: is the parcel center still over the table surface?

    ``margin`` shrinks the horizontal box (keeps a parcel half-hanging off the
    edge from counting as "on table"); the 0.08 m vertical tolerance treats the
    parcel as fallen once its center drops noticeably below the table top.
    """
    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w.torch - env.scene.env_origins  # env-local [N, 3]
    dx = (pos[:, 0] - table_pos[0]).abs()
    dy = (pos[:, 1] - table_pos[1]).abs()
    inside = (dx < table_size[0] / 2.0 - margin) & (dy < table_size[1] / 2.0 - margin)
    above = pos[:, 2] > table_top - 0.08  # 0.08 m: below this it counts as dropped
    return inside & above


def label_up_success_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("parcel"),
    up_cos: float = 0.966,  # cos(15 deg)
    table_pos: tuple[float, float, float] = (0.30, 0.15, 0.35),
    table_size: tuple[float, float, float] = (0.5, 0.7, 0.06),
    table_top: float = 0.38,
) -> torch.Tensor:
    """Sparse success bonus: 1.0 when the label is up and the parcel is on the table."""
    up = label_up(env, asset_cfg) > up_cos
    on_table = parcel_on_table(env, asset_cfg, table_pos, table_size, table_top)
    return (up & on_table).float()


def parcel_dropped_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("parcel"),
    table_top: float = 0.38,
) -> torch.Tensor:
    """1.0 when the parcel fell off the table (below the table top)."""
    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w.torch - env.scene.env_origins  # env-local [N, 3]
    # 与 parcel_on_table 相同的 0.08 m 容差：中心低于桌面 0.08 m 即视为"已掉落"
    return (pos[:, 2] < table_top - 0.08).float()


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Per-step cost that accumulates with episode time (1.0 per policy step).

    The reward manager scales every term by ``weight * dt`` (dt = policy dt),
    so wiring this as ``RewTerm(func=mdp.time_penalty, weight=-0.2)`` charges
    0.2 per simulated second: the total penalty grows linearly with how long
    the episode runs. This pushes the policy to solve the task (and end the
    episode) quickly instead of idling — the counter to the "do nothing"
    local optimum the policy previously fell into.
    """
    return torch.ones(env.num_envs, device=env.device)


def _contact_now(sensor, threshold: float) -> torch.Tensor:
    """(N,) whether ANY sensed body is in contact above ``threshold`` [N]."""
    f = sensor.data.net_forces_w
    f = f.torch if hasattr(f, "torch") else torch.as_tensor(f)
    f = f.reshape(f.shape[0], -1, 3)
    return (f.norm(dim=-1) > threshold).any(dim=-1)


def _any_arm_body_contact(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """(N,) whether any upper-arm link (link1-6, wrist excluded) presses the parcel.

    Reads the two upper-arm body sensors (``left_arm_contact`` /
    ``right_arm_contact``), each matching ``openarm_{side}_link[1-6]`` filtered
    to the parcel. Wrist (link7) and fingers are NOT included — touching the
    parcel with the wrist is allowed, and fingers have their own sensors.
    """
    sensors = env.scene.sensors
    hit = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for n in ("left_arm_contact", "right_arm_contact"):
        if n in sensors:
            hit |= _contact_now(sensors[n], threshold)
    return hit


def arm_body_contact_penalty(
    env: ManagerBasedRLEnv,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """1.0 per step while an upper-arm link (link1-6) presses the parcel.

    Discourages the policy from sweeping / pushing the parcel with the upper
    arm or forearm. The wrist (link7) and the fingertips are allowed — the
    task is meant to be solved with the gripper (and wrist) only. Wire with a
    negative weight (e.g. -2.0 -> -2.0/s while in contact).
    """
    return _any_arm_body_contact(env, force_threshold).float()


class parcel_contact_stage_bonus(ManagerTermBase):
    """One-time bonus when the robot reaches a contact stage with the parcel.

    Contacts are reported by per-finger sensors (``{left,right}_finger{1,2}_
    contact``) and whole-arm-body sensors (``{left,right}_arm_contact``), each
    filtered to the parcel:

    * ``stage="touch"``: ANY **fingertip** presses the parcel (force >
      threshold). Arm-body contact is deliberately excluded — touching the
      parcel with the forearm/wrist is penalised separately by
      :func:`arm_body_contact_penalty`.
    * ``stage="grasp"``: BOTH fingers of one gripper press the parcel at the
      same time — the parcel is pinched between the two fingertips.

    Returns 1.0 on the first step the stage holds and then stays at 0 for the
    rest of the episode (the internal flag is cleared by the reward manager on
    episode reset). Wire the term with ``weight = bonus / dt`` so the configured
    bonus is delivered exactly once (dt = 1/30 s here).
    """

    _FINGER_SENSORS = {
        "left": ["left_finger1_contact", "left_finger2_contact"],
        "right": ["right_finger1_contact", "right_finger2_contact"],
    }

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._given = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        stage: str,
        force_threshold: float = 1.0,
    ) -> torch.Tensor:
        sensors = env.scene.sensors
        if stage == "touch":
            # fingertip contact only (arm-body contact is penalised, not rewarded)
            now = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for names in self._FINGER_SENSORS.values():
                for n in names:
                    if n in sensors:
                        now |= _contact_now(sensors[n], force_threshold)
        elif stage == "grasp":
            left = _contact_now(sensors["left_finger1_contact"], force_threshold) & _contact_now(
                sensors["left_finger2_contact"], force_threshold
            )
            right = _contact_now(sensors["right_finger1_contact"], force_threshold) & _contact_now(
                sensors["right_finger2_contact"], force_threshold
            )
            now = left | right
        else:
            raise ValueError(f"Unknown contact stage: {stage} (expected 'touch' or 'grasp')")

        event = now & ~self._given
        self._given |= now
        return event.float()

    def reset(self, env_ids) -> None:
        self._given[env_ids] = False
