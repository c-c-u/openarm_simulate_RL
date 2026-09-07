# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based RL environment: flip the labeled parcel so its label faces up.

Task layout (per env, Z-up, robot base at origin):

* ``OpenArm``  — OpenArm v1.0 BIMANUAL robot (2 x 7 revolute arm joints + 2 x 2
  prismatic finger joints = 18 DOFs), fixed base at the env origin.
* ``Table``    — kinematic pedestal in front of the robot (top at z = 0.38 m).
* ``Parcel``   — rigid box (0.09 x 0.09 x 0.07 m) with an express label on its
  local +Z face. It starts resting on the table with the label on a random
  face (one of the four sides, or facing down) and a random yaw. The robot has
  to flip it so the label ends up pointing up (+Z).

The "label up" quantity is the world-frame z-component of the parcel's local
+Z axis (1.0 = fully up, -1.0 = down); it is computed from the parcel quaternion
in :mod:`openarm_lab.openarm_flip.mdp`.
"""
from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import actions as mdp_actions
from isaaclab_physx.physics import PhysxCfg
from isaaclab.envs.mdp import observations as mdp_observations
from isaaclab.envs.mdp import rewards as mdp_rewards
from isaaclab.envs.mdp import terminations as mdp_terminations
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg

from . import mdp

# Framework root (this directory): the framework is self-contained and does not
# depend on openarm_sorting (it ships its own copy of the robot URDF and the
# express-label texture).
LAB_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------------------------------------------------------
# Asset paths
# -----------------------------------------------------------------------------

# OpenArm v1.0 BIMANUAL robot (URDF). Collision meshes are the URDF <collision>
# STLs. 18 DOFs: 7 revolute per arm + 2 PRISMATIC fingers per arm (0..0.044 m,
# 0 = open, 0.044 = closed — the parallel jaw can actually enclose the parcel).
#
# Since 2026-09: the robot is spawned from the OFFICIAL Enactic USD
# (openarm_isaac_lab), which ships complete per-link collision meshes (finger
# colliders are the full visual geometry). The old URDF path is kept below for
# reference / rollback.
OPENARM_URDF = LAB_ROOT / "assets/urdf/openarm_bimanual_v1.urdf"
OPENARM_USD_DIR = LAB_ROOT / "assets/usd/openarm_bimanual_v1_lab"

# Official USD (Enactic openarm_isaac_lab). 22 joints: 14 arm revolute +
# 4 prismatic fingers + left/right_hand + left/right_ee_tcp (last four have
# ~zero travel and behave as fixed). Finger joints: *_finger_joint1 = [0, 0.044]
# m, *_finger_joint2 = [-0.0088, 0.0528] m (symmetric pinch around 0.022).
OPENARM_USD = (
    "/home/blanc/projects/openarm_ws/src/openarm_isaac_lab/source/openarm/openarm/"
    "tasks/manager_based/openarm_manipulation/usds/openarm_bimanual/openarm_bimanual.usd"
)

# Labeled parcel USD (hand-authored; label sits on the parcel's local +Z face).
PARCEL_USD = LAB_ROOT / "assets/parcel/parcel.usda"

# ROS package mapping to resolve package://openarm_description/... mesh paths.
OPENARM_PACKAGE = {
    "name": "openarm_description",
    "path": "/home/blanc/projects/openarm_ws/src/openarm_description",
}

# Per-joint-type action scale. Arms are revolute (rad); fingers are prismatic
# with ~0.044 m travel. Home offset = 0.044 (open); a scale of 0.05 lets one
# full action unit (±1 after clipping) sweep the whole range (0.044 ± 0.05 ->
# clamped to [0 (closed), 0.044 (open)] on the official USD build).
ARM_ACTION_SCALE = 0.5  # rad
FINGER_ACTION_SCALE = 0.05  # m (official USD travel ~[0, 0.044] / [-0.0088, 0.0528])

# Table / parcel layout (matches openarm_sorting SceneConfig geometry).
TABLE_POS = (0.30, 0.15, 0.35)
TABLE_SIZE = (0.5, 0.7, 0.06)
TABLE_TOP = TABLE_POS[2] + TABLE_SIZE[2] / 2.0  # 0.38 m
PARCEL_SIZE = (0.09, 0.09, 0.07)
# Env-local (x, y) the parcel is centred on. y = 0.10 keeps it close to the
# robot centreline so both arms can reach it (the arms sit at y ~ +-0.03).
PARCEL_XY = (0.30, 0.10)
PARCEL_POS = (PARCEL_XY[0], PARCEL_XY[1], TABLE_TOP + PARCEL_SIZE[2] / 2.0)  # resting upright

# A label counts as "up" within this angle of world +Z.
LABEL_UP_COS = float(math.cos(math.radians(15)))

# One-time contact bonuses (delivered once per episode by the corresponding
# reward terms, whose weights are rescaled by dt in __post_init__).
# "touch" = ANY robot body (fingertips or any arm link) presses the parcel;
# "grasp" = both fingers of one gripper pinch the parcel.
CONTACT_TOUCH_BONUS = 0.5
CONTACT_GRASP_BONUS = 0.5

# -----------------------------------------------------------------------------
# Cameras (same mounting as the openarm_sorting demo scene)
# -----------------------------------------------------------------------------
# Default camera resolution for the TRAINING scene. The original openarm_sorting
# demo scene used 640x480; the smaller default keeps 3 RTX views per env within
# GPU memory at moderate env counts (raise it for close-up capture).
CAM_WIDTH, CAM_HEIGHT = 320, 240
# Resolution of the front camera when it feeds the policy image observation
# (smaller = cheaper rendering + CNN). 4:3 like the display cameras.
POLICY_CAM_WIDTH, POLICY_CAM_HEIGHT = 96, 72
CAM_DATA_TYPES = ["rgb", "distance_to_image_plane"]
# Camera in front of the midpoint of the two arm shoulders (shoulders sit at
# (0, +-0.031, 0.698) in the v1 URDF, so the midpoint is (0, 0, ~0.70)), at
# shoulder height, centred between the arms, looking diagonally down at the
# parcel on the table.
FRONT_CAM_POS = (0.15, 0.0, 0.70)
FRONT_CAM_TARGET = (0.30, PARCEL_XY[1], 0.40)  # looks at the parcel on the table
WRIST_CAM_LOCAL = (0.0, 0.0, -0.06)        # 6 cm below the EE (tool axis -Z)


def _camera_look_at_quat_xyzw(eye: tuple[float, float, float], target: tuple[float, float, float]):
    """Camera orientation (x, y, z, w) so the camera -Z points from ``eye`` to ``target``.

    USD cameras look along -Z; this mirrors the original scene's look-at helper.
    """
    import numpy as np

    fwd = np.array(eye, dtype=float) - np.array(target, dtype=float)
    fwd = fwd / np.linalg.norm(fwd)  # camera -Z (view direction)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(up, fwd)
    right = right / np.linalg.norm(right)
    up2 = np.cross(fwd, right)
    R = np.column_stack([right, up2, fwd])  # columns: camera x, y, z (z = -view)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2.0
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[3] = (R[k, j] - R[j, k]) / s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    return tuple(float(v) for v in q)


FRONT_CAM_ROT_XYZW = _camera_look_at_quat_xyzw(FRONT_CAM_POS, FRONT_CAM_TARGET)


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------

# Which robot asset to use:
#  * "official" -> Enactic openarm_isaac_lab USD (complete per-link collision
#    meshes; finger colliders = full visual geometry). 22 joints, of which the
#    four *_hand / *_ee_tcp joints have ~zero travel (fixed in practice).
#  * "urdf"     -> local URDF -> cached-USD conversion (18 joints).
ROBOT_SOURCE = "official"


def _official_usd_spawn_cfg() -> sim_utils.UsdFileCfg:
    """Spawner for the official Enactic openarm bimanual USD.

    Mirrors the official ``assets/openarm_bimanual.py`` articulation cfg
    (same rigid/articulation properties + per-link complete collision meshes).
    """
    return sim_utils.UsdFileCfg(
        usd_path=str(OPENARM_USD),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        activate_contact_sensors=True,  # contact reporting on all robot bodies
    )


def _urdf_spawn_cfg() -> sim_utils.UrdfFileCfg:
    """URDF spawner shared by the articulation cfg (drive gains baked in).

    Kept for rollback when ``ROBOT_SOURCE == "urdf"``.
    """
    return sim_utils.UrdfFileCfg(
        asset_path=str(OPENARM_URDF),
        usd_dir=str(OPENARM_USD_DIR),
        usd_file_name="openarm_bimanual_v1.usd",
        fix_base=True,
        merge_fixed_joints=True,
        collision_type="Convex Hull",
        collision_from_visuals=False,
        activate_contact_sensors=True,  # contact reporting on all robot bodies
        ros_package_paths=[dict(OPENARM_PACKAGE)],
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=2000.0, damping=100.0),
        ),
    )


def _robot_spawn_cfg():
    """Pick the robot spawner per ``ROBOT_SOURCE``."""
    if ROBOT_SOURCE == "official":
        return _official_usd_spawn_cfg()
    return _urdf_spawn_cfg()


# Robot home pose (18 DOFs): L1..L7, R1..R7, then fingers.
# Official USD fingers: *_finger_joint1 = [0, 0.044], *_finger_joint2 =
# [-0.0088, 0.0528]; gap measurement shows 0 = CLOSED, 0.044 = OPEN (opposite
# of the old URDF build). The official config initialises fingers at 0.044.
OPENARM_HOME_POS = {
    "openarm_left_joint1": 1.2,
    "openarm_left_joint2": 0.0,
    "openarm_left_joint3": -1.2,
    "openarm_left_joint4": 1.2,
    "openarm_left_joint5": 0.0,
    "openarm_left_joint6": 0.0,
    "openarm_left_joint7": 0.0,
    "openarm_right_joint1": -1.248,
    "openarm_right_joint2": 0.021,
    "openarm_right_joint3": 1.123,
    "openarm_right_joint4": 1.2,
    "openarm_right_joint5": 0.077,
    "openarm_right_joint6": -0.002,
    "openarm_right_joint7": -0.053,
    "openarm_left_finger_joint1": 0.044,  # open
    "openarm_left_finger_joint2": 0.044,  # open
    "openarm_right_finger_joint1": 0.044,  # open
    "openarm_right_finger_joint2": 0.044,  # open
}

# The four fixed-ish joints present in the official USD (excluded from the 18-DOF
# action/observation spaces; they are left un-actuated ~ fixed at 0).
OFFICIAL_FIXED_JOINTS = [
    "openarm_left_hand",
    "openarm_right_hand",
    "openarm_left_ee_tcp_joint",
    "openarm_right_ee_tcp_joint",
]

# 18 controlled joints shared by the URDF and the official USD builds.
CONTROLLED_JOINTS = [
    "openarm_left_joint[1-7]",
    "openarm_right_joint[1-7]",
    "openarm_left_finger_joint[1-2]",
    "openarm_right_finger_joint[1-2]",
]


@configclass
class OpenArmFlipSceneCfg(InteractiveSceneCfg):
    """Scene: OpenArm + kinematic table + labeled parcel."""

    # 默认并行环境数 / 相邻环境原点间距（米）。相机+CNN 开销大，视觉任务训练时按显存下调。
    num_envs = 1024
    env_spacing = 2.0

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/OpenArm",
        spawn=_robot_spawn_cfg(),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), joint_pos=dict(OPENARM_HOME_POS)),
        actuators={
            # 左臂 7 个转动关节（隐式 PD 位置控制：刚度 2000 N·m/rad、阻尼 100 N·m·s/rad）
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                stiffness=2000.0,
                damping=100.0,
            ),
            # 右臂 7 个转动关节（参数同左臂）
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_joint[1-7]"],
                stiffness=2000.0,
                damping=100.0,
            ),
            # 左手两个平移手指关节：刚度 5000（高于臂关节）保证夹紧盒子时位置误差小
            "left_fingers": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[1-2]"],
                stiffness=5000.0,
                damping=200.0,
            ),
            # 右手两个平移手指关节（参数同左手）
            "right_fingers": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_finger_joint[1-2]"],
                stiffness=5000.0,
                damping=200.0,
            ),
        },
    )

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.58)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_POS),
    )

    parcel: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Parcel",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PARCEL_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # 基准质量 0.35 kg（EventCfg 里 startup 还会按 0.2~0.6 kg 随机化质量）
            mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            activate_contact_sensors=True,  # 开启盒子接触上报（阶段奖励 / 碰盒惩罚依赖它）
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=PARCEL_POS),
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(0.9, 0.9, 0.9)),
        collision_group=-1,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        # bright neutral dome light: RTX path tracing eats intensity fast, and a
        # dim scene reads as "grey" (robot is dark-grey, parcel light-brown).
        spawn=sim_utils.DomeLightCfg(intensity=1200.0, color=(1.0, 1.0, 1.0)),
    )

    # 相机占位字段（与 openarm_sorting demo 场景同款安装）：enable_cameras=True 时
    # 由 OpenArmFlipEnvCfg.__post_init__ -> _setup_cameras() 填充。
    front_cam: CameraCfg | None = None
    wrist_left_cam: CameraCfg | None = None
    wrist_right_cam: CameraCfg | None = None

    # 接触传感器占位字段（enable_contact_rewards=True 时由 __post_init__ 填充）：
    # * 4 个"单指"传感器（每指一个 body、过滤到盒子）——驱动 touch/grasp 一次性奖励；
    # * 2 个"上臂"传感器（每臂 link[1-6]、过滤到盒子）——驱动上臂碰盒惩罚。
    left_finger1_contact: ContactSensorCfg | None = None
    left_finger2_contact: ContactSensorCfg | None = None
    right_finger1_contact: ContactSensorCfg | None = None
    right_finger2_contact: ContactSensorCfg | None = None
    # 上臂接触传感器：每侧匹配 link[1-6]（上臂+前臂，**排除腕部 link7**——腕部紧邻
    # 夹爪，允许碰盒），过滤到盒子，用于惩罚用大臂/前臂碰盒子。
    left_arm_contact: ContactSensorCfg | None = None
    right_arm_contact: ContactSensorCfg | None = None


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------

@configclass
class StateObservationsCfg:
    """Full-state observations (policy group only) — used by the fast
    state-only task ``OpenArm-Flip-State-v0``."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # 18 个受控关节位置（限位归一化；±0.02 均匀噪声，使观测对噪声鲁棒）
        joint_pos = ObsTerm(
            func=mdp_observations.joint_pos_limit_normalized,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)},
        )
        # 18 个关节速度（×0.1 缩放；±0.1 噪声）
        joint_vel = ObsTerm(
            func=mdp_observations.joint_vel,
            scale=0.1,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)},
        )
        # 盒子本体状态（env-local 世界系）：位置 / 姿态 / 线速度 / 角速度，让策略感知盒子当前位姿
        parcel_pos = ObsTerm(func=mdp_observations.root_pos_w, params={"asset_cfg": SceneEntityCfg("parcel")})
        parcel_quat = ObsTerm(
            func=mdp_observations.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("parcel"), "make_quat_unique": False},
        )
        parcel_lin_vel = ObsTerm(
            func=mdp_observations.root_lin_vel_w, scale=0.2, params={"asset_cfg": SceneEntityCfg("parcel")}
        )
        parcel_ang_vel = ObsTerm(
            func=mdp_observations.root_ang_vel_w, scale=0.2, params={"asset_cfg": SceneEntityCfg("parcel")}
        )
        # 标签面（盒子局部 +Z）法向的世界系分量 —— 任务的目标量（朝上 = (0, 0, 1)）
        label_normal = ObsTerm(func=mdp.parcel_label_normal_w, params={"asset_cfg": SceneEntityCfg("parcel")})
        # 上一步动作（16 维）：给策略"动作平滑 / 惯性"先验
        last_action = ObsTerm(func=mdp_observations.last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class ImageObservationsCfg:
    """Joint positions + front-camera image (the visual policy input)."""

    @configclass
    class JointObsCfg(ObsGroup):
        """1D group: normalised joint positions only."""

        # 18 个受控关节位置（限位归一化；±0.02 均匀噪声，使观测对噪声鲁棒）
        joint_pos = ObsTerm(
            func=mdp_observations.joint_pos_limit_normalized,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)},
        )

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class ImageObsCfg(ObsGroup):
        """2D group: rgb image from the front camera (channels-first for the CNN)."""

        image = ObsTerm(
            func=mdp_observations.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_cam"),
                "data_type": "rgb",
                "normalize": True,  # 归一化到 [0, 1]
                "permute": True,  # 转置为 (N, C, H, W) 通道在前（CNN 输入格式）
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: JointObsCfg = JointObsCfg()
    image: ImageObsCfg = ImageObsCfg()


# -----------------------------------------------------------------------------
# Actions
# -----------------------------------------------------------------------------

@configclass
class ActionsCfg:
    """Joint position targets: 14 independent arm joints + 2 coupled grippers.

    Each side's two finger joints are **coupled into one gripper dimension**
    (they always receive the same position target, i.e. the parallel jaw opens /
    closes symmetrically). Action space = 14 + 2 = 16 dims.
    """

    # 16 维动作：joint_names（14 个臂关节）各占 1 维 + coupled_joints（left/right_gripper）各占 1 维
    joint_pos: mdp.CoupledJointPositionActionCfg = mdp.CoupledJointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "openarm_left_joint[1-7]",
            "openarm_right_joint[1-7]",
        ],
        coupled_joints={
            "left_gripper": ["openarm_left_finger_joint[1-2]"],
            "right_gripper": ["openarm_right_finger_joint[1-2]"],
        },
        scale={
            "openarm_left_joint[1-7]": ARM_ACTION_SCALE,
            "openarm_right_joint[1-7]": ARM_ACTION_SCALE,
            "openarm_left_finger_joint[1-2]": FINGER_ACTION_SCALE,
            "openarm_right_finger_joint[1-2]": FINGER_ACTION_SCALE,
        },
        use_default_offset=True,  # 以 default_joint_pos 为偏移：自动兼容官方 USD（张开=0.044）与 URDF（张开=0）两套手指约定
    )


# -----------------------------------------------------------------------------
# Rewards
# -----------------------------------------------------------------------------

@configclass
class RewardsCfg:
    """Reward terms."""

    # 密集塑形：标签法向与 +Z 的夹角越小奖励越大（线性）。
    # 门控：只有盒子仍在桌上才给付——盒子掉下去时即使标签碰巧朝上也不给分（防止"翻完就扔"）。
    label_up = RewTerm(
        func=mdp.label_up_reward,
        weight=5.0,  # 密集塑形权重：部分进展也给清晰的正面信号
        params={
            "asset_cfg": SceneEntityCfg("parcel"),
            "table_pos": TABLE_POS,
            "table_size": TABLE_SIZE,
            "table_top": TABLE_TOP,  # 门控参数：由桌子位置/尺寸判断盒子是否仍在桌面上
        },
    )
    # 稀疏成功奖励：标签朝上（与 +Z 夹角 < 15°）**且**盒子仍在桌上，一次性 +200
    success_bonus = RewTerm(
        func=mdp.label_up_success_reward,
        weight=200.0,
        params={
            "asset_cfg": SceneEntityCfg("parcel"),
            "up_cos": LABEL_UP_COS,  # cos(15°)≈0.966：判定"标签朝上"的角度阈值
            "table_pos": TABLE_POS,
            "table_size": TABLE_SIZE,
            "table_top": TABLE_TOP,  # 成功同样必须满足"盒子在桌上"
        },
    )
    # 动作幅度惩罚 −0.005·‖a‖²：保持动作平滑、抑制抖动
    action_l2 = RewTerm(func=mdp_rewards.action_l2, weight=-0.005)
    # 盒子掉下桌的强惩罚
    parcel_dropped = RewTerm(
        func=mdp.parcel_dropped_penalty,
        weight=-5.0,  # "尝试后失败"比"原地不动"划算（早期版本是 -50）
        params={"asset_cfg": SceneEntityCfg("parcel"), "table_top": TABLE_TOP},
    )
    # 时间惩罚：`weight` 即每秒费率（-0.2/s），总扣分随回合时长线性增长——
    # 拖延 / 原地不动的解罚得更多，用来对抗策略的"什么都不做"局部最优。
    time_penalty = RewTerm(
        func=mdp.time_penalty,
        weight=-0.2,
    )

    # 一次性阶段奖励：首次指尖接触盒子 +0.5，随后首次同侧两指"夹住"盒子再 +0.5。
    # 权重在 __post_init__ 里按 weight = bonus / dt 重算（dt = 1/30 s），
    # 因此每次事件实际恰好给付 0.5。
    parcel_touch_bonus = RewTerm(
        func=mdp.parcel_contact_stage_bonus,
        weight=15.0,  # = 0.5 / dt（dt = 1/30 s）→ 事件发生当步给付 +0.5
        params={"stage": "touch", "force_threshold": 1.0},  # force_threshold=1N：合力超过 1 N 才算"接触"
    )
    parcel_grasp_bonus = RewTerm(
        func=mdp.parcel_contact_stage_bonus,
        weight=15.0,  # = 0.5 / dt → 给付 +0.5
        params={"stage": "grasp", "force_threshold": 1.0},  # grasp = 同侧两指同时压住盒子
    )
    # 上臂（link1-6：上臂 / 前臂）碰盒惩罚：按秒计费 -2.0/s。腕部 link7 与指尖允许碰盒
    # （任务要求用夹爪/手腕完成，不允许用大臂把盒子扫来扫去）。
    arm_body_contact = RewTerm(
        func=mdp.arm_body_contact_penalty,
        weight=-2.0,
        params={"force_threshold": 1.0},  # 1 N 以上才算接触
    )


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------

@configclass
class TerminationsCfg:
    """Episode end conditions."""

    # 回合超时（episode_length_s）——RL 环境的标准 time_out 终止
    time_out = DoneTerm(func=mdp_terminations.time_out, time_out=True)

    # 盒子中心离开允许范围（env-local x/y/z，比桌面略宽）即终止：掉下桌、被推飞都会触发
    parcel_out_of_bound = DoneTerm(
        func=mdp.parcel_out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("parcel"),
            "in_bound_range": {"x": (-0.2, 0.9), "y": (-0.6, 1.0), "z": (0.0, 1.0)},
        },
    )

    # 探索期物理发散保护：关节超速即重置该环境（终止环境会在返回观测前被重置，
    # 因此策略不会看到坏状态）。100 rad/s 远高于正常转速（最坏 ~40 rad/s），
    # 能在数值爆炸变成 NaN 之前把环境拦下来。
    bad_robot_state = DoneTerm(
        func=mdp_terminations.joint_vel_out_of_manual_limit,
        params={"max_velocity": 100.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    # NaN 兜底：关节/盒子状态出现 NaN 就终止该环境（原因同上，重置后观测干净）
    nan_state = DoneTerm(func=mdp.nan_state_guard)


# -----------------------------------------------------------------------------
# Events (reset randomization)
# -----------------------------------------------------------------------------

@configclass
class EventCfg:
    """Reset / randomization events."""

    # --- 机器人：每次 reset 回到 home 位姿（关节默认位置）± 小噪声，速度清零 ---
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),  # 关节位置抖动 ±0.05 rad
            "velocity_range": (0.0, 0.0),  # 速度清零
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS),
        },
    )

    # --- 盒子：随机标签面（侧面/朝下）、随机偏航、桌上抖动、零速度 ---
    # side_prob = 标签落在 SIDE 面（需 90° 翻）的概率；以 (1 - side_prob) 的概率标签
    # 朝下（盒子倒扣，需 180° 翻）。混合两种起手教策略学会"通用翻面"，
    # 而不是只会简单的 90° 情形。
    reset_parcel = EventTerm(
        func=mdp.reset_parcel_root_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("parcel"),
            "table_pos": TABLE_POS,
            "table_size": TABLE_SIZE,
            "table_top": TABLE_TOP,
            "parcel_size": PARCEL_SIZE,
            "parcel_xy": PARCEL_XY,
            "pos_jitter": 0.03,  # 桌面上 (x, y) 放置位置抖动 ±3 cm，避免每次重置完全一样
            "side_prob": 0.7,  # 70% 标签在侧面（90° 翻），30% 标签朝下（180° 翻）
            "velocity_range": (0.0, 0.0),  # 重置后盒子速度为零（悬浮在静置高度再落下）
        },
    )

    # --- 物理 / 动力学随机化（startup 一次性，每个环境各抽一档） ---
    # 机器人全身表面摩擦（静态/动态同为 0.6~1.2，128 个桶离散采样）
    randomize_robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )
    # 盒子表面摩擦（0.3~0.9，略低以允许推/滑），恢复系数 0~0.1
    randomize_parcel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("parcel", body_names=".*"),
            "static_friction_range": (0.3, 0.9),
            "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 128,
        },
    )
    # 盒子质量：0.2~0.6 kg 绝对随机（覆盖不同"手感"）
    randomize_parcel_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("parcel"),
            "mass_distribution_params": (0.2, 0.6),
            "operation": "abs",
        },
    )
    # 执行器刚度/阻尼 ×0.7~1.3 缩放（模拟个体差异 / 装配误差）
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.7, 1.3),
            "damping_distribution_params": (0.7, 1.3),
            "operation": "scale",
        },
    )


# -----------------------------------------------------------------------------
# Env
# -----------------------------------------------------------------------------

@configclass
class OpenArmFlipEnvCfg(ManagerBasedRLEnvCfg):
    """OpenArm parcel-flip RL environment (manager-based)."""

    # environment
    scene: OpenArmFlipSceneCfg = OpenArmFlipSceneCfg()
    enable_image_obs: bool = True
    """策略输入 = 归一化关节位置 + 前视相机 RGB 图像。

    图像观测强制开启相机（``enable_cameras`` 会被置 True）。纯状态任务
    （``OpenArm-Flip-State-v0``）把它设成 False，保留完整 68 维状态向量，
    训练快得多。
    """
    enable_cameras: bool = True
    """每个环境都生成前视 + 左右腕部三个相机。

    安装方式与 openarm_sorting demo 场景一致：机器人正前方一个固定相机 +
    每个末端执行器一个"眼在手"相机。渲染相机视图有开销——纯状态快速训练请传
    ``--disable_cameras``（见 scripts/train.py）。
    """
    enable_contact_rewards: bool = True
    """生成盒子/手指接触传感器并启用一次性 touch/grasp 阶段奖励
    （见 mdp.rewards.parcel_contact_stage_bonus）。"""
    observations: StateObservationsCfg | ImageObservationsCfg = StateObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # simulation
    # 显式加大 PhysX GPU 缓冲容量：双臂机器人（18 DOF、~30 碰撞体）跑到 1024+ envs 会
    # 溢出默认缓冲，表现为观测出现 NaN。量级参照 Isaac Lab lift 任务。
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        physics=PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_max_rigid_patch_count=8 * 5 * 2**15,
            gpu_found_lost_pairs_capacity=2**26,
            gpu_max_rigid_contact_count=2**24,
        ),
    )
    decimation = 4  # 每 4 个物理步(120 Hz)执行一次策略 -> 策略频率 30 Hz
    episode_length_s = 10.0  # 回合时长上限：10 秒（= 300 个策略步）

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.sim.render_interval = self.decimation
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(1.4, -1.2, 1.2), lookat=(0.2, PARCEL_XY[1], 0.4))
        if self.enable_image_obs:
            self.enable_cameras = True  # 图像观测必须有相机
            self.observations = ImageObservationsCfg()
        else:
            self.observations = StateObservationsCfg()
        if self.enable_cameras:
            self._setup_cameras()
        if self.enable_contact_rewards:
            self._setup_contact_sensors()
            # 一次性奖励实际给付 0.5：奖励管理器按 dt 缩放权重，
            # 故这里把权重重算为 bonus / dt（dt = 1/30 s → weight = 15.0）
            dt = self.decimation * self.sim.dt
            self.rewards.parcel_touch_bonus.weight = CONTACT_TOUCH_BONUS / dt
            self.rewards.parcel_grasp_bonus.weight = CONTACT_GRASP_BONUS / dt
        if not Path(PARCEL_USD).exists():
            raise FileNotFoundError(
                f"Parcel USD not found at {PARCEL_USD}. "
                "Recreate it from the committed assets/parcel/parcel.usda."
            )

    def _setup_cameras(self) -> None:
        """Create the front + wrist cameras (same mounts as the demo scene)."""
        cam_spawn = sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        )
        # 机器人正前方的固定相机，斜向下看桌上的盒子；
        # 若它还喂给策略图像观测（enable_image_obs），就用低分辨率以省渲染+CNN 开销
        cam_w, cam_h = (POLICY_CAM_WIDTH, POLICY_CAM_HEIGHT) if self.enable_image_obs else (CAM_WIDTH, CAM_HEIGHT)
        self.scene.front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            update_period=0.0,
            height=cam_h,
            width=cam_w,
            data_types=CAM_DATA_TYPES,
            spawn=cam_spawn,
            offset=CameraCfg.OffsetCfg(
                pos=FRONT_CAM_POS, rot=FRONT_CAM_ROT_XYZW, convention="opengl"
            ),
        )
        # 眼在手：挂在每侧末端工具 link(link7) 下方 6 cm，沿工具轴(-Z)看向指尖/盒子
        for side, name in (("left", "wrist_left_cam"), ("right", "wrist_right_cam")):
            setattr(
                self.scene,
                name,
                CameraCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/OpenArm/openarm_{side}_link7/WristCamera",
                    update_period=0.0,
                    height=CAM_HEIGHT,
                    width=CAM_WIDTH,
                    data_types=CAM_DATA_TYPES,
                    spawn=cam_spawn,
                    offset=CameraCfg.OffsetCfg(pos=WRIST_CAM_LOCAL, rot=(0.0, 0.0, 0.0, 1.0), convention="ros"),
                ),
            )

    def _setup_contact_sensors(self) -> None:
        """生成 touch/grasp 阶段奖励所需的接触传感器（body 匹配规则如下）。

        * **4 个"单指"传感器**：每根手指一个 body（``openarm_{side}_{f}_finger``），
          过滤到盒子——用于 grasp 阶段（同侧两指同时压住盒子）与 touch 阶段；
        * **2 个"上臂"传感器**：每侧匹配非腕部连杆 ``openarm_{side}_link[1-6]``
          （上臂 + 前臂，**排除腕部 link7**），过滤到盒子——用于惩罚用大臂/前臂
          推扫盒子；腕部 link7 紧邻夹爪，允许碰盒。

        physx 接触视图每个传感器只支持一个过滤 pattern；一个传感器可用正则匹配多个
        body，净接触力返回形状为 (num_envs, num_bodies, 3)。
        """
        finger_filter = ["{ENV_REGEX_NS}/Parcel.*"]  # 所有传感器都只关心与盒子的接触
        for side in ("left", "right"):
            # finger_joint1 的子体叫 *_right_finger、finger_joint2 的子体叫 *_left_finger
            for f in ("right", "left"):
                name = f"{side}_finger{1 if f == 'right' else 2}_contact"
                setattr(
                    self.scene,
                    name,
                    ContactSensorCfg(
                        prim_path=f"{{ENV_REGEX_NS}}/OpenArm/openarm_{side}_{f}_finger",
                        update_period=0.0,
                        filter_prim_paths_expr=finger_filter,
                    ),
                )
            # 上臂传感器：只匹配 link1..6（排除腕部 link7 及固定的 hand / ee_tcp 连杆）
            setattr(
                self.scene,
                f"{side}_arm_contact",
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/OpenArm/openarm_{side}_link[1-6]",
                    update_period=0.0,
                    filter_prim_paths_expr=finger_filter,
                ),
            )


@configclass
class OpenArmFlipStateEnvCfg(OpenArmFlipEnvCfg):
    """纯状态快速变体：完整 68 维状态观测、无相机。

    注册为 ``OpenArm-Flip-State-v0``；适合快速跑基线、以及在训练视觉策略前
    先调试任务动力学。
    """

    enable_image_obs: bool = False  # 关闭图像观测（走 StateObservationsCfg）
    enable_cameras: bool = False  # 无相机，训练开销小很多


@configclass
class OpenArmFlipEnvCfg_PLAY(OpenArmFlipEnvCfg):
    """播放配置（更少 env、更长回合、更温和的随机化，便于演示）。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 15.0
        # 播放时弱化随机化：只出 90° 侧面起手（side_prob=1.0）、放置抖动减小
        self.events.reset_parcel.params["side_prob"] = 1.0
        self.events.reset_parcel.params["pos_jitter"] = 0.02
