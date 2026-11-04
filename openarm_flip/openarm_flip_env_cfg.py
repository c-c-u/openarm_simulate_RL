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
from ..tools import env_backdrop as backdrop

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
# 背景墙：真实场景照片（缩小 sim→real 的外观差 / 外观域随机化）
# -----------------------------------------------------------------------------
# 工作区四周立四面**真实场景照片**的墙（照片来自 `assets/textures/env/`，CC0，
# 也可用 OPENARM_ENV_TEX_DIR 指向自己现场拍的图）。作用有两个：
#   1. 视觉任务的画面上不再是"灰白空世界"，颜色/纹理分布接近真机背景；
#   2. 配合 `EventCfg.randomize_backdrop` 定期换照片，CNN 学不到固定的背景线索。
#
# 为什么是四面 + 为什么在**每个环境内**各立一套：
#   相机与腕部相机会朝向各个方向，只补一面会在别的视角露出空世界；而每个环境
#   （`env_spacing` = 2 m，即半格 1.0 m）各自有一套墙，就不会看到邻居的墙。
#   墙是**纯视觉几何（无碰撞体）**，对物理与状态观测零影响。
#
# 尺寸约束（重要）：`BACKDROP_DISTANCE` 必须 < env_spacing/2，墙宽也别超过
# env_spacing，否则相邻环境的墙会互相穿插。改了墙的宽/高还要重跑
# `python3 tools/make_backdrop_usd.py --width .. --height ..` 重新生成资产。
BACKDROP_WALL_USD = LAB_ROOT / "assets" / "scene" / "backdrop_wall.usda"
BACKDROP_DISTANCE = 0.95   # 墙到环境原点的水平距离（m）；须 < env_spacing/2 = 1.0
BACKDROP_WIDTH = 1.9       # 墙宽（m），与 assets/scene/backdrop_wall.usda 一致
BACKDROP_HEIGHT = 1.6      # 墙高（m），与资产一致
BACKDROP_WALLS = backdrop.default_walls(
    table_center=(0.0, 0.0),          # 以环境原点为中心（不是桌子），保证落在自己那格里
    distance=BACKDROP_DISTANCE,
    width=BACKDROP_WIDTH,
    height=BACKDROP_HEIGHT,
)
# 墙名 -> OpenArmFlipSceneCfg 里的字段名
BACKDROP_FIELD_NAMES = {
    "Backdrop_PlusX": "backdrop_plus_x",
    "Backdrop_MinusX": "backdrop_minus_x",
    "Backdrop_PlusY": "backdrop_plus_y",
    "Backdrop_MinusY": "backdrop_minus_y",
}

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
# 前视相机（非策略图像观测时）的分辨率。用 640x480 而非 320x240 是有意的：
# 12 mm 广角（见 FRONT_CAM_FOCAL）下 640x480 的像素密度恰好等于"24 mm + 320x240"，
# 也就是**盒子在画面上占的像素数不变（86 px vs 原来 94 px），但视野大了 9 倍**。
FRONT_CAM_WIDTH, FRONT_CAM_HEIGHT = 640, 480
CAM_DATA_TYPES = ["rgb", "distance_to_image_plane"]
# 前视相机：机器人正前方（两肩中点 (0,0,~0.70) 前方 15 cm），俯视桌面上的盒子。
#
# 取景历史（重要，别改回去）：
#   原配置 24 mm / pos (0.15,0,0.70) / target (0.30,0.10,0.40) 视场只有 47°，
#   距盒子仅 0.35 m。实测采 100 帧：翻转时策略把盒子推走 15.8 cm（+x/+y），
#   盒子自第 ~40 帧起因顶部越界被截断，**62% 的帧可见面积不足 50%**，而第 40 帧
#   恰好是盒子翻正的时刻 —— 最有信息量的一段全丢了。
#
#   按工作包络（x 0.24-0.52, y 0.02-0.30, z 0.41-0.46 m，4 种朝向，2700 个采样点）
#   评估候选相机，结论：
#     24 mm 现状              包络覆盖 11.0%（最大越界 175 px）
#     12 mm 原位 + 重新瞄准   包络覆盖 99.7%（最大越界 2 px）  <- 采用
#     8.5 mm                  覆盖 100% 但盒子只剩 28 px，过度加宽反而有害
#   注：**要覆盖的范围固定时，盒子像素数只由"覆盖范围"和"分辨率"决定，与焦距无关**
#   （焦距变小就能靠近，两个效应正好抵消）。所以加宽 FOV 不损失分辨率。
# 朝向说明（实测分解，别凭感觉调）：
#   俯仰 pitch ≈ 50°（下视）—— **几何必需**：相机在肩高 0.72 m，盒子在桌面约 0.42 m，
#     不下视就看不到桌面。水平相机（pitch=0）时盒子在光轴下方 38°+，而已知垂直半视场
#     只有 33.2°，盒子直接掉出画面。想减小 pitch 只能把相机**后移**
#     （pitch = atan(0.30 / 水平距离)：退到 x=-0.05 才降到 37°）。
#   偏航 yaw = 0（光轴严格平行机器人 +x）—— 原先 yaw 有 31°，因为相机在机器人中线
#     y=0 而盒子在 y=0.10，靠得很近（0.20 m）时 atan(0.12/0.20)=31°，画面里桌面长边
#     是斜的。yaw=0 时相机右轴恰好 = 世界 +y，桌面长边在画面里严格水平。
#   横滚 roll = 0 —— look-at 构造保证相机右轴水平（cross(world_up, fwd) 必水平）。
#
# 为什么相机要放在 y=0.14 这条纵线上（而不是机器人中线 y=0）：盒子不在中线上。
# 实测取景率（scripts/probe_workspace.py，64 envs × 600 步、34793 个真实样本，
# 判据 = 盒子 8 角点全部落在画面内）：
#     相机位置           光轴      yaw    整回合   达标前且前60步   盒子像素
#     (0.15, 0,    0.72) 瞄准盒子  31.0°   87.4%      96.8%        133 px
#     (0.10, 0.14, 0.72) 沿 +x      0.0°   85.0%      97.5%        130 px  <- 采用
#     (0.15, 0,    0.72) 沿 +x      0.0°   51.3%      85.0%        151 px  <- 字面"正前方"，不可用
# 即"光轴沿正前方"与"盒子留在画面内"两者，只有在相机与盒子同一纵线上时才能兼得；
# 坚持把相机留在机器人中线，就只能斜着看（否则丢掉近一半取景率）。
# 另注：整回合取景率偏低是"盒子被推远"造成的，配合采集脚本的
# --window_steps/--after_success 窗口即可回到 97.5%。
FRONT_CAM_POS = (0.10, 0.14, 0.72)
FRONT_CAM_TARGET = (0.35, 0.14, 0.42)  # 与相机同 y => yaw 严格为 0；瞄准盒子轨迹中心
# 前视相机焦距（mm）。12 mm 配合 20.955 mm 水平像幅 => 水平视场 82.3°、垂直 66.4°。
# 腕部相机保持 24 mm（它们只看 15 cm 外的指尖，不需要广角）。
FRONT_CAM_FOCAL = 12.0
WRIST_CAM_FOCAL = 24.0

# 腕部相机在 link7 局部系中的安装偏移：沿 **局部 +X 15 cm**。
# 几何依据（实测 link7 局部系）：
#   * 夹爪开合方向 = 局部 -Y（两指尖连线方向）
#   * 工具轴 / 腕部转轴 = 局部 +Z（指向指尖，与 link7->hand 方向余弦 0.921）
#   * 垂直于这两者 = 局部 X 轴；其中 +X 在世界系 z 分量为正（左 +0.223 / 右 +0.205），
#     即位于夹爪的"上方"一侧 —— 正是所需安装位（既不挡开合方向，也不挡工具轴前方视野）。
#   * 注：早期版本用的是 (0,0,-0.06)（沿工具轴反向 6 cm）+ 恒等朝向，方向是反的，
#     相机被装到"手腕后方并朝后看"自身几何，图像全黑。
WRIST_CAM_LOCAL = (0.15, 0.0, 0.0)
# 相机注视点：从 link7 沿工具轴前方这么远（米）。0.13 ≈ 指尖位置，
# 让相机从侧上方俯视夹爪；调大（如 0.25）则更朝前看盒子可能出现的位置。
WRIST_CAM_LOOKAT_FORWARD = 0.13


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
            # 真机 kp/kd（按关节给定，见 REAL_ARM_JOINT_DRIVE / REAL_FINGER_DRIVE）。
            # 这里只是导入时"烘焙"进 USD 的初值：运行期真正生效的是
            # ArticulationCfg.actuators（它还带力矩/速度上限，本处无法表达）。
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=_merged_drive("stiffness"),
                damping=_merged_drive("damping"),
            ),
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

# -----------------------------------------------------------------------------
# 执行器 / 关节驱动参数 —— **真机参数**（OpenArm v1.0，达妙 DM8009/DM4340/DM4310）
# -----------------------------------------------------------------------------
# 2026-10（sim-to-real）：这里原来是"隐式 PD 刚度 2000 N·m/rad、阻尼 100，力矩与
# 速度都不设限"。那等于把真机 30~200 倍柔、且明显有力矩/速度上限的关节建模成近乎
# 刚性伺服，策略因此学会"把目标推到关节限位之外、靠限位饱和来甩臂"（实测 checkpoint
# 的 `last_action` 统计 σ≈4~6，即原始动作常年 ±5 以上）。换成下表参数后动力学变了，
# **旧 checkpoint 不能直接续训/播放，需要重训**。
#
# 数值来源（真机侧一手文件，均在 openarm_ws/src/openarm_description 下）：
#   * kp / kd：`assets/robot/openarm_v1.0/config/arm/control_gains.yaml`
#   * 力矩 / 速度上限：`assets/robot/openarm_v1.0/config/arm/joint_limits.yaml`
#     （关节侧数值，已折算减速比；左右臂同值）
#   * 夹爪不在这两个文件里，按电机额定值 + 传动比换算，见 REAL_FINGER_DRIVE 注释
#
# ⚠️ 关于 ImplicitActuator 的两套"上限"（Isaac Lab 3.0 语义，别混淆）：
#   * `joint_effort_limit` / `joint_velocity_limit` —— **求解器级**硬钳位，是物理上
#     真正生效的力矩/速度上限；
#   * `actuator_effort_limit` / `actuator_velocity_limit` —— 电机额定值，只用于填充
#     soft limits（终止项/奖励项读它）与遥测，本身不钳位。
# 为了让仿真真的"发力/提速受限"，两者都设成同一个真机值。
REAL_ARM_JOINT_DRIVE = {
    # 关节后缀: (kp [N·m/rad], kd [N·m·s/rad], 力矩上限 [N·m], 速度上限 [rad/s])
    "joint1": (70.0, 2.75, 40.0, 16.754666),  # DM8009，大电机
    "joint2": (70.0, 2.50, 40.0, 16.754666),  # DM8009
    "joint3": (70.0, 2.00, 27.0, 5.445426),   # DM4340，中电机
    "joint4": (60.0, 2.00, 27.0, 5.445426),   # DM4340
    "joint5": (10.0, 0.70, 7.0, 20.943946),   # DM4310，小电机
    "joint6": (10.0, 0.60, 7.0, 20.943946),   # DM4310（行程仅 ±45°，最易饱和）
    "joint7": (10.0, 0.50, 7.0, 20.943946),   # DM4310
}

# 夹爪：真机是**单个**电机 + 丝杠（不是仿真里的两个平移关节），MIT 帧用
# kp=50 N·m/rad、kd=1.0（`openarm_ws/README/deploy_packages/scripts/vr_bridge.py`
# 的 `send_mit_frame(bus, 0x08, finger, 0.0, 50.0, 1.0, ...)`）；指令范围
# −60°…−10°（共 50° = 0.8727 rad）对应手指行程 0…0.044 m，故传动比
#   r = 0.044 / 0.8727 = 0.05042 m/rad
# 把电机侧（旋转）量换算到关节侧（平移）：
#   k[N/m]   = kp / r²      ≈ 19690    (= 50 / 0.05042²)
#   c[N·s/m] = kd / r²      ≈   394    (= 1.0 / 0.05042²)
#   F[N]     = τ_max / r    ≈   198    (电机 10 N·m)
#   v[m/s]   = ω_max · r    ≈   1.51   (电机 30 rad/s)
# ⚠️ 这组是**换算值**而非实测值，k 又直接决定夹紧盒子时的接触刚度 —— 上真机前建议
#    用夹爪阶跃响应 / 夹持力做一次 sysid 校准。
REAL_FINGER_DRIVE = (19690.0, 393.6, 198.0, 1.513)


def _arm_drive_dicts(side: str) -> dict[str, dict[str, float]]:
    """把 :data:`REAL_ARM_JOINT_DRIVE` 展开成 ImplicitActuatorCfg 需要的按关节字典。

    键必须是**完整关节名**：Isaac Lab 用 ``re.fullmatch`` 匹配键，且 ``strict=True``
    要求每个键都命中该组内的关节 —— 所以左右臂各建一份，不能共用。
    """

    def per_joint(idx: int) -> dict[str, float]:
        # REAL_ARM_JOINT_DRIVE 的键已经含 "joint" 前缀（如 "joint1"）
        return {f"openarm_{side}_{n}": v[idx] for n, v in REAL_ARM_JOINT_DRIVE.items()}

    effort = per_joint(2)
    velocity = per_joint(3)
    return {
        "stiffness": per_joint(0),
        "damping": per_joint(1),
        "joint_effort_limit": effort,
        "actuator_effort_limit": effort,
        "joint_velocity_limit": velocity,
        "actuator_velocity_limit": velocity,
    }


def _finger_drive_dicts(side: str) -> dict[str, dict[str, float]]:
    """夹爪两个平移手指关节的驱动参数（同一份换算值，见 :data:`REAL_FINGER_DRIVE`）。"""
    stiffness, damping, effort, velocity = REAL_FINGER_DRIVE
    names = [f"openarm_{side}_finger_joint1", f"openarm_{side}_finger_joint2"]
    return {
        "stiffness": dict.fromkeys(names, stiffness),
        "damping": dict.fromkeys(names, damping),
        "joint_effort_limit": dict.fromkeys(names, effort),
        "actuator_effort_limit": dict.fromkeys(names, effort),
        "joint_velocity_limit": dict.fromkeys(names, velocity),
        "actuator_velocity_limit": dict.fromkeys(names, velocity),
    }


LEFT_ARM_DRIVE = _arm_drive_dicts("left")
RIGHT_ARM_DRIVE = _arm_drive_dicts("right")
LEFT_FINGER_DRIVE = _finger_drive_dicts("left")
RIGHT_FINGER_DRIVE = _finger_drive_dicts("right")


def _merged_drive(param: str) -> dict[str, float]:
    """合并两侧臂+夹爪的某个参数（给 URDF 导入器烘焙初值用，它不分侧）。"""
    merged: dict[str, float] = {}
    for group in (LEFT_ARM_DRIVE, RIGHT_ARM_DRIVE, LEFT_FINGER_DRIVE, RIGHT_FINGER_DRIVE):
        merged.update(group[param])
    return merged


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
            # 左右臂各 7 个转动关节：**真机 kp/kd + 力矩/速度上限**（见上方
            # REAL_ARM_JOINT_DRIVE 的注释与数值来源），不再是 2000/100 的刚性伺服。
            # 换参数后旧 checkpoint 作废，需要重训。
            "left_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_joint[1-7]"],
                **LEFT_ARM_DRIVE,
            ),
            "right_arm": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_joint[1-7]"],
                **RIGHT_ARM_DRIVE,
            ),
            # 左右手两个平移手指关节：单电机 + 丝杠，kp/kd 与力矩/速度上限按
            # 传动比换算到关节侧（见 REAL_FINGER_DRIVE 注释）。
            "left_fingers": ImplicitActuatorCfg(
                joint_names_expr=["openarm_left_finger_joint[1-2]"],
                **LEFT_FINGER_DRIVE,
            ),
            "right_fingers": ImplicitActuatorCfg(
                joint_names_expr=["openarm_right_finger_joint[1-2]"],
                **RIGHT_FINGER_DRIVE,
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

    # --- 光照（三层：环境 + 主光 + 补光）---
    # 之前只有一盏 1200 强度的穹顶光 —— 均匀无方向，没有阴影层次，浅色物体还会被
    # "洗白"，这正是观感偏假的主要原因。改为：
    #   dome  : 环境填充光（弱，模拟室内漫反射）
    #   key   : 主光（方向光，斜向照下，产生明确阴影与高光）
    #   fill  : 补光（另一侧低强度，避免阴影死黑）
    # 强度经过实测标定：三盏灯合计过强会让画面均值冲到 ~238（接近过曝、丢细节）。
    # 当前档位：**环境 40 / 主光 300 / 补光 320** —— 前视整体亮度约 200（比最亮档低 ~10%），
    # 明暗对比更强、阴影更明显，且过曝 0%、死黑 0%。
    # 历史档位（供回退/对比）：
    #   1200 单灯  ->  环境220+主光1100  ->  环境150+主光1100  ->  环境120+主光900  ->  当前
    # ⚠️ 调整灯光后要给渲染留几帧收敛（RTX 时序降噪需要多帧），否则截到的图会偏亮。
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        # 环境光（填充光）：调暗一些可提高明暗对比、让阴影更明显；太亮会把画面整体冲白。
        spawn=sim_utils.DomeLightCfg(intensity=40.0, color=(1.0, 0.99, 0.97)),  # 微暖的环境光（F 档）
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=300.0, color=(1.0, 0.98, 0.94), angle=12.0),  # 主光（F 档）
        # 方向光沿自身 -Z 发射；默认朝下。绕 X 轴倾斜一点 -> 光从前方斜上方来，
        # 让盒子/桌面产生方向性阴影（旋转由四元数给出）。
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3007, 0.0, 0.0, 0.9537)),  # ≈ 绕 X 转 35°
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.DistantLightCfg(intensity=320.0, color=(0.97, 0.98, 1.0), angle=25.0),
        # 另一侧补光，减轻背光面过暗
        init_state=AssetBaseCfg.InitialStateCfg(rot=(-0.2164, -0.2164, -0.0475, 0.9511)),
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

    # 背景墙占位字段（enable_backdrop=True 时由 __post_init__ -> _setup_backdrop() 填充）。
    # 四面各一个实体、每环境一套；纯视觉（无碰撞体），照片材质在 startup 时绑定。
    backdrop_plus_x: AssetBaseCfg | None = None
    backdrop_minus_x: AssetBaseCfg | None = None
    backdrop_plus_y: AssetBaseCfg | None = None
    backdrop_minus_y: AssetBaseCfg | None = None


# -----------------------------------------------------------------------------
# Observations
# -----------------------------------------------------------------------------

@configclass
class StateObservationsCfg:
    """状态观测（**默认 62 维**）：不含盒子的线速度/角速度。

    排布::

        0-17  joint_pos(18) | 18-35 joint_vel(18) | 36-38 parcel_pos(3)
       39-42  parcel_quat(4) | 43-45 label_normal(3) | 46-61 last_action(16)

    为什么去掉盒子速度（6 维）：真机上盒子速度是最难获得的量（需要多帧差分 +
    滤波，噪声大且带延迟）。消融实验（``scripts/vel_ablation.py``）显示：把
    盒子速度置零，成功率从 93.4% 只掉到 89.8%（−3.5pp），而加 30% 噪声几乎无
    影响（−0.4pp）——即策略对速度的精度要求极低，属于"锦上添花"而非"必需"。
    去掉后感知系统只需输出"位置 + 姿态"，单帧 CNN 即可，部署大幅简化。

    注意：关节速度 18 维**不能去**（消融显示掉 14.1pp），但它恰好是真机上最
    容易获得的量（编码器差分/驱动器直接提供）。
    """

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
        # 盒子位姿（env-local 世界系）：位置 + 姿态 —— 真机靠视觉/标记估计
        parcel_pos = ObsTerm(func=mdp_observations.root_pos_w, params={"asset_cfg": SceneEntityCfg("parcel")})
        parcel_quat = ObsTerm(
            func=mdp_observations.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("parcel"), "make_quat_unique": False},
        )
        # 标签面（盒子局部 +Z）法向的世界系分量 —— 任务的目标量（朝上 = (0, 0, 1)）
        label_normal = ObsTerm(func=mdp.parcel_label_normal_w, params={"asset_cfg": SceneEntityCfg("parcel")})
        # 上一步动作（16 维）：给策略"动作平滑 / 惯性"先验
        last_action = ObsTerm(func=mdp_observations.last_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class StateObservationsFullCfg:
    """状态观测（**68 维，兼容旧 checkpoint**）：额外含盒子线速度/角速度。

    排布::

        0-17  joint_pos(18) | 18-35 joint_vel(18) | 36-38 parcel_pos(3)
       39-42  parcel_quat(4) | 43-45 parcel_lin_vel(3) | 46-48 parcel_ang_vel(3)
       49-51  label_normal(3) | 52-67 last_action(16)

    2026-09 之前训练的所有 checkpoint（含 09-16 那版 93% 的 model_20000）都用
    这个排布。要播放/续训那些旧模型时，把 ``OpenArmFlipEnvCfg.use_parcel_vel``
    设为 True 即可（``play.py`` / ``eval_model.py`` / ``train.py`` 会依据
    checkpoint 的输入维度自动判定，无需手动设置）。

    字段顺序必须与旧 checkpoint 完全一致（不能靠继承追加，否则拼接顺序变化
    会让旧权重对错位置）。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy (68-dim, legacy order)."""

        joint_pos = ObsTerm(
            func=mdp_observations.joint_pos_limit_normalized,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)},
        )
        joint_vel = ObsTerm(
            func=mdp_observations.joint_vel,
            scale=0.1,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)},
        )
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
        label_normal = ObsTerm(func=mdp.parcel_label_normal_w, params={"asset_cfg": SceneEntityCfg("parcel")})
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

    # --- 场景表面纹理（startup 一次）---
    # 给盒子/标签/桌面/地面贴上程序化纹理（tools/make_textures.py 生成，随仓库分发）。
    # 材质建立在 /World/Looks 下全局共享；几何体的 UV 由资产自己提供
    # （assets/parcel/parcel.usda 里 author 了 primvars:st；否则纹理采样会失败 → 渲染成灰白）。
    apply_textures = EventTerm(
        func=mdp.apply_scene_textures,
        mode="startup",
        params={
            "material_bindings": [
                # 快递盒：**用纯色**（assets/parcel/parcel.usda 里的 primvars:displayColor），
                # 不绑纹理 —— 纸箱纹理试过几轮，观感反而更假（且容易做出木纹感）。
                # 想换回纹理时，把下面这段加回 material_bindings 即可（纹理可用
                # `python3 tools/make_textures.py` 重新生成 cardboard.png）：
                #   {"looks_name": "CardboardMat", "texture": "cardboard",
                #    "targets": ["{ENV_REGEX_NS}/Parcel/Body"],
                #    "roughness": 0.85, "metallic": 0.0, "uv_scale": (1.0, 1.0)},
                {
                    "looks_name": "LabelMat",
                    "texture": "express_label",
                    "targets": ["{ENV_REGEX_NS}/Parcel/Label"],
                    "roughness": 0.35,  # 面单是印刷纸，比瓦楞纸略光
                    "metallic": 0.0,
                    "uv_scale": (1.0, 1.0),
                },
                {
                    "looks_name": "TableMat",
                    "texture": "table",
                    "targets": ["{ENV_REGEX_NS}/Table"],
                    "roughness": 0.9,  # 哑光台面：高粗糙度才不会有明显高光反射
                    "metallic": 0.0,
                    "uv_scale": (6.0, 6.0),
                },
                {
                    # 注意：GroundPlane 资产真正的可见几何是
                    # /World/GroundPlane/Environment/Geometry（自带上网格线材质），
                    # 直接绑到 /World/GroundPlane 只会作用到外层 Xform 与碰撞平面，
                    # 地面仍显示 Isaac 默认的蓝青色网格 —— 必须指向这个具体几何。
                    "looks_name": "FloorMat",
                    "texture": "floor",
                    "targets": ["/World/GroundPlane/Environment/Geometry"],
                    "roughness": 0.9,
                    "metallic": 0.0,
                    "uv_scale": (12.0, 12.0),
                },
            ]
        },
    )

    # --- 腕部相机位姿同步（每个策略步一次） ---
    # 为什么必须手动同步：Isaac Lab 在 fabric 模式下不回写 link 的 USD 变换，挂在
    # link 下的相机 prim 生成后位姿即被冻结（实测：机械臂移动 4.4 cm、相机位移
    # 0.00 cm），图像全黑。这里每步用 link 的**物理**位姿直接设置相机世界位姿
    # （见 mdp.events.sync_wrist_cameras）。
    # interval_range_s=(0,0) + is_global_time=True -> 每步触发、对所有环境一次性更新。
    sync_wrist_cameras = EventTerm(
        func=mdp.sync_wrist_cameras,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "camera_bindings": {
                "wrist_left_cam": "openarm_left_link7",
                "wrist_right_cam": "openarm_right_link7",
            },
            "local_offset": WRIST_CAM_LOCAL,  # 局部 +X 15 cm（夹爪侧上方）
            "look_at_forward": WRIST_CAM_LOOKAT_FORWARD,  # 注视 link7 工具轴前方 13 cm（指尖处）
        },
    )

    # --- 背景墙照片随机化（interval：每隔一段时间给每面墙换一张真实场景照片） ---
    # 目的：让视觉策略/感知 CNN 学不到"某面墙总是某种颜色"这种线索（那种线索在真机上
    # 一定失效）。材质在 /World/Looks 下**全局共享**，这里只改纹理文件路径，开销极小；
    # 但也因为共享，它是"全局换图"而不是"每个环境各换一张"。
    # 参数（wall_looks / images）由 OpenArmFlipEnvCfg.__post_init__ 按
    # assets/textures/env/ 的实际内容填充；照片为空时本项自动什么都不做。
    randomize_backdrop = EventTerm(
        func=mdp.randomize_backdrop_textures,
        mode="interval",
        interval_range_s=(10.0, 25.0),
        is_global_time=True,
        params={"wall_looks": [], "images": []},
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
    lean_cameras: bool = False
    """裁剪相机配置以加速**视觉策略训练**（默认 False，保持原行为）。

    开启后：
    * 只创建**前视相机**（策略唯一使用的相机），不创建左右腕部相机；
    * 前视相机只渲染 ``rgb``，渲染 ``depth``（``distance_to_image_plane``）。

    实测吞吐（``scripts/bench_vision.py``，视觉任务 ``OpenArm-Flip-v0``）：

    ==========  ==================  ==============
    num_envs    当前配置             lean（提速）
    ==========  ==================  ==============
    64          204 env-steps/s     495（2.4×）
    256         —                   1696
    512         —                   2790
    ==========  ==================  ==============

    对 20000 次 PPO 迭代的推算：512 envs + lean ≈ **24.5 小时**
    （对比状态任务同规模约 14 小时，约 1.7 倍）。用 ``train.py --lean_cameras`` 开启。
    """
    enable_contact_rewards: bool = True
    """生成盒子/手指接触传感器并启用一次性 touch/grasp 阶段奖励
    （见 mdp.rewards.parcel_contact_stage_bonus）。"""
    enable_backdrop: bool = True
    """在每个环境四周立四面贴**真实场景照片**的背景墙（见 BACKDROP_* 常量）。

    墙是纯视觉几何（无碰撞体）：不影响物理、不影响 62/68 维状态观测，只影响渲染
    —— 也就是只影响视觉策略与感知数据的画面。照片来自 ``assets/textures/env/``
    （CC0，可用 ``OPENARM_ENV_TEX_DIR`` 换成自己现场拍的图）；目录为空时自动跳过，
    场景回到"灰白空世界"。
    """
    enable_backdrop_floor: bool = False
    """把地面也换成**真实地面照片**（``assets/textures/env/floor_*.jpg``）。

    默认 False：地面继续用程序化 ``floor.png``（现有外观不变）。开启后地面改用
    ``floor_*`` 里排序第一张照片（木地板/水泥地），配合背景墙构成完整"真实房间"。
    """
    use_parcel_vel: bool = False
    """状态观测是否包含盒子的线速度/角速度。

    * ``False``（默认）：62 维观测（``StateObservationsCfg``）——为 sim-to-real
      设计，真机只需估计盒子位置+姿态（单帧视觉即可），不必求速度。
    * ``True``：68 维观测（``StateObservationsFullCfg``）——与 2026-09 之前
      训练的旧 checkpoint 兼容（如 09-16 的 93% 模型）。

    ``play.py`` / ``eval_model.py`` / ``eval_breakdown.py`` / ``train.py`` 会读取
    checkpoint 的输入维度自动设置本开关，一般无需手动改。
    """
    observations: StateObservationsCfg | StateObservationsFullCfg | ImageObservationsCfg = StateObservationsCfg()
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
        elif self.use_parcel_vel:
            # 68 维：含盒子速度，兼容 2026-09 之前的旧 checkpoint
            self.observations = StateObservationsFullCfg()
        else:
            # 62 维（默认）：不含盒子速度，便于真机部署（只需位置+姿态感知）
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
        if self.enable_backdrop:
            self._setup_backdrop()

    def _setup_backdrop(self) -> None:
        """在每个环境四周立四面背景墙，并给它们绑定真实场景照片。

        做三件事：
        1. 按 :data:`BACKDROP_WALLS` 的位姿 spawn ``assets/scene/backdrop_wall.usda``
           （每个环境一套、四面各一个实体；**无碰撞体**，纯视觉）；
        2. 把照片材质加进 ``apply_textures``（startup 事件）的绑定列表 —— 材质统一建在
           ``/World/Looks`` 下全局共享，避免每个环境复制一份；
        3. 把随机化事件 ``randomize_backdrop`` 的参数填上（照片列表 + 材质名），
           让训练中定期换图。

        找不到照片时不报错、只打印一行提示（场景退化为灰白背景，其它一切照常）。
        """
        images = backdrop.list_wall_images()
        if not images:
            print(
                "[BACKDROP] 没找到真实场景照片，背景墙不生成。把图片放进 "
                f"{backdrop.DEFAULT_ENV_TEX_DIR} 或设置 OPENARM_ENV_TEX_DIR 后重启。",
                flush=True,
            )
            return
        if not Path(BACKDROP_WALL_USD).exists():
            raise FileNotFoundError(
                f"Backdrop wall USD not found at {BACKDROP_WALL_USD}. "
                "Regenerate it with `python3 tools/make_backdrop_usd.py`."
            )

        for wall in BACKDROP_WALLS:
            field = BACKDROP_FIELD_NAMES[wall.name]
            # 每个墙用独立的 spawn cfg 实例：场景管理器会往 spawn 里写 spawn_path，
            # 共享同一个实例会互相覆盖。
            setattr(
                self.scene,
                field,
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/" + wall.name,
                    spawn=sim_utils.UsdFileCfg(usd_path=str(BACKDROP_WALL_USD)),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=wall.center,
                        # 面板局部 +Z 是法向，转到"朝向工作区"
                        rot=backdrop.plane_quat_xyzw(wall.inward_normal),
                    ),
                ),
            )

        bindings = self.events.apply_textures.params.setdefault("material_bindings", [])
        bindings.extend(
            backdrop.material_bindings(
                BACKDROP_WALLS,
                images,
                target_prefix="{ENV_REGEX_NS}",  # 每环境一套墙
            )
        )
        if self.enable_backdrop_floor:
            floors = backdrop.list_floor_images()
            if floors:
                for spec in bindings:
                    if spec["looks_name"] == "FloorMat":
                        spec["texture"] = str(floors[0])
                print(f"[BACKDROP] 地面换成真实照片 {floors[0].name}", flush=True)
            else:
                print("[BACKDROP] enable_backdrop_floor=True 但没找到 floor_* 照片，地面保持原样",
                      flush=True)
        self.events.randomize_backdrop.params.update(
            wall_looks=[f"{w.name}Mat" for w in BACKDROP_WALLS],
            images=[str(p) for p in images],
        )
        print(backdrop.summary(BACKDROP_WALLS, images), flush=True)

    def _setup_cameras(self) -> None:
        """Create the front + wrist cameras (same mounts as the demo scene)."""
        # 腕部相机：24 mm 长焦，只看 15 cm 外的指尖/夹爪，不需要广角
        wrist_cam_spawn = sim_utils.PinholeCameraCfg(
            focal_length=WRIST_CAM_FOCAL,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        )
        # 前视相机：12 mm 广角（水平视场 ~82°），保证翻转全过程盒子都在画面内
        front_cam_spawn = sim_utils.PinholeCameraCfg(
            focal_length=FRONT_CAM_FOCAL,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        )
        # 机器人正前方的固定相机，斜向下看桌上的盒子；
        # 若它还喂给策略图像观测（enable_image_obs），就用低分辨率以省渲染+CNN 开销
        # （视场由焦距决定，与分辨率无关 —— 降分辨率只降像素密度，不缩视野）
        cam_w, cam_h = (
            (POLICY_CAM_WIDTH, POLICY_CAM_HEIGHT)
            if self.enable_image_obs
            else (FRONT_CAM_WIDTH, FRONT_CAM_HEIGHT)
        )
        # lean_cameras：只建策略真正需要的相机与数据类型 —— 视觉 RL 训练必用。
        # 实测（scripts/bench_vision.py，64 envs）：204 -> 495 env-steps/s（2.4 倍提速）。
        # 原因：腕部相机（2×320×240）与 depth 都不喂给策略，却每步都在渲染。
        front_data_types = ["rgb"] if self.lean_cameras else CAM_DATA_TYPES
        self.scene.front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/FrontCamera",
            update_period=0.0,
            height=cam_h,
            width=cam_w,
            data_types=front_data_types,
            spawn=front_cam_spawn,
            offset=CameraCfg.OffsetCfg(
                pos=FRONT_CAM_POS, rot=FRONT_CAM_ROT_XYZW, convention="opengl"
            ),
        )
        if self.lean_cameras:
            return  # 腕部相机不参与策略观测，训练时直接不创建
        # 眼在手：装在 link7 局部 +X 15 cm（见 WRIST_CAM_LOCAL 注释），俯视指尖/盒子。
        # 注意：实测这两台相机当前**看不到盒子**（视野被夹爪占据，盒子在相机后方），
        # 采集数据集时不要指望它们；要么调 WRIST_CAM_LOOKAT_FORWARD/安装位，要么只用前视。
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
                    spawn=wrist_cam_spawn,
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
