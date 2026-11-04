# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm 遥操作环境（Teleop），用于用键盘 / SpaceMouse / 游戏手柄（后续可换 XR 头显）
直接手动操作机械臂，而不是训练策略。

与 RL 任务（``OpenArm-Flip-*-v0``）的区别：

* **动作空间换成 IK**：RL 任务是"关节位置目标"（16 维耦合夹爪），遥操作需要
  "末端位姿 → IK → 关节目标"，因此用
  :class:`DifferentialInverseKinematicsActionCfg`（**绝对位姿模式** + DLS IK；
  不用相对增量模式的原因见该类 docstring —— 相对模式会因 PD 重力静差累积下坠）。
* **动作维度 8**：7 维绝对目标位姿 ``[pos(3), quat(4)]`` + 1 维夹爪二元指令。
  设备（``Se3Keyboard`` / ``Se3SpaceMouse`` / ``Se3Gamepad``）原生输出的是 7 维
  ``[dx,dy,dz, rx,ry,rz, gripper]``（6 维增量 + 夹爪），由
  ``scripts/teleop_openarm.py`` 把增量累加到固定 target 后转成绝对命令。
  动作项顺序必须"臂在前、夹爪在后"。
* **单臂先行**：目前只驱动**右臂**（7 关节）+ 右夹爪，左臂由 actuator 保持在 home
  位姿。双臂需要两个设备 / 两路 IK，待 XR 双手柄接入时再扩展。
* **不训练**：num_envs=1、关闭相机与图像观测、回合长度设为极大（不会自动超时重置，
  用 'R' 键手动重置）；奖励项全部清空、终止只保留物理发散/NaN 保护。

用法::

    # 键盘（推荐先跑这个）
    python.sh scripts/teleop_openarm.py --teleop_device keyboard

    # SpaceMouse（3D 鼠标）
    python.sh scripts/teleop_openarm.py --teleop_device spacemouse

后续接 PICO 4 Ultra（CloudXR.js / WebXR）：XR 遥操作走 IsaacTeleop 栈
（``--xr`` + ``isaacteleop`` extra），会把同一套 IK 动作项作为落点，只是输入源换成头显。
"""
from __future__ import annotations

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg
from isaaclab.envs.mdp import terminations as mdp_terminations
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass

from . import mdp
from .openarm_flip_env_cfg import (
    OpenArmFlipEnvCfg,
)

# isaaclab_teleop / isaacteleop 只在接入 XR 头显（PICO/Quest via CloudXR）时才需要，
# 且它们不随本项目的基础安装一起进来。这里条件导入：装了就启用 XR 配置，没装也
# 不影响键盘 / SpaceMouse 遥操作与 RL 训练。
#   - isaaclab_teleop：Isaac Lab 仓库里的包（source/isaaclab_teleop），需要加进
#     PYTHONPATH 或 pip install -e；
#   - isaacteleop：pip 包，**务必用 --no-deps 安装**（它的依赖声明会让 pip 升级
#     torch/numpy/CUDA 包，破坏 Isaac Sim 自带的环境；Isaac Sim 自带的
#     numpy 2.3.1 / scipy 1.17 / torch 2.11 已满足其要求）。
try:
    from isaaclab_teleop import IsaacTeleopCfg
    from isaaclab_teleop.xr_cfg import XrCfg

    _XR_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - 取决于是否安装了 XR 依赖
    IsaacTeleopCfg = None  # type: ignore[assignment]
    XrCfg = None  # type: ignore[assignment]
    _XR_AVAILABLE = False

##
# 遥操作参数
##

# XR 视角预设：决定"你戴着头显时，虚拟视角在哪里、朝哪看"。
#   pos = anchor_pos：**映射到头显空间原点的那一点**（通常给地面高度 z=0，
#         你的头显高度 ~1.6 m 会自动叠加上去）
#   rot = anchor_rot：仿真相对头显原点基准朝向的旋转（xyzw）。null（0,0,0,1）时
#         面向仿真 +X；绕 Z 轴 ±90° 可让侧面站位也朝向机器人。
# 场景布局：机器人基座在原点，桌面盒子在 (0.30, 0.10, 0.38)（+X 方向）。
XR_VIEW_PRESETS: dict[str, dict[str, tuple[float, ...]]] = {
    # 站在机器人**后方** 0.9 m、人眼高度，面向 +X：机器人与前方桌面都在视野内（默认）
    "behind": {"pos": (-0.9, 0.0, 0.0), "rot": (0.0, 0.0, 0.0, 1.0)},
    # 站在机器人**左侧**（+Y 侧）1.0 m，朝向 -Y（即看向机器人/桌面）
    "side_left": {"pos": (0.15, 1.0, 0.0), "rot": (0.0, 0.0, -0.7071068, 0.7071068)},
    # 站在机器人**右侧**（-Y 侧）1.0 m，朝向 +Y
    "side_right": {"pos": (0.15, -1.0, 0.0), "rot": (0.0, 0.0, 0.7071068, 0.7071068)},
    # 站在机器人**斜后方**（左后 45°）1.1 m：能同时看到机械臂侧面与桌面
    "behind_left": {"pos": (-0.75, 0.75, 0.0), "rot": (0.0, 0.0, -0.3826834, 0.9238795)},
    # 旧的俯视视角（机器人斜上方 1.8 m 往下看），留作对照
    "overhead": {"pos": (0.30, 0.0, 0.20), "rot": (0.0, 0.0, 0.0, 1.0)},
}
XR_DEFAULT_VIEW = "behind"

# 被遥操作的臂：先只驱动右臂（左臂保持 home）。
TELEOP_ARM_SIDE = "right"
TELEOP_ARM_JOINTS = [f"openarm_{TELEOP_ARM_SIDE}_joint[1-7]"]
TELEOP_GRIPPER_JOINTS = [f"openarm_{TELEOP_ARM_SIDE}_finger_joint[1-2]"]
# 末端工具点：OpenArm v1.0 在 fixed joint 之后没有独立 TCP，用最后一个臂连杆 link7。
TELEOP_EE_BODY = f"openarm_{TELEOP_ARM_SIDE}_link7"

# 夹爪开合目标（官方 USD 约定：0 = 闭合、0.044 = 张开）。
GRIPPER_OPEN_VALUE = 0.044
GRIPPER_CLOSE_VALUE = 0.0

# 末端工具点到指尖中点的偏移（米）。0.11 m 来自 openarm_sorting 的 EE_TO_FINGERTIP
# 实测值（link7 → 两指中点），让 IK 目标点落在"两指之间"而不是腕部：
# 这样用键盘/手柄微调后，指尖能对准物体。
EE_TO_FINGERTIP = 0.11


@configclass
class OpenArmTeleopActionsCfg:
    """8 维遥操作动作：7 维**绝对**目标位姿（IK）+ 1 维二元夹爪。

    顺序很重要：Isaac Lab 的遥操作设备输出 ``[x, y, z, rx, ry, rz, gripper]``，
    action manager 按声明顺序切分给各动作项，因此 ``arm_action`` 必须在
    ``gripper_action`` 之前。

    **为什么用绝对位姿（``use_relative_mode=False``）而不是相对增量**：
    相对模式下 IK 控制器的 ``set_command`` 会把增量加到**当前实际 EE 位姿**上
    （见 ``DifferentialIKController.set_command`` 的 ``apply_delta_pose(ee_pos, ...)``）。
    而 PD 控制存在重力静差（关节实际角总比目标低一点），于是"目标跟着实际位姿
    一起下垂"被每步固化 → 松手后机械臂**持续缓缓下坠**（实测 300 步 EE 从
    0.535 m 掉到 0.309 m）。

    绝对模式下目标由使用方（``scripts/teleop_openarm.py``）自行维护：只在按键时
    把增量累加到固定的 ``target`` 上，松手时 target 不变，IK 每步都以该固定目标
    纠偏，重力下垂被闭环消除。
    """

    # 绝对位姿 IK：命令 = [pos(3), quat(4, xyzw)]，IK 内部按当前位姿与目标之差求解。
    # ⚠️ scale 必须为 1.0：``process_actions`` 会把 scale 无条件乘到命令上，
    #    绝对模式下令其 ≠1 会把四元数也乘坏（相对模式才需要 scale 限幅）。
    arm_action: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=TELEOP_ARM_JOINTS,
        body_name=TELEOP_EE_BODY,
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        scale=1.0,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, EE_TO_FINGERTIP)),
    )

    # 二元夹爪：设备第 7 维 +1 → open_command_expr，-1 → close_command_expr。
    gripper_action: BinaryJointPositionActionCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=TELEOP_GRIPPER_JOINTS,
        open_command_expr={f"openarm_{TELEOP_ARM_SIDE}_finger_joint[1-2]": GRIPPER_OPEN_VALUE},
        close_command_expr={f"openarm_{TELEOP_ARM_SIDE}_finger_joint[1-2]": GRIPPER_CLOSE_VALUE},
    )


@configclass
class TeleopRewardsCfg:
    """遥操作不训练策略，因此**不配置任何奖励项**。

    （RL 的奖励项会读取接触传感器，而遥操作环境为省资源关闭了那些传感器；
    留空即可，也省掉每步无意义的 reward 计算。）
    """


@configclass
class TeleopTerminationsCfg:
    """遥操作只保留"安全保护"终止：物理发散 / NaN 时重置该环境。

    刻意**不启用** RL 里的 ``time_out``（遥操作回合设为极长，用 'R' 键手动重置）
    与 ``parcel_out_of_bound``（允许把盒子推下桌，练习时不必被强制中断）。
    """

    # 关节超速（>100 rad/s，正常最坏 ~40）→ 判定物理发散，重置
    bad_robot_state = DoneTerm(
        func=mdp_terminations.joint_vel_out_of_manual_limit,
        params={"max_velocity": 100.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 关节/盒子状态出现 NaN → 重置（重置后观测干净，不会把 NaN 传给上层）
    nan_state = DoneTerm(func=mdp.nan_state_guard)


def _build_openarm_teleop_pipeline():
    """构建 IsaacTeleop 重定向管线：XR 手柄 → 8 维绝对位姿动作（给 XR 遥操作用）。

    输出与 :class:`OpenArmTeleopActionsCfg` 的动作布局**完全一致**：
    ``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper_value]``
    ——前 7 维给 ``arm_action``（绝对位姿 IK），最后 1 维给 ``gripper_action``。

    管线组成（与 Isaac Lab 官方 Franka stack 任务同款）：

    * ``ControllersSource.RIGHT`` —— 右手手柄位姿（XR runtime 提供）；
    * ``Se3AbsRetargeter`` —— 手柄位姿 → EE 绝对位姿（7 维 pos+quat，xyzw，正是
      Isaac Lab 的约定，无需重排）；
    * ``GripperRetargeter`` —— 手柄扳机/捏合 → 夹爪开合标量；
    * ``TensorReorderer`` —— 把上面两路拼成单个 8 维 ``action`` 张量。

    延迟 import ``isaacteleop``：只有走 XR 时才需要它（见模块顶部说明）。
    """
    from isaacteleop.retargeters import (
        GripperRetargeter,
        GripperRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    import numpy as np

    # --- 自定义：取出右手柄按钮/扳机值 ---
    # 为什么需要它：本任务的操控方式是**离合(clutch)**——只有按住侧键时机械臂才跟随
    # 手柄的相对位移，松开则冻结（便于把手柄挪回舒适位置再继续推）。判断"是否按住"
    # 需要手柄的 squeeze 值，而现成的 retargeter 都不透出它。
    # 另外控制器输入是 Optional（手柄可能不可用），而 TensorReorderer 的 scalar 输入是
    # 必需的 —— 直接接线会报 "Cannot connect optional output ... to required input"，
    # 所以照 GripperRetargeter 的写法包一层：Optional 输入 → 必需 scalar 输出（无效时给 0）。
    from isaacteleop.retargeting_engine.interface import BaseRetargeter, RetargeterIOType
    from isaacteleop.retargeting_engine.interface.tensor_group_type import (
        OptionalType,
        TensorGroupType,
    )
    from isaacteleop.retargeting_engine.tensor_types import (
        ControllerInput,
        ControllerInputIndex,
        DLDataType,
        NDArrayType,
    )

    class SqueezeRetargeter(BaseRetargeter):
        """把右手柄的按钮/扳机值暴露成必需标量输出（供离合判断与按键调试）。

        输出顺序（也是 pipeline 动作向量末尾 6 维的顺序）::

            squeeze, trigger, primary, secondary, thumbstick_click, menu

        为什么把全部按钮都透出：实测 PICO 4 Ultra 上 ``squeeze_value`` 恒为 0
        （侧键没有映射到该字段），需要先把各字段都打出来、确认哪个字段响应按键，
        再把离合触发键切到那个字段上（见脚本的 ``--xr_clutch_button``）。
        """

        #: 输出字段名 → ControllerInput 里的索引名
        _FIELDS = (
            ("squeeze", "SQUEEZE_VALUE"),
            ("trigger", "TRIGGER_VALUE"),
            ("primary", "PRIMARY_CLICK"),
            ("secondary", "SECONDARY_CLICK"),
            ("thumbstick_click", "THUMBSTICK_CLICK"),
            ("menu", "MENU_CLICK"),
        )

        def __init__(self, hand_side: str = "right", name: str = "buttons") -> None:
            self._hand_side = hand_side
            super().__init__(name=name)

        def input_spec(self) -> RetargeterIOType:
            return {f"controller_{self._hand_side}": OptionalType(ControllerInput())}

        def output_spec(self) -> RetargeterIOType:
            return {
                "buttons": TensorGroupType(
                    "buttons",
                    [
                        NDArrayType(
                            "values",
                            shape=(len(self._FIELDS),),
                            dtype=DLDataType.FLOAT,
                            dtype_bits=32,
                        )
                    ],
                )
            }

        def _compute_fn(self, inputs, outputs, context) -> None:  # noqa: ANN001
            group = inputs[f"controller_{self._hand_side}"]
            out = outputs["buttons"]
            values = np.zeros(len(self._FIELDS), dtype=np.float32)
            # 手柄不可用（未握持/未追踪）时全为 0 = 未按键（不跟随，安全）
            if not getattr(group, "is_none", False):
                for i, (_, index_name) in enumerate(self._FIELDS):
                    idx = getattr(ControllerInputIndex, index_name, None)
                    if idx is not None:
                        values[i] = float(group[idx])
            out[0] = values

    # 输入源：手柄 + 手部追踪（重定向器会自动发现所需的 tracker）
    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    # XR 世界 → 仿真世界的锚点变换（由 IsaacTeleopDevice 提供）
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))

    squeeze = SqueezeRetargeter(hand_side="right", name="squeeze_rt")
    connected_squeeze = squeeze.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )
    # 右手手柄 → EE 绝对位姿。这里不做额外的 wrist 补偿（OpenArm 的 EE 是 link7，
    # 已通过 body_offset 前移到指尖）；手感不对时可调 target_offset_roll/pitch/yaw。
    se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    se3 = Se3AbsRetargeter(se3_cfg, name="ee_pose")
    connected_se3 = se3.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    # 右手扳机/捏合 → 夹爪开合
    gripper = GripperRetargeter(GripperRetargeterConfig(hand_side="right"), name="gripper")
    connected_gripper = gripper.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # 拼成 14 维动作张量：pos(3) + quat(4) + gripper(1) + 按钮组(6)。
    # 前 8 维是给环境的动作；后 6 维只用于脚本侧判断（离合触发键 + 按键调试）。
    # 离合(clutch)：按住触发键时机械臂才跟随手柄相对位移，松开冻结 —— 便于把手柄
    # 挪回舒适位置再继续推。
    ee_pose_elements = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
    gripper_elements = ["gripper_value"]
    # 自定义 retargeter 透出的按钮字段（顺序与 SqueezeRetargeter._FIELDS 一致）
    squeeze_elements = ["squeeze", "trigger", "primary", "secondary", "thumbstick_click", "menu"]
    reorderer = TensorReorderer(
        input_config={
            "ee_pose": ee_pose_elements,
            "gripper_command": gripper_elements,
            "squeeze_rt": squeeze_elements,
        },
        output_order=ee_pose_elements + gripper_elements + squeeze_elements,
        name="action_reorderer",
        input_types={"ee_pose": "array", "gripper_command": "scalar", "squeeze_rt": "array"},
    )
    connected_reorderer = reorderer.connect(
        {
            "ee_pose": connected_se3.output("ee_pose"),
            "gripper_command": connected_gripper.output("gripper_command"),
            # 按钮组（squeeze/trigger/primary/...）
            "squeeze_rt": connected_squeeze.output("buttons"),
        }
    )
    return OutputCombiner({"action": connected_reorderer.output("output")})


@configclass
class OpenArmTeleopEnvCfg(OpenArmFlipEnvCfg):
    """遥操作配置：复用翻面任务的场景（机械臂 + 桌子 + 快递盒），但改为手动操作。"""

    def __post_init__(self):
        # ⚠️ 三个开关必须在 super().__post_init__() 之前设置：基类会按这些开关
        # 决定是否创建相机传感器与接触传感器（在之后设置会留下已创建的相机，
        # 而 AppLauncher 未启用相机渲染 → 相机初始化报 Py_Graph 错误）。
        self.enable_image_obs = False  # 遥操作不需要策略图像观测
        self.enable_cameras = False  # 也不需要 scene 相机（只有 GUI 视口渲染）
        self.enable_contact_rewards = False  # 接触传感器是 RL 奖励用的，一并关掉

        super().__post_init__()

        # --- 单环境：遥操作只操作一台机械臂 ---
        self.scene.num_envs = 1

        # --- 观测：super() 已按 enable_image_obs=False 选好 68 维状态观测 ---
        # 但其中的 last_action 项引用了 RL 动作项的名字 "joint_pos"；遥操作的动作项
        # 叫 "arm_action"，不改会在 observation manager 初始化时 KeyError。
        self.observations.policy.last_action.params["action_name"] = "arm_action"

        # --- 动作：换成 绝对位姿 IK(7) + 二元夹爪(1) = 8 维 ---
        self.actions = OpenArmTeleopActionsCfg()

        # --- 奖励/终止：换成遥操作专用（无奖励、只留安全保护终止） ---
        self.rewards = TeleopRewardsCfg()
        self.terminations = TeleopTerminationsCfg()

        # --- 回合：设成极大，避免操作中途被超时重置（用 'R' 键手动重置） ---
        self.episode_length_s = 1.0e6

        # --- 遥操作设备：键盘 / SpaceMouse（可用 --teleop_device 选择） ---
        # 灵敏度参考 Isaac Lab 官方 stack 任务的取值（0.05 m / 0.05 rad 每步）。
        self.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
                "spacemouse": Se3SpaceMouseCfg(
                    pos_sensitivity=0.05,
                    rot_sensitivity=0.05,
                    sim_device=self.sim.device,
                ),
            }
        )

        # --- XR（PICO 4 Ultra / Quest 等，经 CloudXR.js 串流）---
        # 只有装了 isaaclab_teleop + isaacteleop 才启用；未装时配置保持 None，
        # 键盘 / SpaceMouse 路径照常可用（见本文件顶部 import 处的说明）。
        if _XR_AVAILABLE:
            # XR 锚点：``anchor_pos`` 是"**映射到头显空间原点的那一点**" —— 戴上头显后你的
            # 站位/身高（头显约 1.6 m）会叠加到它上面，所以 z 应取仿真地面高度 0，
            # 而不是把"视线高度"直接写进去（写 0.2 会让视角升到 1.8 m，变成从高处俯视）。
            # 默认 behind：站在机器人后方 0.9 m、人眼高度、面向 +X —— 机器人与其前方桌面
            # 都在视野内，比原先架在斜上方俯视自然。脚本可用 --xr_view 切换预设。
            _view = XR_VIEW_PRESETS[XR_DEFAULT_VIEW]
            self.xr = XrCfg(anchor_pos=_view["pos"], anchor_rot=_view["rot"])
            # IsaacTeleop 管线：XR 手柄 → 8 维绝对位姿动作，与 keyboard/SpaceMouse
            # 路径并存，由运行脚本决定走哪条（见 scripts/teleop_openarm.py 顶部说明）。
            self.isaac_teleop = IsaacTeleopCfg(
                pipeline_builder=_build_openarm_teleop_pipeline,
                sim_device=self.sim.device,
                xr_cfg=self.xr,
            )

        # --- 重置时把盒子放回"标签朝上"的直立姿态：遥操作练习时起点更可控 ---
        # side_prob=1.0 → 全部 90° 侧面起手（不打乱，便于重复练习同一动作）。
        self.events.reset_parcel.params["side_prob"] = 1.0
        self.events.reset_parcel.params["pos_jitter"] = 0.0
