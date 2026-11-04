# OpenArm Lab — 快递盒翻面（标签朝上）RL 训练框架

用 **Isaac Lab 3.0（manager-based 风格）+ RSL-RL（PPO）** 训练 OpenArm 双臂机械臂，
把桌上带快递标签的盒子**翻面成「标签朝上」**的强化学习项目，运行在 **Isaac Sim 6.0.1**
standalone 之上。框架自带快递标签纹理与带标签的盒子 USD，机器人默认从**官方 Enactic USD**
（`openarm_isaac_lab`）加载——该 USD 带完整的分连杆碰撞网格（手指碰撞体就是完整的
手指几何），夹爪不再“穿模”过盒子。

技术栈速览：

| 组件 | 说明 |
|---|---|
| 环境框架 | Isaac Lab 3.0 `ManagerBasedRLEnv`，Scene/Observations/Actions/Rewards/Terminations/Events 全部用 `@configclass` 声明式配置 |
| 算法 | RSL-RL 5.x PPO（MLP 与 CNN 两套策略配置） |
| 仿真 | Isaac Sim 6.0.1 standalone，`python.sh` 启动，GPU 物理（PhysX） |
| 任务 | `OpenArm-Flip-v0`（视觉/CNN）、`OpenArm-Flip-State-v0`（纯状态/MLP 基线）、`OpenArm-Flip-Play-v0`（播放配置） |

---

## 一、目录结构与逐文件说明

```
openarm_lab/                          # 项目根 = Python 包 openarm_lab（在 Isaac 扩展环境中被 import）
├── __init__.py                       # 包入口：re-export 三个任务 cfg 与 OPENARM_FLIP_ENVS
├── README.md                         # 本文档
├── TRAINING.md                       # 训练/播放/续训命令中文速查（完整命令清单）
├── openarm_flip/                     # 任务包：环境 cfg + MDP 项 + gym 注册
│   ├── __init__.py                   # gym.register 四个任务（v0 / State-v0 / Play-v0 / Teleop-v0）
│   ├── openarm_flip_env_cfg.py       # ★核心环境配置（见下方逐文件说明）
│   ├── openarm_teleop_env_cfg.py     # ★遥操作环境（IK 动作 + 键盘/SpaceMouse 设备，不训练）
│   ├── mdp/                          # MDP（奖励/观测/事件/终止/动作）自定义项
│   │   ├── __init__.py               # 聚合导出本包 MDP 项 + re-export Isaac Lab 自带事件项
│   │   ├── actions.py                # CoupledJointPositionAction（耦合夹爪动作项）
│   │   ├── rewards.py                # label_up 门控奖励、一次性 touch/grasp、上臂碰盒惩罚等
│   │   ├── observations.py           # parcel_label_normal_w（标签法向观测）
│   │   ├── events.py                 # reset_parcel_root_state（随机朝向/位置重置）等
│   │   └── terminations.py           # parcel_out_of_bound、nan_state_guard
│   └── agents/                       # （Python 3 命名空间包，无 __init__.py）
│       └── rsl_rl_ppo_cfg.py         # PPO 超参：MLP 版 + CNN 版两套 runner cfg
├── scripts/                          # 可执行脚本（用 $ISAAC_ROOT/python.sh 运行）
│   ├── train.py                      # RSL-RL PPO 训练入口（支持续训）
│   ├── play.py                       # GUI/无头播放已训 checkpoint（--cam top/side/follow；--task 自动推断）
│   ├── play_latest.py                # 自动播放"最新一次训练"的 checkpoint（免翻 logs）
│   ├── teleop_openarm.py             # ★手动遥操作 OpenArm（键盘 / SpaceMouse / 手柄）
│   ├── smoke_test.py                 # 端到端环境冒烟测试（随机动作，无策略）
│   ├── eval_model.py                 # 无头成功率评估（成功 = 标签朝上且盒子在桌上；--task 自动推断）
│   └── camera_check.py               # 相机传感器输出校验
├── configs/
│   └── rsl_rl_ppo.yaml               # PPO 超参参考清单（镜像 agents/rsl_rl_ppo_cfg.py）
├── assets/                           # 自包含资产（不依赖外部 demo 工程）
│   ├── parcel/parcel.usda            # 带标签快递盒 USD（displayColor 着色，无材质绑定）
│   ├── textures/express_label.png    # 快递标签纹理资源
│   ├── urdf/openarm_bimanual_v1.urdf # 双臂 URDF（仅 ROBOT_SOURCE="urdf" 回退路径使用）
│   ├── urdf/openarm_unimanual.urdf   # 单臂 URDF（项目附带资产，当前双臂任务未使用）
│   └── usd/openarm_bimanual_v1_lab/  # URDF→USD 转换缓存（回退路径按需重新生成）
└── logs/
    └── rsl_rl/
        ├── openarm_flip_img/<时间戳>/  # 视觉任务训练输出
        └── openarm_flip/<时间戳>/      # 纯状态任务训练输出
            ├── model_*.pt              # 每 200 轮存一个 checkpoint
            └── events.out.tfevents.*   # TensorBoard 曲线
```

### 1. `openarm_flip/` —— 任务包核心

#### `openarm_flip/__init__.py`

按 Isaac Lab 惯例（`env_cfg_entry_point` / `rsl_rl_cfg_entry_point` / `default_agent`）
注册三个 gym 任务：

* `OpenArm-Flip-v0` —— **视觉任务（默认）**：策略输入 = 18 维归一化关节位置 + 前视相机 RGB
  图像（CNN 策略），`rsl_rl_cfg_entry_point` 指向 `OpenArmFlipImgPPORunnerCfg`；
* `OpenArm-Flip-State-v0` —— 纯状态基线（**默认 62 维**状态、MLP、无相机），对应
  `OpenArmFlipPPORunnerCfg`；62 维 = 去掉盒子速度，为 sim-to-real 设计
  （旧 checkpoint 是 68 维，脚本会依据 checkpoint 自动切换，见 `checkpoint_utils.py`）；
* `OpenArm-Flip-Play-v0` —— 视觉任务的播放配置（`OpenArmFlipEnvCfg_PLAY`）。

#### `openarm_flip/openarm_flip_env_cfg.py` —— ★核心环境配置

模块内按 `Scene / Observations / Actions / Rewards / Terminations / Events / Env`
分节，全部用 `@configclass` 声明：

* **模块级常量**：
  * `ROBOT_SOURCE = "official"` —— 机器人来源开关：`"official"` 用 Enactic
    `openarm_isaac_lab` 官方 USD（默认，完整碰撞网格）；`"urdf"` 走本地
    `assets/urdf/openarm_bimanual_v1.urdf` → USD 转换（回退用）。
  * `OPENARM_USD` —— 官方 USD 绝对路径（指向 `openarm_isaac_lab` 的
    `openarm_bimanual.usd`；机器人在别的 checkout 时需改这里）。
  * `OPENARM_HOME_POS` —— 机器人 home 位姿字典（双臂各 7 关节 + 4 手指关节）。
    注意官方 USD 手指约定：**0 = 闭合，0.044 = 张开**（home 从张开 0.044 开始；
    旧 URDF 版正好相反，代码里通过读取 `default_joint_pos` 自动兼容）。
  * `CONTROLLED_JOINTS` / `OFFICIAL_FIXED_JOINTS` —— 18 个受控关节
    （双臂 14 旋转 + 4 手指平移）；官方 USD 里的 `left/right_hand`、
    `left/right_ee_tcp_joint` 4 个“固定关节”被排除在动作/观测之外。
  * `ARM_ACTION_SCALE = 0.5`（rad）、`FINGER_ACTION_SCALE = 0.05`（m）—— 动作缩放。
  * `LABEL_UP_COS = cos(15°)` —— “标签朝上”判定阈值。
  * `CONTACT_TOUCH_BONUS / CONTACT_GRASP_BONUS = 0.5` —— 一次性接触奖励额度。
  * 相机常量：`CAM_WIDTH/HEIGHT = 320×240`（展示/腕部相机）、
    `POLICY_CAM_WIDTH/HEIGHT = 96×72`（喂 CNN 的前视相机分辨率）、
    `FRONT_CAM_POS/TARGET`、`WRIST_CAM_LOCAL`。
* **`OpenArmFlipSceneCfg`**：机器人（4 组 ImplicitActuator，**已换成真机参数**：臂
  kp/kd = 70/2.75、70/2.5、70/2.0、60/2.0、10/0.7、10/0.6、10/0.5，力矩上限
  40/40/27/27/7/7/7 N·m，速度上限 16.75/16.75/5.45/5.45/20.9/20.9/20.9 rad/s；
  手指由单电机+丝杠换算得 k=19690 N/m、c=394 N·s/m、F=198 N、v=1.51 m/s ——
  详见 `REAL_ARM_JOINT_DRIVE` / `REAL_FINGER_DRIVE`）、桌子（运动学 pedestal，桌面
  z = 0.38 m）、盒子
  （`parcel.usda`，质量 0.35 kg，开接触传感器）、地面、穹顶灯，以及**相机与接触传感器
  的占位字段**（由 `EnvCfg.__post_init__` 填充）。
* **`StateObservationsCfg`（62 维，默认）/ `StateObservationsFullCfg`（68 维，兼容旧
  checkpoint）/ `ImageObservationsCfg`**：由 `OpenArmFlipEnvCfg.use_parcel_vel` 选择；
  视觉任务的 `policy` 组 = 18 关节位置、`image` 组 = 前视相机 RGB（归一化、转置为
  `(N, C, H, W)`，CNN 输入）。
* **`ActionsCfg`**：16 维动作 —— 14 个手臂关节各 1 维 + `left_gripper`/`right_gripper`
  两个耦合夹爪维（每维同时驱动同侧两个手指关节）。
* **`RewardsCfg`**：见下文「奖励表」。**门控参数**：`label_up` 密集塑形与成功奖励都
  只在盒子仍在桌上时给付（防“翻完就扔”）；**一次性奖励**权重在 `__post_init__` 里按
  `weight = bonus / dt` 重算（dt = 1/30 s），实际每次事件恰好 +0.5。
* **`TerminationsCfg`**：超时、盒子出界、关节超速（100 rad/s 防炸）、NaN 状态保护。
* **`EventCfg`**：重置/随机化。`reset_parcel` 的 `side_prob = 0.7` 表示 70% 概率标签在
  侧面（需 90° 翻）、30% 概率标签朝下（需 180° 翻）；`pos_jitter = 0.03` 为桌上
  (x, y) ±3 cm 随机放置；另有摩擦/质量/执行器增益的 startup 随机化。
* **`OpenArmFlipEnvCfg`** 三个布尔开关：`enable_image_obs`（图像观测，强制开相机）、
  `enable_cameras`（3 个相机）、`enable_contact_rewards`（阶段接触奖励与对应传感器）。
  `__post_init__` 负责按开关装配观测组、相机（`_setup_cameras`）与接触传感器
  （`_setup_contact_sensors`），并重算一次性奖励权重。
* **`OpenArmFlipStateEnvCfg`**：`enable_image_obs=False, enable_cameras=False` 的纯状态变体。
* **`OpenArmFlipEnvCfg_PLAY`**：播放配置 —— 16 envs、回合 15 s、`side_prob=1.0`（全
  是 90° 侧面）、`pos_jitter=0.02`（随机化更温和）。

#### `openarm_flip/mdp/` —— 自定义 MDP 项

* `actions.py` —— `CoupledJointPositionAction`（及其 cfg）：Isaac Lab 自带动作项是
  “每关节一维”，而平行夹爪同侧两个手指必须**联动**。本动作项把每侧两个手指关节耦合进
  **一个动作维度**（同目标位置驱动），动作缓冲布局 = `[14 独立臂关节..., left_gripper,
  right_gripper]`，缩放/偏移按关节解析；读取 `default_joint_pos` 作偏移，因此能自动
  兼容 URDF 与官方 USD 相反的手指约定。最终动作空间 = 14 + 2 = **16 维**。
* `rewards.py` —— 全部奖励项（详见「奖励表」）：
  * `label_up()`：标签法向的 z 分量（1 = 朝上，-1 = 朝下）；
  * `parcel_on_table()` / `label_up_reward()`：**门控密集塑形**（盒子掉下桌就一分不得，
    防“翻完即扔”），`label_up_success_reward()` 为稀疏成功奖励（标签朝上 <15° 且在桌上）；
  * `parcel_dropped_penalty()`、`time_penalty()`、`arm_body_contact_penalty()`（上臂
    link1–6 碰盒惩罚，腕部 link7 与指尖允许）；
  * `parcel_contact_stage_bonus`（`ManagerTermBase` 类）：touch（任一指尖首次碰盒）/
    grasp（同侧两指同时夹住）两个**一次性阶段奖励**——命中第一步返回 1.0，之后整回合
    恒为 0（内部 `_given` 标志在回合重置时清除）。
* `observations.py` —— `parcel_label_normal_w()`：盒子局部 +Z（标签面）法向的世界系
  单位向量，把它转到世界 +Z 即任务目标。
* `events.py` —— `reset_parcel_root_state()`：按 `side_prob` 抽标签面（`k` 选四个侧面
  之一或朝下）、均匀随机偏航、`pos_jitter` 抖动、按所选面落到正确静置高度、零速度；
  另有 Isaac Lab 自带事件（`reset_joints_by_offset`、摩擦/质量/增益随机化）经
  `mdp/__init__.py` re-export。
* `terminations.py` —— `parcel_out_of_bound()`（盒子出界即终止）、`nan_state_guard()`
  （物理状态 NaN 时终止该环境——RL 环境会在返回观测前重置终止环境，因此策略不会看到
  NaN）。
* `mdp/__init__.py` —— 聚合导出上述 MDP 项。

### 2. `openarm_flip/agents/rsl_rl_ppo_cfg.py`

两套 PPO runner 配置（均 rsl-rl 5.x、`num_steps_per_env=24`、`max_iterations=20000`、
`save_interval=200`、自适应学习率等）：

* `OpenArmFlipPPORunnerCfg`（MLP，`experiment_name="openarm_flip"`）—— 状态任务用，
  actor/critic 网络 `[256, 128, 64]`，`obs_groups = {"actor": ["policy"], ...}`；
* `OpenArmFlipImgPPORunnerCfg`（CNN，`experiment_name="openarm_flip_img"`）—— 视觉任务
  用，actor/critic 各带 3 层 CNN（通道 `[8, 16, 32]`，stride 2），
  `obs_groups = {"actor": ["policy", "image"], ...}`（1D 关节组 + 2D 图像组）。

两个 experiment_name 正好对应 `logs/rsl_rl/` 下的两个日志目录。

### 3. `scripts/`

* `train.py` —— PPO 训练入口：解析 CLI → 从 gym spec 的 entry point 解析 env/agent cfg
  → `handle_deprecated_rsl_rl_cfg` 清理过时字段 → 建 `logs/rsl_rl/<experiment>/<时间戳>/`
  → 创建环境并包上 `RslRlVecEnvWrapper` → `OnPolicyRunner.learn()`。支持
  `--checkpoint` 续训；捕获 NaN 报错时打印物理量诊断（哪个关节/盒子 NaN）。
  注意 Isaac Lab 3.0 的 AppLauncher **只有显式传 `visualizer="kit"` 才开 GUI**——
  脚本在非 `--headless` 时自动加上。
* `play.py` —— 播放已训 checkpoint（默认任务 `OpenArm-Flip-Play-v0`）。GUI 下支持
  `--cam top|side|follow` 相机预设（正上方看标签 / 经典 3/4 视角 / 高机位跟拍），
  无头模式则只滚动输出“标签朝上”的环境步数统计。
* `smoke_test.py` —— 无策略端到端验证：建环境、打印观测组 shape 与动作维度、跑若干步
  随机动作看 label_up 均值和回报、再触发一次完整 reset 检查随机化生效。
* `eval_model.py` —— 无头成功率评估：回放 `episodes × num_envs` 个回合，
  **成功判据 = 标签法向与 +Z 夹角 < 15° 且盒子仍在桌上**（掉桌不算成功），并统计
  机械臂动作占比、盒子被推动占比、全程最大 label_up。
* `camera_check.py` —— 校验场景相机（前视 + 左右腕）的 RGB/深度输出 shape 与数值统计，
  排查渲染/相机安装问题。

### 4. 资产与配置

* `assets/parcel/parcel.usda` —— 手写的带标签盒子 USD：`/Parcel/Body` 为
  0.09×0.09×0.07 m 立方体（唯一碰撞体，RigidBody），`/Parcel/Label` 是贴在局部 +Z 面上
  的**纯视觉**标签薄片。颜色用 `primvars:displayColor` 顶点色（盒身浅棕、标签白色）
  而**不绑定材质**——Isaac Sim 的 RTX 渲染对原生 Cube 几何的材质绑定不生效（会渲染成
  灰/白色），`displayColor` 则始终有效。局部 +Z 面即任务要转到朝上的“标签面”。
* `assets/textures/express_label.png` —— 快递标签纹理资源（随框架分发；当前
  `parcel.usda` 用 displayColor 纯色画标签，未绑定该贴图，纹理保留备用）。
* `assets/urdf/` —— `openarm_bimanual_v1.urdf`（双臂，仅 `ROBOT_SOURCE="urdf"` 回退用）、
  `openarm_unimanual.urdf`（单臂，附带资产，当前任务未用）。
* `configs/rsl_rl_ppo.yaml` —— PPO 超参参考清单，与 `agents/rsl_rl_ppo_cfg.py` 镜像
  （训练实际以注册的 agent cfg 为准，yaml 供对照查阅）。
* `logs/` —— 训练输出：TensorBoard events + `model_*.pt` checkpoint。
* `TRAINING.md` —— 训练/播放/续训的完整命令清单与参数说明（中文速查）。

---

## 二、任务规格

| 项目 | 内容 |
|---|---|
| 任务 id | `OpenArm-Flip-v0`（视觉/CNN）· `OpenArm-Flip-State-v0`（纯状态/MLP）· `OpenArm-Flip-Play-v0`（播放配置） |
| 机器人 | OpenArm v1.0 **双臂**（官方 USD），22 个 USD 关节中 18 个受控（2×7 旋转臂 + 2×2 平移手指）；`*_hand`/`*_ee_tcp` 4 个固定关节排除在外。手指约定：**0 = 闭合，0.044 = 张开**（home 张开 0.044） |
| 动作 | 关节位置目标（相对 home 偏移）：**16 维** = 14 臂关节（各 ±0.5 rad）+ 2 个耦合夹爪维（左右各一，同侧两指联动、同目标，每指 ±0.05 m），隐式 PD 控制 |
| 物体 | 刚性盒 0.09 × 0.09 × 0.07 m，标签在局部 +Z 面（初始竖直静置中心约在桌上 (0.30, 0.10)） |
| 初始姿态 | 盒子躺在桌上：**70% 标签在侧面（需 90° 翻）+ 30% 标签朝下（需 180° 翻）**，偏航随机，(x, y) 有 ±3 cm 抖动 |
| 目标 | 标签法向与世界 +Z 夹角 < 15°（`cos > 0.966`）**且盒子仍在桌上** |
| 回合 | 10 s @ 30 Hz 策略频率（sim dt = 1/120 s，decimation 4） |
| 物理随机化 | 机器人/盒子摩擦、盒子质量、执行器刚度阻尼（startup 一次性），关节位置小噪声 |

### 观测

* **状态任务**（`OpenArm-Flip-State-v0`）：
  * **62 维（默认，新训练用）**：18 关节位置（限位归一化，±0.02 噪声）+ 18 关节速度
    （×0.1）+ 盒子位置 3 + 盒子四元数 4 + 标签法向 3 + 上一步动作 16。
    **不含盒子速度** —— 真机上速度最难测；消融实验（`scripts/vel_ablation.py`）显示
    置零只掉 3.5pp、加 30% 噪声几乎无影响（−0.4pp），而去掉关节速度会掉 14.1pp
    （所以保留）。这样真机感知只需输出"位置+姿态"，单帧 CNN 即可。
  * **68 维（旧）**：上述 62 维 + 盒子线速度 3（×0.2）+ 盒子角速度 3（×0.2）。
    2026-09 及之前训练的 checkpoint 都用这版；`play.py`/`eval_model.py`/
    `eval_breakdown.py`/`train.py` 会按 checkpoint 的输入维度自动选择排布。
* **视觉任务**（`OpenArm-Flip-v0`）：`policy` 组 = 18 关节位置；`image` 组 = 前视相机
  RGB（**归一化、通道在前，shape `(N, 3, 72, 96)`**，96×72 低分辨率以降低渲染 + CNN
  开销）。CNN 同时消费这两组。

### 奖励表

Isaac Lab 奖励管理器按 `weight * dt` 缩放每项（dt = 策略步长 1/30 s）；**一次性奖励**在
cfg 里写 `weight = bonus / dt = 15.0`，`__post_init__` 按 `0.5 / dt` 重算，因此实际每次
事件恰好 +0.5。

| 奖励项 | 类型 | 权重 | 说明 |
|---|---|---|---|
| `label_up` | 密集塑形 | +5.0 | 标签法向 z 分量（越大越接近朝上）；**门控**：仅盒子仍在桌上时给付，防“翻完就扔” |
| `success_bonus` | 稀疏一次性 | +200 | 标签朝上（<15°）**且**盒子在桌上（`label_up_success_reward`） |
| `parcel_touch_bonus` | 一次性 | cfg 15.0（=0.5/dt，实发 +0.5） | 首次**指尖**碰盒（touch 阶段；上臂碰盒不计入、另行惩罚） |
| `parcel_grasp_bonus` | 一次性 | cfg 15.0（=0.5/dt，实发 +0.5） | 首次同侧两指**同时**夹住盒子（grasp 阶段） |
| `arm_body_contact` | 惩罚（按秒） | −2.0/s | 上臂 link1–6（非腕部）碰盒期间持续扣分；腕部 link7 与指尖允许 |
| `action_l2` | 惩罚 | −0.005 | `−0.005·‖a‖²`，抑制抖动/大幅动作 |
| `parcel_dropped` | 惩罚 | −5.0 | 盒子掉下桌（低于桌面 0.08 m）当步给 1.0 |
| `time_penalty` | 惩罚（按秒） | −0.2/s | 每模拟秒 −0.2，随回合时间线性累积，对抗“原地不动”局部最优 |

---

## 三、运行环境与依赖

* Isaac Sim 6.0.1 standalone，默认位于
  `/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64`（可用 `ISAAC_ROOT`
  覆盖）；
* Isaac Lab 各包以 editable 方式装进 kit python（`isaaclab`、`isaaclab_rl`、
  `isaaclab_physx`、`isaaclab_contrib`、`isaaclab_visualizers`，见 `$ISAAC_ROOT/kit/
  python/bin/python3 -m pip list`）；
* torch 2.11（standalone 预置，脚本会提前把它加进 `sys.path`）；
* 官方 Enactic 机器人包 `openarm_isaac_lab`（`OPENARM_USD` 指向其
  `openarm_bimanual.usd`，路径不同需改 `openarm_flip_env_cfg.py`）；
* `openarm_description` 网格包（仅 `ROBOT_SOURCE="urdf"` 回退路径需要，用于解析
  `package://openarm_description/...` 网格路径）。

> **GPU 物理**：默认 `--device cuda:0` 走 GPU。注意 PhysX “gpu override” 必须设为
> **Force GPU**（`overwrite_gpu_setting(1)`，0 = Force CPU!）并配合
> `/physics/suppressReadback=True`，`omni.physics.tensors` 才能在 GPU 上跑——修复在本地
> `isaaclab_physx` 的 `physx_manager.py`。CPU 也能用：`--device cpu` + 少一些 env。

## 四、相机

场景相机默认开启（见 `openarm_flip_env_cfg.py` 的 `_setup_cameras`）：

* **前视相机**：固定在双肩中点前方（双肩在 (0, ±0.031, 0.698)，相机在 (0.15, 0, 0.70)，
  斜向下看桌上的盒子）；
* **腕部相机**：每侧末端工具 link 下方 6 cm（`wrist_left_cam` / `wrist_right_cam`，
  眼在手），RGB + 深度，320×240（近距离采集可调大 `CAM_WIDTH/HEIGHT`）。

**腕部相机位姿同步（重要）**：腕部相机挂在 `link7` 下，但 Isaac Lab 在 fabric 模式下
**不把 link 的变换回写 USD**，因此相机 prim 生成后位姿即被冻结（实测：机械臂移动
4.4 cm、相机位移 0.00 cm），图像全黑（相机停在错误位置并朝向自身几何）。现在由
`EventCfg.sync_wrist_cameras`（interval 事件，每个策略步一次）按 link 的**物理**位姿
主动设置相机世界位姿（实现见 `mdp/events.py`，思路同 openarm_sorting 的
`WristCamera.update_pose`）。

修复前后实测（`scripts/camera_check.py --save /tmp/cam`）：

| | 修复前 | 修复后 |
|---|---|---|
| wrist RGB 非零像素 | **0%**（全黑，文件 302 B） | **92~95%**（文件 18 KB） |
| wrist depth 均值 | 0.075 m（贴脸/埋在几何里） | **0.5~1.5 m**（正常看到桌面场景） |
| 是否跟随机械臂 | 否（位姿冻结） | 是（位姿随 link7 更新） |
| 安装方向 | 偏移 `-Z` + 恒等朝向 → 朝手腕后方 | 偏移 `+Z` 6 cm + 朝 `+Z` → 朝指尖/盒子 |

> 注意 `Camera.data.pos_w` **不会**反映 `set_world_poses` 的结果（实测读数不刷新），
> 判断相机是否跟随要**看图像/depth**，不要看 `pos_w`。

**视觉任务必须开相机**：前视相机每个策略步以 96×72 渲染并喂给 CNN 图像观测，
`--disable_cameras` 对它无效（脚本会警告并忽略）。只有纯状态任务
（`OpenArm-Flip-State-v0`）能无相机运行（`--disable_cameras` 或
`smoke_test.py --no_cameras`）。相机在无头模式离屏渲染；用 `scripts/camera_check.py`
验证。

## 五、快速开始

所有命令在项目根目录 `/home/blanc/isaacsim/openarm_lab` 下用 `python.sh` 执行
（完整命令清单与参数表见 **[TRAINING.md](TRAINING.md)**）：

```bash
ISAAC_ROOT=/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64

# 1. 端到端冒烟验证（需相机）
$ISAAC_ROOT/python.sh scripts/smoke_test.py --num_envs 8 --device cuda:0 --headless

# 2. 训练视觉策略（CNN：关节 + 前视相机图像）
$ISAAC_ROOT/python.sh scripts/train.py --task OpenArm-Flip-v0 \
    --num_envs 512 --device cuda:0 --headless --max_iterations 5000

# 2b. 快速纯状态基线（无相机、MLP，约快一个量级）
$ISAAC_ROOT/python.sh scripts/train.py --task OpenArm-Flip-State-v0 \
    --num_envs 1024 --device cuda:0 --headless --disable_cameras --max_iterations 5000

# 3. GUI 播放已训策略（默认 OpenArm-Flip-Play-v0；可加 --cam top/side/follow）
$ISAAC_ROOT/python.sh scripts/play.py \
    --checkpoint logs/rsl_rl/openarm_flip_img/<run>/model_5000.pt --num_envs 4

# 4. 从 checkpoint 续训
$ISAAC_ROOT/python.sh scripts/train.py \
    --checkpoint logs/rsl_rl/openarm_flip_img/<run>/model_3000.pt --max_iterations 10000

# 5. 无头成功率评估
$ISAAC_ROOT/python.sh scripts/eval_model.py \
    --task OpenArm-Flip-v0 \
    --checkpoint logs/rsl_rl/openarm_flip_img/<run>/model_19999.pt \
    --num_envs 32 --episodes 10

# 6. 手动遥操作机械臂（键盘；不需要 VR 设备）
$ISAAC_ROOT/python.sh scripts/teleop_openarm.py --teleop_device keyboard

# 7. 遥操作链路无头自检（建环境 + 注入位移，验证 IK 真能驱动机械臂）
$ISAAC_ROOT/python.sh scripts/teleop_openarm.py --headless --smoke_steps 90
```

* 训练日志落在 `logs/rsl_rl/openarm_flip_img/<时间戳>/`（视觉任务）或
  `logs/rsl_rl/openarm_flip/<时间戳>/`（状态任务）；用
  `tensorboard --logdir logs/rsl_rl` 查看曲线。
* **去 `--headless` 即开 GUI**：Isaac Lab 3.0 的 AppLauncher 只有显式传
  `visualizer="kit"` 才会弹窗（脚本已处理），请在有桌面的会话里运行（本机 GNOME，
  远程需 `ssh -X`/VNC）。
* **切换播放配置**：`play.py` 默认 `--task OpenArm-Flip-Play-v0`，对应
  `OpenArmFlipEnvCfg_PLAY`（16 envs、回合 15 s、`side_prob=1.0` 只出 90° 侧面、
  `pos_jitter=0.02` 随机化更温和，适合稳定演示）。

### 手动遥操作（`OpenArm-Teleop-v0`）

用键盘 / SpaceMouse / 游戏手柄直接操作机械臂，**不训练策略**：

```bash
$ISAAC_ROOT/python.sh scripts/teleop_openarm.py --teleop_device keyboard
# 或 --teleop_device spacemouse（需 hid 包：pip install hidapi）/ --teleop_device gamepad
```

| 项目 | 说明 |
|---|---|
| 任务 | `OpenArm-Teleop-v0`（`openarm_teleop_env_cfg.py`） |
| 动作 | **8 维** = 7 维**绝对**目标位姿（`pos(3)+quat(4)`，DLS IK）+ 1 维二元夹爪。设备输出的是 6 维增量，由脚本 `apply_delta_pose` 累加到**固定 target** 上（见下方"为什么用绝对位姿"） |
| 被操作的臂 | **右臂**（7 关节）+ 右夹爪；左臂保持在 home。改 `TELEOP_ARM_SIDE` 可切左臂 |
| 环境 | 1 个 env、无相机、无接触传感器、无奖励项；回合极长（用 'R' 手动重置） |
| 键盘 | `W/S`=前后 `A/D`=左右 `Q/E`=上下 · `Z/X`=roll `T/G`=pitch `C/V`=yaw · `K`=夹爪开合 · `R`=重置环境 |
| 自检 | `--headless --smoke_steps 300` → 建环境、松手保持、注入位移；打印漂移量与 EE 位移并判定 `IK WORKS` / `松手不漂移` |

**为什么用绝对位姿而不是相对增量（避坑记录）**：相对模式下 IK 的 `set_command` 把增量加到
**当前实际 EE 位姿**上，而 PD 控制存在重力静差（关节实际角总比目标低一点），于是"目标跟着
实际位姿一起下垂"被每步固化 —— 表现为**松手后机械臂缓缓持续下坠**（实测零指令 300 步
EE 从 0.535 m 掉到 0.309 m，降幅递减但不停止）。改成绝对模式后，target 由脚本维护、
松手不变，IK 每步以该固定目标纠偏，重力下垂被闭环消除（实测 150 步漂移仅 **2.8 mm**）。
配套注意：绝对模式下 `arm_action.scale` **必须为 1.0**（`process_actions` 会把 scale
无条件乘到命令上，乘到四元数上会把它乘坏）。

**为什么先做这一步（而不是直接上 VR）**：遥操作的难点是"输入设备 → SE3 → IK → 关节目标"
这条链路和 OpenArm 的 IK 标定，与输入源是不是 VR 无关。先用键盘/SpaceMouse 打通并调好
灵敏度/IK 参数，之后接头显只是替换输入源。

### XR 遥操作（PICO 4 Ultra / Meta Quest，经 CloudXR.js）

**已就绪**：XR 重定向管线（`openarm_teleop_env_cfg._build_openarm_teleop_pipeline`）把
头显手柄映射成与本项目动作项完全一致的 **8 维绝对位姿 + 夹爪** 动作，脚本内置 `--xr` 分支。

**依赖（已装，装法很关键）**：
```bash
# isaacteleop 必须 --no-deps 装到隔离目录：它的依赖声明会让 pip 升级
# torch/numpy/CUDA 包，破坏 Isaac Sim 自带环境（Isaac Sim 自带的 numpy 2.3.1 /
# scipy 1.17.0 / torch 2.11.0 已满足它全部要求）。
$ISAAC_ROOT/kit/python/bin/python3 -m pip install --no-deps \
    --target /home/blanc/isaacsim/openarm_lab/.xr_deps 'isaacteleop==1.4.142'
# 官方 teleop 脚本会 import SpaceMouse，需要 hid：
$ISAAC_ROOT/kit/python/bin/python3 -m pip install hidapi
```
`isaaclab_teleop`（Isaac Lab 仓库内的包）与 `.xr_deps` 由 `xr_register.py` /
脚本的 `--xr` 分支自动注入 `sys.path`，**不需要**也**不要**动 kit python 的其它包。

**运行（PC 端）**：
```bash
$ISAAC_ROOT/python.sh scripts/teleop_openarm.py --xr            # 首次会交互式询问接受 CloudXR EULA
$ISAAC_ROOT/python.sh scripts/teleop_openarm.py --xr --accept_cloudxr_eula   # 或用此参数表示已接受
```
启动后终端会打印头显端需要的一切：CloudXR profile、本机局域网 IP、客户端网址。
（本机 IP 是 `192.168.1.8`；用 UDP 探测获取，避免 `gethostbyname` 误报 127.0.1.1。）

**两个已踩过的坑（脚本已自动处理，记录以备升级后复查）**：

1. **官方 launcher 的就绪检测必然超时**。isaacteleop 的 `CloudXRLauncher` 轮询
   `<run>/runtime_started` 哨兵文件，但本机 CloudXR SDK 6.1.0（apiVersion 1.4.329）
   启动后**不写它**（实测盯 75 s，run 目录里只出现 `cloudxr.env`/`cloudxr.pid`/`eula_accepted`），
   于是必然报 `CloudXR runtime failed to start within 30s. Process exited with code 1.`
   —— 而原生日志其实显示 runtime 已 "accepting connections"。
   *（曾尝试自己补写该哨兵：它会阻止 runtime 启动 —— 因为该文件同时是 runtime 的 lock file，
   预先存在会被判定为"已有实例"。已放弃此做法。）*
   → 现由脚本自行拉起 runtime 进程（`_start_cloudxr_runtime`），并设
   `ISAACLAB_CXR_SKIP_AUTOLAUNCH=1` 让 session_lifecycle 不再自启；就绪判据改为扫描
   原生日志（`cxr_server.*` / `cxr_streamsdk.*`，两者的 flush 时机不同，都要看）。
   传 `--auto_cloudxr` 可切回官方自动路径。

2. **上一次运行残留的 runtime 会占住 IPC**，导致新 runtime 报
   `ERROR_IPC_SERVER_FAILED_TO_START` 并干净退出（exit 0），表现为"runtime 提前退出"。
   残留通常来自上次被 Ctrl+C / timeout 强杀（atexit 未执行）。
   → 脚本启动前会自动清理（`_kill_stale_runtime`，只杀 cmdline 含
   `isaacteleop`/`cloudxr` 的进程，避免误伤）。清理放在**创建环境之前**：残留 runtime
   会占 GPU 显存，晚清理会让 PhysX 初始化失败（`Failed to get DOF velocities from backend`）。

3. **头显访问 `https://<ip>:48322/` 报"无法连接"**。48322 是 CloudXR 的 WSS/TLS 代理，
   原由 `CloudXRLauncher` 在 runtime 就绪后启动 —— 我们绕过了 launcher，于是代理也不会启动。
   → 脚本在 runtime 就绪后自行启动该代理（`_start_wss_proxy`，复用
   `isaacteleop.cloudxr.wss.run`，把 48322 的 WebRTC/WS 流量转发到 runtime 的 49100），
   并等待端口可连后打印状态。
   注意它要求 **websockets ≥ 14**（Isaac 自带 12 只有旧 API，且 `wss.py` 无回退会
   `sys.exit`）→ 已把 `websockets==14.2` 一并装进 `.xr_deps`，由
   `_ensure_websockets_v14()` 前置该目录并丢弃旧模块缓存。
   实测 `curl -k https://192.168.1.8:48322/` 返回 200 与 "Certificate Accepted" 页面。

**退出方式**：用 Ctrl+C 让 atexit 正常停掉 runtime 与 WSS 代理。若用 `timeout` 等外部
超时工具强杀，注意 `python.sh` 不会 `exec`，**只杀得掉 bash、Python 子进程会存活**并占住
GPU（下次运行会因显存不足失败）。清理残留：

```bash
pkill -9 -f "[t]eleop_openarm"; pkill -9 -f "[i]saacteleop"
```

**运行（PICO 4 Ultra 端）**：
1. 系统需 **PICO OS 15.4.4U+**（CloudXR.js 要求 HTTPS 模式）
2. 头显浏览器打开 `https://nvidia.github.io/IsaacTeleop/client/release-1.4.x`
3. **Server IP** 填 PC 的局域网 IP（如 `192.168.1.8`）→ **Connect**
4. 首次需接受自签名证书：点页面上的 `https://<ip>:48322/` 链接 → **Advanced** → **Proceed**，
   回到客户端再点 Connect
5. **VR 视角（站位/朝向）**：默认 `behind` —— 虚拟视角在机器人**后方 0.9 m、人眼高度、面向 +X**，
   机器人与其前方桌面都在视野内。可切换：
   ```bash
   --xr_view behind|side_left|side_right|behind_left|overhead   # 预设
   --xr_anchor_pos X Y Z --xr_anchor_rot QX QY QZ QW           # 完全自定义
   ```
   要点：`anchor_pos` 是"**映射到头显空间原点的那一点**"，你戴头显时的站位/身高（约 1.6 m）
   会**叠加**到它上面 —— 所以 z 通常填 `0`（仿真地面）。早期把 z 写成 `0.20` 会让视角升到
   1.8 m，表现为"从机器人正上方俯视"；`anchor_rot` 默认 `(0,0,0,1)` 面向 +X，
   侧面站位需绕 Z 轴 ±90°（预设里已算好四元数）。

6. **操作方式（离合 / clutch 模式）**：
   - **按住右手柄侧键（squeeze / grip，握把侧面那个键）** → 机械臂按**手柄的相对位移**跟随运动；
   - **松开侧键** → 机械臂立即冻结，此时可把手柄挪回舒适位置，再按住继续（像鼠标"抬起—挪回—再拖"），
     不必让手柄的绝对位置去对应机械臂的工作空间；
   - **扳机（trigger）** 控制夹爪开合（阈值 0.5），与侧键互不干扰；
   - 手感可调：`--xr_pos_scale`（位移比例，默认 1.0；调小更精细）、`--xr_rot_scale`（旋转比例）、
     `--xr_squeeze_threshold`（触发阈值）；
   - **触发键可切换**：`--xr_clutch_button squeeze|trigger|primary|secondary|thumbstick_click|menu`
     （默认 `squeeze` = 侧键，PICO 4 Ultra 实测可用）。不同头显的按键上报字段可能不同，
     若某个键读不到值，看终端那行 `按钮: squeeze=… trigger=… primary=…` 哪个随按键变化，
     就把它设为触发键；
   - 头显内 UI 可暂停/恢复、重置环境；终端每 5 秒打印
     `[XR] active=… clutching=… pos=… squeeze=…` 便于诊断。

   **为什么某个腕关节（joint6）幅度特别大 / 常贴限位**：joint6 的行程只有 ±0.785 rad，
   仅其他关节（±1.571）的一半，所以同样的姿态调整下它最先到限位；它又是腕部 roll 且靠近
   末端（`body_offset=0.11 m` 让腕部旋转还会带动控制点），IK 的最小范数解会较多地使用它。
   最常见的诱因是**目标被推到可达范围之外**（早期绝对位姿模式下，手柄的绝对位置 z≈1.4 m
   直接成了 EE 目标，全关节一起饱和）。现在的应对：
   - clutch 相对位移本身已避免"绝对位置当目标"；
   - 脚本默认把 EE 目标**限幅**在工作空间安全盒 `(0.02,-0.45,0.20)~(0.68,0.45,0.80)` 内
     （`--no-xr_clamp_target` 可关），超出即削平并打印一次提示；
   - 想更精细就把 `--xr_pos_scale` 调小（如 0.5），手柄同样动作下目标变化减半。

   实现要点（备查）：按钮值来自 pipeline 里自定义的 retargeter，有两个坑：
   ① 控制器输入是 Optional，而 `TensorReorderer` 的 scalar 输入是必需的，直接接线会报
   `Cannot connect optional output … to required input`，所以要包一层转成必需输出
   （手柄不可用时输出 0 = 不跟随，保证安全）；
   ② 该输出必须是**单个 NDArrayType 数组**（`out[0] = np.array([...])`），用若干 `FloatType`
   标量成员逐个赋值读不出值（实测按键全为 0）。
   因此 teleop 动作张量为 **14 维** `[pos(3), quat(4), gripper, squeeze, trigger, primary,
   secondary, thumbstick_click, menu]`：脚本取前 8 维喂给环境，其余用于离合判断与按键诊断。

**建议的系统准备（需 sudo，可选但影响手感）**：
```bash
sudo apt-get install -y libvulkan1 libbsd0        # 已安装 ✓
sudo cpupower frequency-set -g performance        # 当前是 powersave，会显著拖慢 IK（单线程瓶颈）
sudo ufw allow 49100/tcp && sudo ufw allow 47998/udp && sudo ufw allow 48322/tcp   # 若启用了 ufw
```

> **硬件提示**：官方推荐 24 GB 显存 + 64 GB 内存；本机 **RTX 4070 12 GB + 31 GB** 低于推荐
> （Isaac Lab 启动时的工作站检查会报 4 项 WARN，但不阻塞）。遥操作已关掉 scene 相机与图像观测
> 以尽量省显存；若帧率不理想可降低桌面分辨率/关闭其它占显存进程。
> 参考文档：Isaac Lab 仓库 `docs/source/how-to/cloudxr_teleoperation.rst`。

## 六、视觉任务（CNN 策略）训练

视觉任务 `OpenArm-Flip-v0` 的策略是 CNN（输入 = 18 维关节位置 + 前视相机 rgb
`(N,3,72,96)`），训练方式与状态任务相同，只是**必须开相机**：

```bash
# 视觉任务训练（--lean_cameras 强烈建议：实测提速 2.4 倍）
$ISAAC_ROOT/python.sh scripts/train.py --task OpenArm-Flip-v0 \
    --num_envs 256 --device cuda:0 --headless --lean_cameras --max_iterations 5000
```

### 为什么要 `--lean_cameras`

默认配置会为每个环境创建 **3 个相机**（前视 + 左右腕），且每个都渲染 **rgb + depth**，
但 CNN 策略只用**前视相机的 rgb** —— 其余约 98% 的渲染像素纯属浪费。开启该开关后
只保留前视相机与 rgb。

实测吞吐（`scripts/bench_vision.py`，RTX 4070 12 GB）：

| num_envs | 默认配置 | `--lean_cameras` | 每 PPO 迭代（lean） | 20000 迭代推算 |
|---|---|---|---|---|
| 64 | 204 env-steps/s | **495**（2.4×） | 3.10 s | 17.2 h |
| 256 | — | **1696** | 3.62 s | 20.1 h |
| 512 | — | **2790**（无 OOM） | 4.40 s | **24.5 h** |

对比：状态任务（无相机）同样 20000 次迭代约 14 小时 —— 即**视觉训练只慢约 1.7 倍**，
完全可承受。建议 `--num_envs 256`（性价比好）或 512（每样本最省）。

> 注意：`lean_cameras` 必须在**构造 env cfg 时**传入（相机在 `__post_init__` 里创建），
> `train.py --lean_cameras` 已处理好；手动改 `env_cfg.lean_cameras = True` 是无效的。

CNN 侧超参与状态任务一致（有界 std `(0.2, 2.0)`、`entropy_coef=0.005`、
`learning_rate=1e-3` + adaptive，见 `agents/rsl_rl_ppo_cfg.py` 的
`OpenArmFlipImgPPORunnerCfg`）。CNN 结构为 3 层（通道 [8,16,32]、stride 2）。

## 七、感知精度需求（sim-to-real 参考）

策略对"盒子位姿"感知误差的容忍度实测（`scripts/perception_sensitivity.py`，
62 维最优模型，每模式 512 回合，95% CI 约 ±2.8pp）：

| 扰动类型 | 幅度 | 成功率 | 相对基线 |
|---|---|---|---|
| 基线（真值） | — | 88% | — |
| 位置随机噪声 | 1 / 2 / 4 / **8 cm** | 88.7 / 88.7 / 86.9 / 85.9% | +1.0 ~ **−1.8pp** |
| 姿态随机噪声 | 10° / 30° / **60°** | 87.3 / 87.9 / **77.5%** | −0.4 / +0.2 / **−10.2pp** |
| 位置**常偏**（标定误差） | 1 / 2 / **4 cm** | 86.5 / 85.7 / **83.6%** | −1.6 / −2.3 / **−4.5pp** |
| 姿态**常偏** | 10 / 20° / **40°** | 86.1 / 86.9 / **79.9%** | −2.0 / −1.2 / **−8.2pp** |
| 延迟 | 2 步(67ms) / 6 步(200ms) | 88.7 / 83.8% | +1.0 / −3.9pp |

**结论**：容差相当宽 ——

* **位置**：随机误差到 8 cm 仍几乎无损；**常偏误差**敏感些，≤2 cm 基本无损、4 cm 掉 4.5pp
* **姿态**：≤30° 基本无损，60° 才明显退化
* **延迟**：≤200 ms 影响可控（−3.9pp）

原因是任务本身是**闭环推箱子**：策略每 33 ms 重新观测并修正，感知偏差会在后续步
被自然纠正；而且它只需"往标签方向推"，不需要精确抓取。若换成"精确抓取"类任务，
容差会紧得多。

> 因此若用 CNN 做感知前端（输出盒子位置+姿态给 62 维策略），典型单目位姿估计
> 误差（数毫米 / 数度）远在容差内 —— **方案可行**。但需另加处理的是**遮挡**
> （会造成"大错"而非"小噪声"）与相机内外参标定。

## 八、场景外观：纹理与光照

### 纹理（自包含，程序化生成）

```bash
python3 tools/make_textures.py            # 重新生成全部纹理（--seed 可复现）
```

生成到 `assets/textures/`（随仓库分发，**不依赖任何外部下载**，便于迁移到其他设备）：

| 文件 | 用途 | 说明 |
|---|---|---|
| `express_label.png` | 标签面单 | 原有素材，之前**一直没被用上**，现已真正绑定 |
| `table.png` | 桌面 | **哑光层压板**（浅灰白 + 细密颗粒 + 轻微划痕，无木纹；材质 roughness=0.9） |
| `floor.png` | 地面 | 中性灰 + 低频斑驳（替换 Isaac 默认的蓝青色网格地面） |

> **快递盒用纯色**（`parcel.usda` 的 `primvars:displayColor = (0.66, 0.45, 0.24)`），
> 不绑纹理 —— 纸箱纹理试过几轮（含单向瓦楞条纹、各向同性带通噪声），观感反而更假
> 且容易做出木纹感，最终回到纯色。想换回纹理时：
> `python3 tools/make_textures.py` 会重新生成 `cardboard.png`，再把 `CardboardMat`
> 那段加回 `EventCfg.apply_textures`（cfg 里已留好注释）。

**绑定方式**：`EventCfg.apply_textures`（startup 事件 → `mdp/events.py: apply_scene_textures`）
在 `/World/Looks` 下创建**全局共享**的 PBR 材质（`UsdPreviewSurface` + `UsdUVTexture`），
再按 `{ENV_REGEX_NS}` 正则绑定到各几何体。共享材质是刻意的——若把材质写进
`parcel.usda`，1024 个环境会各自复制一份材质层级，白白增加开销。

### ⚠️ 两个必须知道的坑

1. **Cube 必须自己 author `st` primvar，否则纹理采样失败**。
   早先文档里写的"RTX 不认 Cube 上的材质绑定"其实**不准确**：真正原因是原生
   `UsdGeom.Cube` 没有 UV，`UsdUVTexture` 取不到坐标，表面就渲染成灰/白。
   补上 `primvars:st`（6 面 × 4 角 = 24 个值，`interpolation = faceVarying`）后
   Cube 也能正常贴图——实测盒面渲染色与纹理均值吻合。UV 现已写进
   `assets/parcel/parcel.usda`。
2. **`GroundPlane` 要绑到真正的可见几何，并注意它自带光源**。
   该资产的可见地面是 `/World/GroundPlane/Environment/Geometry`（自带蓝青色网格
   材质），直接绑到 `/World/GroundPlane` 只会作用到外层 Xform 与碰撞平面，地面
   仍是默认网格；此外资产里还藏着一盏 `SphereLight`（额外照亮 → 整体过曝）。
   现在事件会绑定到具体几何，并自动关闭该资产的灯。

### 光照（三层，替代原来的单盏穹顶光）

| 灯 | 类型 | 强度 | 作用 |
|---|---|---|---|
| `dome_light` | DomeLight | 220 | 微暖环境填充光 |
| `key_light` | DistantLight | 1100 | 主光（斜向照下，产生方向性阴影/高光） |
| `fill_light` | DistantLight | 320 | 另一侧补光，避免阴影死黑 |

强度是按实测标定的：三盏灯合计过强会把前视相机画面均值推到 ~238（接近过曝、
丢细节），当前配置下约为 217 且 0% 过曝。

### 效果与代价（实测）

| | 改之前 | 改之后 |
|---|---|---|
| 盒面渲染色 | (196, 175, 138) 纯色 | **(207, 169, 99)** 瓦楞纸色调，std 明显（有纹理） |
| 腕部相机色偏 R−B | **−46.8**（明显偏蓝，来自默认网格地面） | **−1.9**（基本中性） |
| 前视相机过曝像素 | — | 0% |
| 视觉训练吞吐（256 envs, lean） | 1696 env-steps/s | **1806 env-steps/s**（几乎无代价） |

> 纹理与光照只影响渲染，**不改变任何物理与观测数值**——状态策略的结果完全不受影响
> （62 维观测里没有图像）。它们的作用是让视觉任务 / 感知网络 / 演示更接近现实。

## 九、设计要点与调参笔记

* **可到达性**：盒子放在桌上 (x ≈ 0.30 m, y ≈ 0.10 m)，贴近机器人中轴线，双臂都能
  够到。翻面本质是推/夹行为：平行夹爪张开约 0.10 m 可包住 0.09 m 的盒子，但用手指
  推扫也是合法策略——密集 `label_up` 塑形 + 稀疏奖励共同驱动该行为。
* **随机化 / 课程**：盒子标签面（侧面/朝下）、偏航、位置抖动、质量、摩擦、执行器增益
  与关节噪声都在 `EventCfg` 里随机；想上课程可以调高 `side_prob` 或加入“标签朝上起手”。
* **机器人手指约定**：官方 USD **0 = 闭合、0.044 = 张开**（与旧 URDF 相反），home 从
  张开 0.044 开始；夹爪目标经 `use_default_offset` 自动兼容两套约定。
* **视觉输入**：CNN 吃前视相机 RGB（96×72）+ 18 关节位置。训练慢或图像异常时先跑
  `scripts/camera_check.py` 并检查 `front_cam` 安装；图像观测组 shape 应为
  `(N, 3, 72, 96)`（可用 `scripts/smoke_test.py` 确认非平凡）。
* **奖励塑形**：策略卡住时——调低 `action_l2` 权重、加大成功奖励、或调高 `side_prob`
  （90° 比 180° 好学会）。
* **训练稳定性（std 爆炸修复）**：2026-09-06 的 20000 轮训练后期不收敛，根因是
  rsl-rl 的 `Policy/mean_std` 从 1.0 单调爆炸到 ~20（默认 std 无上限 + `init_std=1.0`
  过大）——策略方差失控后采样动作远超动作范围，机械臂纯随机乱甩，critic 损失涨到
  数千。现在的做法（`agents/rsl_rl_ppo_cfg.py`）：自定义
  `OpenArmFlipGaussianDistribution`，把 std 经 sigmoid 恒定钳制在 `(0.2, 2.0)`
  （`init_std=1.0`）；**其余超参沿用 2026-09-06 的设置**（`entropy_coef=0.005`、
  `learning_rate=1e-3` + adaptive）。
  ⚠️ **前车之鉴**：2026-09-10 的训练曾把熵系数降到 0.001、学习率降到 3e-4(fixed)，
  结果**探索被压死**——`mean_std` 从 0.3 坍缩到 0.15、20000 轮里成功奖励一次都没
  触发、策略退化成"完全不动"（回报恒 ≈ −17）。本任务不能削减探索强度。
  **不要设 `clip_actions`**（曾试 1.0，会把旧 checkpoint 的大幅动作压死，导致
  播放时机械臂几乎不动）；**换配置后旧 checkpoint 不能直接续训/播放。**

## 十、已知问题与注意事项

* **显存 OOM（多 env / 相机开销）**：视觉任务每个 env 有 3 个 RTX 相机视图（其中前视
  每步渲染进 CNN），`--num_envs` 开太大容易爆显存，降到 64~256；物理缓冲容量已按
  1024+ envs 显式加大（`PhysxCfg`），但相机是主要开销。多个训练实例并行也会叠加 OOM。
* **checkpoint 与最新环境不匹配**：机器人从 URDF 版换成官方 USD 后，模型结构/手指方向
  均变化，旧 `logs/rsl_rl/openarm_flip*/` 里的 `model_*.pt` **不能**直接续训或播放，
  需要从头重训（`ROBOT_SOURCE` 切回 `"urdf"` 同理）。
* **GPU 物理不生效**：确认 `isaaclab_physx/physx_manager.py` 里
  `overwrite_gpu_setting(1)`（0 = Force CPU!），否则会退回 CPU 物理（慢且结果不同）。
* **GUI 弹不出窗口**：必须在有桌面的会话运行（AppLauncher 需 `visualizer="kit"`，
  脚本已处理），不要用 SSH/无图形终端。
* **相机相关报错**：视觉任务别加 `--disable_cameras` / `--no_cameras`（会被忽略/警告）。
* **NaN 崩溃**：环境内置关节超速与 NaN 状态终止保护，正常不会出现；万一出现，`train.py`
  会打印 NaN 诊断（哪个关节/盒子、量程范围）帮助定位。
