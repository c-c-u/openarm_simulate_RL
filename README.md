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
│   ├── __init__.py                   # gym.register 三个任务（v0 / State-v0 / Play-v0）
│   ├── openarm_flip_env_cfg.py       # ★核心环境配置（见下方逐文件说明）
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
│   ├── play.py                       # GUI/无头播放已训 checkpoint（--cam top/side/follow）
│   ├── smoke_test.py                 # 端到端环境冒烟测试（随机动作，无策略）
│   ├── eval_model.py                 # 无头成功率评估（成功 = 标签朝上且盒子在桌上）
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
* `OpenArm-Flip-State-v0` —— 纯状态基线（68 维状态、MLP、无相机），对应
  `OpenArmFlipPPORunnerCfg`；
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
* **`OpenArmFlipSceneCfg`**：机器人（4 组 ImplicitActuator：左右臂刚度 2000/阻尼 100，
  左右手指刚度 5000/阻尼 200）、桌子（运动学 pedestal，桌面 z = 0.38 m）、盒子
  （`parcel.usda`，质量 0.35 kg，开接触传感器）、地面、穹顶灯，以及**相机与接触传感器
  的占位字段**（由 `EnvCfg.__post_init__` 填充）。
* **`StateObservationsCfg` / `ImageObservationsCfg`**：前者 68 维状态进 `policy` 组；
  后者 `policy` 组 = 18 关节位置、`image` 组 = 前视相机 RGB（归一化、转置为
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

* **状态任务**（`OpenArm-Flip-State-v0`）：`policy` 组拼接 **68 维**状态：
  18 关节位置（限位归一化，±0.02 均匀噪声）+ 18 关节速度（×0.1）+ 盒子位置 3 + 盒子四元数 4
  + 盒子线速度 3（×0.2）+ 盒子角速度 3（×0.2）+ 标签法向 3 + 上一步动作 16。
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

## 六、设计要点与调参笔记

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

## 七、已知问题与注意事项

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
