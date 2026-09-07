# OpenArm Lab — 训练命令速查

所有命令都在框架根目录 `/home/blanc/isaacsim/openarm_lab` 下，用 Isaac Sim 自带的 `python.sh` 执行。

```bash
cd /home/blanc/isaacsim/openarm_lab
ISAAC_ROOT=/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64
```

---

#> **两个任务**：
>
> * **`OpenArm-Flip-v0`（默认，视觉任务）**：策略输入 = **18 个归一化关节位置 + 前视相机 RGB 图像**
>   （CNN 策略，图像 96×72）。**必须开相机**，`--disable_cameras` 对它无效。
> * **`OpenArm-Flip-State-v0`（纯状态任务，跑得快 ~30 倍）**：68 维状态（关节+盒子+标签朝向），
>   MLP 策略，无相机，可用 `--disable_cameras`。
>
> 场景里仍带 3 个相机（前视 1 + 左右腕部 2，眼在手）供观察/调试：
> `scripts/camera_check.py` 检查相机，去掉 `--headless` 用 GUI 播放（`scripts/play.py`）。

# 1. 从头开始训练（视觉任务：关节位置 + 图像，CNN）

```bash
$ISAAC_ROOT/python.sh scripts/train.py --task OpenArm-Flip-v0 \
    --num_envs 256 --device cuda:0 --headless
```

- `--task` 默认就是 `OpenArm-Flip-v0`，可不写
- 相机开销大：显存不够时 `--num_envs` 降到 64/128（相机 + CNN 比纯状态任务慢）
- 日志目录：`logs/rsl_rl/openarm_flip_img/<时间戳>/`

## 1b. 从头训练纯状态基线（快，MLP）

```bash
$ISAAC_ROOT/python.sh scripts/train.py --task OpenArm-Flip-State-v0 \
    --num_envs 1024 --device cuda:0 --headless --disable_cameras
```

- 日志目录：`logs/rsl_rl/openarm_flip/<时间戳>/`

## 2. 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--task` | `OpenArm-Flip-v0` | `OpenArm-Flip-v0`（视觉/CNN）、`OpenArm-Flip-State-v0`（状态/MLP）、播放用 `OpenArm-Flip-Play-v0` |
| `--num_envs` | 1024 | 并行环境数；视觉任务显存紧张降到 256/128/64 |
| `--device` | `cuda:0` | `cpu` 或 `cuda:N` |
| `--max_iterations` | 20000 | 总训练轮数 |
| `--seed` | 随机 | 随机种子，复现用 |
| `--checkpoint` | 无 | 指定 `model_*.pt` 则从该点继续训练 |
| `--headless` | 关 | 开启则无 GUI 训练 |
| `--disable_cameras` | 关 | 仅对 `OpenArm-Flip-State-v0` 有效（视觉任务自动忽略） |

# 3. 训练/播放时开着 GUI 看效果

```bash
$ISAAC_ROOT/python.sh scripts/train.py --task OpenArm-Flip-v0 --num_envs 64 --device cuda:0
```

（去掉 `--headless`，能看到机械臂推盒子的实时画面；GUI 模式相机数少开点）

> **GUI 弹窗说明（Isaac Lab 3.0）**：AppLauncher 只有在显式请求 `kit` 可视化器时
> 才会打开图形窗口（脚本内已处理：非 `--headless` 时自动加 `visualizer="kit"`），
> 因此直接去掉 `--headless` 运行即可弹出 "Isaac Lab 3.0.0" 窗口。
> 若窗口没出现，请确认是在有桌面的会话（本机 GNOME `:1`）里运行，不要用
> SSH/无图形终端；若在远程，需 `ssh -X` 或用 VNC。

## 4. 从已有 checkpoint 继续训练

视觉任务：

```bash
$ISAAC_ROOT/python.sh scripts/train.py \
    --checkpoint logs/rsl_rl/openarm_flip_img/<时间戳目录>/model_800.pt \
    --num_envs 256 --device cuda:0 --headless --max_iterations 20000
```

纯状态基线把路径换成 `logs/rsl_rl/openarm_flip/<时间戳目录>/model_800.pt`。

## 5. 查看训练曲线

```bash
tensorboard --logdir /home/blanc/isaacsim/openarm_lab/logs/rsl_rl
```

浏览器打开 http://localhost:6006
重点关注：`Episode_Reward/success_bonus`（>0 说明学会翻箱）、`Train/mean_episode_length`、`Train/mean_reward`

## 6. 播放训练好的模型（看实际动作）

视觉任务（默认用 `OpenArm-Flip-Play-v0`，16 envs、更稳的随机化）：

```bash
$ISAAC_ROOT/python.sh scripts/play.py \
    --checkpoint logs/rsl_rl/openarm_flip_img/<时间戳目录>/model_19999.pt \
    --num_envs 4
```

- 加 `--headless` 则只输出统计不弹窗
- 评估成功率（无 GUI，统计输出）：

```bash
$ISAAC_ROOT/python.sh scripts/eval_model.py \
    --task OpenArm-Flip-v0 \
    --checkpoint logs/rsl_rl/openarm_flip_img/<时间戳目录>/model_19999.pt \
    --num_envs 32 --episodes 10
```

## 7. 训练结果在哪

```
logs/rsl_rl/
├── openarm_flip_img/<时间戳>/     # 视觉任务（CNN）
│   ├── model_0.pt ~ model_19999.pt   # 每 200 轮存一个 checkpoint
│   ├── events.out.tfevents.*          # tensorboard 曲线
│   └── git/                           # 训练时代码版本快照
└── openarm_flip/<时间戳>/          # 纯状态基线（MLP）
```

## 8. 当前任务与奖励设置

- 机器人：OpenArm v1.0 双机械臂（**官方 USD**，`openarm_isaac_lab`），受控 18 自由度
  （双臂 7 旋转 + 双手 2 平移夹爪；USD 里的 hand/ee_tcp 固定关节不参与动作/观测）
- 目标：把快递盒翻到标签朝上（初始 70% 标签在侧面需 90° 翻，30% 标签朝下需 180° 翻）
- 奖励：5×label_up 塑形 + 200 成功 + 0.5 手指触碰（一次性）+ 0.5 夹住（一次性）
  − 2/秒 手臂非手指部位碰盒惩罚（只准用手指/夹爪操作）
  − 0.005·‖a‖² − 5·掉落 − 0.2/秒耗时
- 动作 16 维：14 手臂关节（0.5 rad）+ 2 个耦合夹爪（左右各一，同侧两指联动，0.05 m）
- 视觉任务观测：`policy` 组 = 18 维关节位置（归一化），`image` 组 = 前视相机 RGB (3×72×96，通道在前)；状态任务另含盒子位姿/速度、标签法向、16 维上一步动作

> **换机器人 = 旧 checkpoint 作废**：从 URDF 版换成官方 USD 后，模型结构/手指方向
> 都变了，之前 `logs/rsl_rl/openarm_flip*/` 里的 `model_*.pt` 不能直接续训/播放，
> 需要从头重新训练（`--task OpenArm-Flip-State-v0` 状态版或 `OpenArm-Flip-v0` 视觉版）。

## 9. 故障排查

- **NaN 崩溃**：环境已内置防护（关节超速/NaN 状态自动重置该环境），正常不会出现
- **GPU 显存不足**：视觉任务 `--num_envs` 降到 64/128；或先用状态任务 1b 调试
- **GPU 物理不生效**：确认 `isaaclab_physx` 的 `physx_manager.py` 中为 `overwrite_gpu_setting(1)`（0 = 强制 CPU）
- **视觉任务报相机渲染错误**：不要加 `--disable_cameras`（视觉任务必须开相机）
- **机器人不动/乱动**：检查 `--device` 与日志目录，必要时 `--seed` 固定重训
