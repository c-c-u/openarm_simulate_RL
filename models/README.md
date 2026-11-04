# 已训练模型（标记的最优模型放这里）

本目录存放**经过确认的最优策略 checkpoint**，作为项目的"正式发布版本"。
`logs/` 下的 checkpoint 属于训练过程的原始产物（会被 `.gitignore` 排除且数量庞大），
本目录只保留明确标注过、指标已实测的那几个。

## 当前最优模型

| 文件 | 观测维度 | 成功率（宽松判据） | 来源 | 说明 |
|---|---|---|---|---|
| **`openarm_flip_state62_best.pt`** | **62 维**（不含盒子速度） | **88.8%**（768 回合，95% CI 86.4–90.8%） | 由 `model_20000` 权重手术得到 | ⭐ **推荐部署版本**：sim-to-real 友好，感知只需位置+姿态 |
| （原始 68 维，参考） | 68 维 | 93.6%（512 回合） | `logs/rsl_rl/openarm_flip/2026-09-16_23-16-18/model_20000.pt` | 训练原始产物（logs 被 gitignore，不在本目录） |

### `openarm_flip_state62_best.pt` 实测指标

评测口径：`scripts/eval_breakdown.py --num_envs 128 --episodes 6`（768 回合，与训练配置一致，`side_prob=0.7`）

| 指标 | 数值 |
|---|---|
| 过程中曾达到（宽松） | **88.8%**（682/768，95% CI 86.4–90.8%） |
| 结束时仍保持（严格） | **82.7%**（635/768） |
| 标签在侧面（90° 翻） | **90.3%**（495/548） |
| 标签朝下（180° 翻） | **85.0%**（187/220） |
| 成功后保持到回合结束 | 93.1%（635/682） |
| 首次达成用时 | 平均 **1.28 s**、中位 **0.73 s** |
| 主要失败模式 | 掉下桌 125/768（16.3%）、推出界 0 |

**与 68 维原版的差距约 4.8 个百分点**（93.6% → 88.8%），与消融实验的预测
（把盒子速度置零 → 89.8%，95% CI 86.1–92.8%）在统计上一致，说明**手术是精确的**，
掉点全部来自"少了速度信息"本身，而非模型被破坏。

若要拿回这部分性能，可在 62 维环境下从这个模型续训几千轮（`scripts/train.py --checkpoint`）。

### 补充说明

* **来源**：`2026-09-16_23-16-18/model_20000.pt`（2026-09-16 续训 20000 轮的最终模型，
  68 维观测下实测 93.6%）
* **处理方式**：`scripts/strip_parcel_vel.py` 权重手术——删掉盒子线速度/角速度 6 维，
  并把它们的常数贡献补进第一层 bias：
  `b_new = b_old + W[:, 43:49] @ ((0 - μ) / (σ + eps))`
  （`eps=1e-2` 是 rsl-rl `EmpiricalNormalization` 的默认值，**漏掉它会让补偿算错**）
* **等价性**：脚本内置前向对比验证（随机观测上最大误差 2.1e-05，输出量级 18.5）；
  另用 `scripts/check_obs_layout.py` 验证了 62 维观测的实际拼接顺序与假设一致
  （`label_normal` 模长−1 = 2.4e-07、`parcel_pos` 与真值完全一致、`label_normal[2] == label_up`）
* **为什么去掉盒子速度**：真机上盒子速度最难获得（需多帧差分+滤波，噪声大、有延迟）；
  消融实验（`scripts/vel_ablation.py`）显示置零只掉 3.5pp、加 30% 噪声几乎无影响
  （−0.4pp），属于"锦上添花"而非"必需"。去掉后真机感知只需输出**位置 + 姿态**，
  单帧 CNN 即可。注意**关节速度 18 维不能去**（消融掉 14.1pp），但它恰是真机最易获得的量

## 用法

```bash
cd /home/blanc/isaacsim/openarm_lab
export ISAAC_ROOT=/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64

# GUI 播放（脚本会自动识别这是 62 维模型并匹配合适的观测配置）
$ISAAC_ROOT/python.sh scripts/play.py --checkpoint models/openarm_flip_state62_best.pt --num_envs 4

# 细分评测（成功率、按朝向/位置分组、失败模式、用时分布）
$ISAAC_ROOT/python.sh scripts/eval_breakdown.py --checkpoint models/openarm_flip_state62_best.pt \
    --num_envs 64 --episodes 4

# 从这个模型继续训练（62 维观测；会自动匹配维度）
$ISAAC_ROOT/python.sh scripts/train.py --task OpenArm-Flip-State-v0 \
    --checkpoint models/openarm_flip_state62_best.pt --num_envs 1024 --headless --disable_cameras
```

## 观测维度约定（重要）

状态任务现在有**两套观测排布**，脚本会按 checkpoint 的输入维度自动选择
（见 `openarm_flip/checkpoint_utils.py`）：

| 维度 | 组成 | 适用 |
|---|---|---|
| **62**（默认，新训练用） | joint_pos(18) + joint_vel(18) + parcel_pos(3) + parcel_quat(4) + label_normal(3) + last_action(16) | 新训练、sim-to-real 部署 |
| **68**（旧） | 上述 + parcel_lin_vel(3) + parcel_ang_vel(3) | 2026-09 及之前训练的旧 checkpoint |

* 从零训练 → 默认 62 维
* 播放/评估/续训旧模型 → 自动切换到 68 维（无需手动设置）
* 手动强制：`env_cfg.use_parcel_vel = True/False`（见 `openarm_flip_env_cfg.py`）

## 重新生成这个模型

```bash
# 1) 找训练产出（logs 被 gitignore，需本地存在）
CKPT=logs/rsl_rl/openarm_flip/2026-09-16_23-16-18/model_20000.pt

# 2) 手术成 62 维（自带数值等价性验证）
/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64/kit/python/bin/python3 \
    scripts/strip_parcel_vel.py --checkpoint $CKPT \
    --output models/openarm_flip_state62_best.pt --verify

# 3) 复测
$ISAAC_ROOT/python.sh scripts/eval_breakdown.py \
    --checkpoint models/openarm_flip_state62_best.pt --num_envs 64 --episodes 4
```
