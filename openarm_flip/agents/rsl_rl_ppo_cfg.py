# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the OpenArm parcel-flip task.

训练稳定性说明（为什么这里自定义了带 std 上限的高斯分布）：

2026-09-06 的 20000 轮训练在后期不收敛，根因是策略方差（rsl-rl 的
``Policy/mean_std``）从 1.0 单调爆炸到 ~20：rsl-rl 默认的 std 范围是
(1e-6, 1e6) ≈ 无上限，而 ``init_std=1.0`` 相对 16 维动作空间（±1 归一化）
过大。std 一旦失控，采样动作远超出动作范围，机械臂变成纯随机乱甩，critic
损失涨到数千、回报曲线后期崩盘。

修复方案（与 Isaac Lab 官方 ``ur10_particle_push`` 任务同款）：

* 自定义 :class:`OpenArmFlipGaussianDistribution`：把 std 映射到
  sigmoid 有界区间 ``std_range``（当前 (0.2, 2.0)，``init_std=1.0``），无论策略
  参数如何更新，实际探索方差都落在有限范围内 —— 既杜绝 std 爆炸，又保留足够
  的探索幅度。
* **熵系数与学习率保持 2026-09-06 那次训练的取值**（``entropy_coef=0.005``、
  ``learning_rate=1e-3`` + adaptive）：2026-09-10 的训练曾把两者降到
  0.001 / 3e-4(fixed)，结果探索被压死 —— ``mean_std`` 从 0.3 坍缩到 0.15、
  20000 轮里 ``success_bonus`` 一次都没触发、策略退化成"完全不动"
  （reward 恒 ≈ -17）。可见本任务的探索强度不能削减。
* **不要设 ``clip_actions``**：策略输出是关节位置目标，经动作层 scale 与
  关节限位消化。曾尝试 ``clip_actions=1.0``，但它会把训练时无裁剪的旧
  checkpoint 的动作（输出常 >1）压到 ±1，导致播放时机械臂几乎不动
  （见文件内 runner cfg 的注释）。

探索幅度参考：2026-09-06 那次能学会，是因为当时 std 无界、一路涨到 ~20，
策略得以输出远超 ±1 的动作（实测 mean 达 ±6~98），靠关节限位做大幅甩臂并
偶然翻动盒子。因此本任务需要**较大**的探索方差：``std_range`` 上界不宜压得
太低（曾用 (0.1, 0.8) 直接导致学不会），下界则保证最低探索地板。
"""
from __future__ import annotations

import torch
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


class OpenArmFlipGaussianDistribution(GaussianDistribution):
    """有界标准差的 state-independent 高斯分布（防训练中 std 爆炸）。

    与 rsl-rl 原版 :class:`GaussianDistribution` 的唯一区别：std 不是裸可学习
    参数（可无限增大），而是经 ``sigmoid`` 压缩到 ``[std_range[0], std_range[1]]``
    内。``init_std`` 必须先归一化到该区间的比例再取 logit，保证初始 std 精确
    落在期望值上；``update`` 时对 sigmoid 输出仍保留梯度，策略可正常学习
    "该用多大探索方差"。
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (0.2, 2.0),
        std_type: str = "scalar",
        **kwargs,
    ) -> None:
        if len(std_range) != 2 or std_range[0] <= 0.0 or std_range[0] >= std_range[1]:
            raise ValueError("std_range must contain positive, increasing lower and upper bounds.")
        if not std_range[0] < init_std < std_range[1]:
            raise ValueError("init_std must lie strictly inside std_range.")
        if std_type != "scalar":
            raise ValueError("OpenArmFlipGaussianDistribution supports only scalar standard deviations.")
        super().__init__(output_dim, init_std=init_std, std_type=std_type, **kwargs)
        self.std_range = (float(std_range[0]), float(std_range[1]))
        # 把 init_std 在区间内的相对位置换算成 sigmoid 前的 logit，保证初始 std 精确
        initial_fraction = (init_std - self.std_range[0]) / (self.std_range[1] - self.std_range[0])
        initial_logit = torch.logit(torch.tensor(initial_fraction, dtype=self.std_param.dtype))
        with torch.no_grad():
            self.std_param.fill_(initial_logit)

    def update(self, mlp_output: torch.Tensor) -> None:
        """按 sigmoid 有界映射更新高斯分布（std 恒在 std_range 内）。"""
        minimum_std, maximum_std = self.std_range
        std = minimum_std + (maximum_std - minimum_std) * torch.sigmoid(self.std_param)
        self._distribution = Normal(mlp_output, std)


@configclass
class OpenArmFlipGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """有界 std 高斯分布的配置（供 MLP 与 CNN 两套 actor 共用）。"""

    class_name: str = (
        "openarm_lab.openarm_flip.agents.rsl_rl_ppo_cfg:OpenArmFlipGaussianDistribution"
    )
    init_std: float = 1.0
    std_range: tuple[float, float] = (0.2, 2.0)


@configclass
class OpenArmFlipPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO 超参（rsl-rl 5.x，MLP actor/critic）——给纯状态任务 OpenArm-Flip-State-v0 用。"""

    # 每环境每迭代 24 步；最多 20000 迭代；每 200 迭代存一次 checkpoint。
    # experiment_name 决定日志目录：logs/rsl_rl/openarm_flip/
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "openarm_flip"

    # 注意：不要设置 clip_actions。策略输出是关节位置目标（经动作层的 scale
    # 与关节限位消化）；早期版本曾尝试 clip_actions=1.0，但那会与训练时
    # 无裁剪的旧 checkpoint 不兼容（旧模型输出常 >1，播放被压到 ±1 后机械臂
    # 几乎不动），且 std 有界化已足以防止方差爆炸。

    # 观测组映射：环境只暴露一个 "policy" 组（68 维状态），actor 与 critic 共用
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    # actor：高斯分布策略网络（有界 std：1.0 起、范围 [0.2, 2.0]），观测归一化 +
    # 三层 MLP [256,128,64]（elu）
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        distribution_cfg=OpenArmFlipGaussianDistributionCfg(),
        obs_normalization=True,
        hidden_dims=[256, 128, 64],
        activation="elu",
    )
    # critic：价值网络（结构同 actor，无分布头）
    critic: RslRlMLPModelCfg = RslRlMLPModelCfg(
        obs_normalization=True,
        hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # PPO 算法超参：clip 0.2、熵系数 0.005、5 个学习 epoch、4 个 mini-batch、
    # 学习率 1e-3 adaptive（目标 KL 0.01）、grad norm 1.0、gamma/lam 0.99/0.95。
    # 熵系数与学习率沿用 2026-09-06 那次训练的设置：2026-09-10 的训练把两者
    # 调低到 0.001 / 3e-4(fixed) 后，探索被压死（mean_std 坍缩到 0.15、20000 轮
    # 从未成功一次），故恢复为原值以保证探索强度。
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class OpenArmFlipImgPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """CNN 版 PPO（actor/critic 都带 CNN）——给视觉任务 OpenArm-Flip-v0 / Play-v0 用。"""

    # 同上；experiment_name 对应日志目录 logs/rsl_rl/openarm_flip_img/
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "openarm_flip_img"

    # 观测组映射："policy" 是 1D 关节位置组，"image" 是 2D 相机图像组，两者都喂给网络
    obs_groups = {"actor": ["policy", "image"], "critic": ["policy", "image"]}

    # actor：3 层 CNN（输出通道 [8,16,32]、3×3 核、stride 2 下采样，elu）压平后接 MLP
    # [256,128]；图像已归一化，故 obs_normalization=False（关节位置组也走同一路径）
    actor: RslRlCNNModelCfg = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[8, 16, 32],
            kernel_size=[3, 3, 3],
            stride=[2, 2, 2],
            padding="zeros",
            activation="elu",
        ),
        distribution_cfg=OpenArmFlipGaussianDistributionCfg(),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
    )
    # critic：价值网络，CNN 结构同 actor
    critic: RslRlCNNModelCfg = RslRlCNNModelCfg(
        cnn_cfg=RslRlCNNModelCfg.CNNCfg(
            output_channels=[8, 16, 32],
            kernel_size=[3, 3, 3],
            stride=[2, 2, 2],
            padding="zeros",
            activation="elu",
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
    )

    # PPO 算法超参与 MLP 版一致（见 OpenArmFlipPPORunnerCfg）：熵系数 0.005、
    # 学习率 1e-3 adaptive（同样沿用 2026-09-06 那次训练的设置以保证探索强度）
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
