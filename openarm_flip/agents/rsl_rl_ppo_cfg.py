# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for the OpenArm parcel-flip task."""
from __future__ import annotations

from isaaclab_rl.rsl_rl import (
    RslRlCNNModelCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils.configclass import configclass


@configclass
class OpenArmFlipPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO 超参（rsl-rl 5.x，MLP actor/critic）——给纯状态任务 OpenArm-Flip-State-v0 用。"""

    # 每环境每迭代 24 步；最多 20000 迭代；每 200 迭代存一次 checkpoint。
    # experiment_name 决定日志目录：logs/rsl_rl/openarm_flip/
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 200
    experiment_name = "openarm_flip"

    # 观测组映射：环境只暴露一个 "policy" 组（68 维状态），actor 与 critic 共用
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    # actor：高斯分布策略网络（init_std=1.0），观测归一化 + 三层 MLP [256,128,64]（elu）
    actor: RslRlMLPModelCfg = RslRlMLPModelCfg(
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
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
    # 自适应学习率（目标 KL 0.01）、grad norm 1.0、gamma/lam 0.99/0.95
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
        distribution_cfg=RslRlCNNModelCfg.GaussianDistributionCfg(init_std=1.0),
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

    # PPO 算法超参与 MLP 版一致（见 OpenArmFlipPPORunnerCfg）
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
