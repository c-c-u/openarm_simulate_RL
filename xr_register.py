# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""供 Isaac Lab 官方遥操作脚本使用的"外部任务注册"回调。

Isaac Lab 的 ``scripts/environments/teleoperation/teleop_se3_agent.py`` 不会自动
import 第三方任务包，因此需要 ``--external_callback`` 指定一个函数，在解析任务之前
把我们的 gym 任务注册进去。

本模块同时负责把 XR 依赖注入 ``sys.path``（它们不随本项目基础安装进来）：

* ``isaaclab_teleop``  —— Isaac Lab 仓库内的包（``source/isaaclab_teleop``），
  用 editable 安装或 PYTHONPATH 注入；
* ``isaacteleop``      —— pip 包，**必须用 ``--no-deps`` 安装到独立目录**：它的依赖
  声明会让 pip 升级 torch / numpy / CUDA 包，从而破坏 Isaac Sim 自带环境
  （Isaac Sim 自带的 numpy 2.3.1 / scipy 1.17.0 / torch 2.11.0 已满足其要求）。
  默认目录可用环境变量 ``OPENARM_XR_DEPS`` 覆盖。

用法（PICO 4 Ultra / Quest，经 CloudXR.js）::

    PYTHONPATH=/home/blanc/isaacsim \\
    $ISAAC_ROOT/python.sh <isaaclab>/scripts/environments/teleoperation/teleop_se3_agent.py \\
        --task OpenArm-Teleop-v0 \\
        --visualizer kit --xr \\
        --external_callback openarm_lab.xr_register.register_openarm_teleop
"""
from __future__ import annotations

import os
import pathlib
import sys

# 默认的 XR 依赖位置（均可用环境变量覆盖）
_XR_DEPS_DIR = pathlib.Path(
    os.environ.get("OPENARM_XR_DEPS", "/home/blanc/isaacsim/openarm_lab/.xr_deps")
)
_ISAACLAB_TELEOP_SRC = pathlib.Path(
    os.environ.get("ISAACLAB_TELEOP_SRC", "/home/blanc/projects/isaaclab/source/isaaclab_teleop")
)


def _inject(path: pathlib.Path, *, prepend: bool) -> None:
    """把路径加入 sys.path（已存在则跳过）。"""
    if path.exists() and str(path) not in sys.path:
        if prepend:
            sys.path.insert(0, str(path))
        else:
            # isaacteleop 放后面：让 Isaac Sim 自带的 numpy/torch/scipy 优先被使用
            sys.path.append(str(path))


def register_openarm_teleop():
    """注册 OpenArm 的 gym 任务并注入 XR 依赖路径。

    Returns:
        本回调不新增任何 CLI 参数，故返回空列表（官方脚本用它做未识别参数检查）。
    """
    _inject(_ISAACLAB_TELEOP_SRC, prepend=True)
    _inject(_XR_DEPS_DIR, prepend=False)

    import openarm_lab  # noqa: F401  —— import 即完成 gym.register

    missing = []
    try:
        import isaaclab_teleop  # noqa: F401
    except ModuleNotFoundError:
        missing.append(f"isaaclab_teleop（期望在 {_ISAACLAB_TELEOP_SRC}）")
    try:
        import isaacteleop  # noqa: F401
    except ModuleNotFoundError:
        missing.append(f"isaacteleop（期望在 {_XR_DEPS_DIR}）")
    if missing:
        print(
            "[xr_register] 缺少 XR 依赖: " + "、".join(missing) + "\n"
            "  安装 isaacteleop（务必 --no-deps，避免升级 torch/numpy）:\n"
            f"  $ISAAC_ROOT/kit/python/bin/python3 -m pip install --no-deps "
            f"--target {_XR_DEPS_DIR} 'isaacteleop==1.4.142'",
            flush=True,
        )
    else:
        print("[xr_register] OpenArm 任务已注册，XR 依赖就绪", flush=True)
    return []
