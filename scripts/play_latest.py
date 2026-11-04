#!/usr/bin/env python3
# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""自动播放"最新一次训练"的最终 checkpoint。

用途
----
每次训练完不用手动翻 logs 目录找模型文件路径：

    python.sh scripts/play_latest.py                        # 初始视角播放（默认）
    python.sh scripts/play_latest.py --cam top              # 或用顶部俯视看标签
    python.sh scripts/play_latest.py --list                 # 只列出候选不播放

脚本扫描 ``logs/rsl_rl/<experiment>/<时间戳>/model_*.pt``，按 checkpoint 的
**修改时间** 选全局最新一个，并自动推断正确任务：

* 目录 ``openarm_flip``     → 状态 MLP 模型 → 任务 ``OpenArm-Flip-State-v0``
* 目录 ``openarm_flip_img`` → 视觉 CNN 模型 → 任务 ``OpenArm-Flip-Play-v0``
  （Play 配置：16 envs / 15 s / 只出 90° 侧面，适合演示）

推断出的 ``--checkpoint``/``--task`` 会原样转发给 ``scripts/play.py`` 执行，
其余参数（--num_envs/--device/--steps/--seed/--headless/--cam）全部透传。
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

# 项目根：本文件在 openarm_lab/scripts/ 下
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl"
ISAAC_ROOT = pathlib.Path(
    os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64")
)
PYTHON_SH = ISAAC_ROOT / "python.sh"

# 实验目录名 -> 该目录下 checkpoint 对应的 gym 任务 id
# （与 openarm_lab.openarm_flip.EXPERIMENT_TASKS 保持一致；此处复制一份是为了
#   让本脚本在启动 Isaac Sim 之前就能完成定位与推断，不依赖 isaaclab 导入。）
EXPERIMENT_TASKS = {
    "openarm_flip": "OpenArm-Flip-State-v0",  # MLP 状态模型（OpenArmFlipPPORunnerCfg）
    "openarm_flip_img": "OpenArm-Flip-Play-v0",  # CNN 视觉模型（OpenArmFlipImgPPORunnerCfg）
}


def find_latest_checkpoint(experiment: str | None) -> tuple[pathlib.Path, str] | None:
    """返回 (最新 checkpoint 路径, 对应任务 id)；没有候选时返回 None。

    规则：遍历 ``logs/rsl_rl/*/`` 下的全部 ``model_*.pt``（可按 ``experiment``
    过滤），按文件 mtime 取最新 —— 文件写入时间即训练保存时间，跨 run 也正确。
    """
    candidates: list[pathlib.Path] = []
    if experiment is not None:
        dirs = [LOG_ROOT / experiment]
    else:
        # 自动模式：扫描 logs/rsl_rl 下所有已知实验目录
        dirs = [LOG_ROOT / name for name in EXPERIMENT_TASKS if (LOG_ROOT / name).exists()]
    for d in dirs:
        if d.exists():
            candidates += list(d.glob("*/model_*.pt"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    # 由 "logs/rsl_rl/<experiment>/<run>/model_*.pt" 反推实验名 -> 任务 id
    experiment = latest.relative_to(LOG_ROOT).parts[0]
    return latest, EXPERIMENT_TASKS[experiment]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play the checkpoint of the latest training run (auto-locate + forward to play.py)."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="只列出 logs 下各 run 的候选 checkpoint，不播放。",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="只在该实验目录里找最新：openarm_flip | openarm_flip_img（默认两者取全局最新）。",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="直接指定 checkpoint（跳过自动查找）。仍会按路径自动推断任务。",
    )
    # ---- 以下参数原样透传给 play.py ----
    parser.add_argument("--num_envs", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default="cuda:0", help=argparse.SUPPRESS)
    parser.add_argument("--steps", type=int, default=600, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cam", type=str, default="", help=argparse.SUPPRESS)
    args, _ = parser.parse_known_args()

    # ---- 1) 定位 checkpoint（或直接用 --checkpoint）----
    if args.checkpoint is not None:
        ckpt = pathlib.Path(args.checkpoint).resolve()  # 允许相对项目根的路径
        if not ckpt.exists():
            sys.exit(f"[play_latest] checkpoint not found: {ckpt}")
    else:
        found = find_latest_checkpoint(args.experiment)
        if found is None:
            if args.experiment is not None:
                sys.exit(f"[play_latest] no model_*.pt found under {LOG_ROOT / args.experiment}")
            sys.exit(f"[play_latest] no model_*.pt found under {LOG_ROOT}")
        ckpt, _ = found
        print(f"[play_latest] latest checkpoint: {ckpt}", flush=True)

    # ---- 2) 由路径推断任务 id（logs/rsl_rl/<experiment>/<run>/model_*.pt）----
    try:
        experiment_dir = ckpt.relative_to(LOG_ROOT).parts[0]
        task = EXPERIMENT_TASKS.get(experiment_dir)
        if task is None:
            sys.exit(
                f"[play_latest] cannot infer task from experiment dir '{experiment_dir}'. "
                f"Known: {sorted(EXPERIMENT_TASKS)}"
            )
    except ValueError:
        sys.exit(f"[play_latest] checkpoint must live under {LOG_ROOT}: {ckpt}")

    print(f"[play_latest] task       : {task}", flush=True)

    # ---- 3) --list 模式：打印后退出 ----
    if args.list:
        print(f"[play_latest] (list mode) would run: --task {task} --checkpoint {ckpt}")
        return

    # ---- 4) 转发给 play.py（在项目根目录下执行，继承终端输出）----
    cmd = [str(PYTHON_SH), "scripts/play.py", "--task", task, "--checkpoint", str(ckpt)]
    for flag, value in (
        ("--num_envs", args.num_envs),
        ("--device", args.device),
        ("--steps", args.steps),
        ("--seed", args.seed),
    ):
        cmd += [flag, str(value)]
    if args.headless:
        cmd += ["--headless"]
    if args.cam:
        cmd += ["--cam", args.cam]
    print(f"[play_latest] running: {' '.join(cmd)}", flush=True)
    sys.exit(subprocess.call(cmd, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()
