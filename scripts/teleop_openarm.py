#!/usr/bin/env python3
# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""用键盘 / SpaceMouse / 手柄手动遥操作 OpenArm 机械臂（阶段 0，不需要 VR 硬件）。

这是"接 PICO 4 Ultra"的前置步骤：先把"输入设备 → SE3 增量 → IK → 关节目标"
这条链路和 OpenArm 的 IK 标定跑通；之后换成 XR 头显只是把输入源替换掉
（XR 走 Isaac Lab 的 IsaacTeleop/CloudXR 栈，落点仍是同一套 IK 动作项）。

用法::

    # 键盘（推荐先跑这个）
    python.sh scripts/teleop_openarm.py --teleop_device keyboard

    # SpaceMouse（3D 鼠标，手感最好）
    python.sh scripts/teleop_openarm.py --teleop_device spacemouse

    # 无 GUI 冒烟：只建环境 + 设备并空跑若干步，验证配置正确
    python.sh scripts/teleop_openarm.py --headless --smoke_steps 60

键盘按键（Isaac Lab ``Se3Keyboard`` 默认映射）::

    位移   W/S = ±X（前后）   A/D = 左/右   Q/E = 上/下
    旋转   Z/X = roll        T/G = pitch  C/V = yaw
    夹爪   K = 切换开合（开 ↔ 合）
    重置   R = 重置环境（盒子回到桌面起点）
    退出   直接关闭窗口，或 Ctrl+C

被操作的臂：**右臂**（7 关节）+ 右夹爪；左臂保持 home 位姿不动。
改左臂/双臂请调 ``openarm_flip/openarm_teleop_env_cfg.py`` 里的 ``TELEOP_ARM_SIDE``。
"""
from __future__ import annotations

import os
import pathlib
import sys

# make the framework package importable when run via `python.sh <script>`
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

# Two Isaac Sim python deps live only inside extension "pip_prebundle"
# directories that the extension system exposes after the app starts, but they
# are needed earlier (AppLauncher imports torch; the PhysX backend resolves the
# 'newton' package). numpy etc. already come from PYTHONPATH.
_ISAAC_ROOT = pathlib.Path(
    os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64")
)
for _rel in (
    "extsDeprecated/omni.isaac.ml_archive/pip_prebundle",  # torch / torchvision
    "exts/isaacsim.pip.newton/pip_prebundle",  # newton physics bindings
):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperate the OpenArm bimanual robot (keyboard/spacemouse).")
parser.add_argument("--task", type=str, default="OpenArm-Teleop-v0", help="Gym task id (teleop env).")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=("keyboard", "spacemouse", "gamepad"),
    help="Input device to teleoperate with.",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Multiplier on the device sensitivity.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (keep 1 for teleop).")
parser.add_argument("--device", type=str, default="cuda:0", help="Simulation device (cpu or cuda:N).")
parser.add_argument("--headless", action="store_true", help="Run without the GUI (no manual input).")
parser.add_argument(
    "--xr",
    action="store_true",
    help="Use an XR headset (PICO 4 Ultra / Quest via CloudXR.js) instead of a local device. "
    "Needs isaaclab_teleop + isaacteleop (see xr_register.py); PICO 4 Ultra requires PICO OS 15.4.4U+.",
)
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help="CloudXR device profile when --xr: 'cloudxrjs' (Quest/Pico, default), 'avp' (Apple Vision Pro), "
    "'standalone' (headless, no client), or a path to a custom .env profile.",
)
parser.add_argument(
    "--accept_cloudxr_eula",
    action="store_true",
    help="首次使用 CloudXR 时必须：确认你已阅读并接受 NVIDIA CloudXR EULA "
    "(https://docs.nvidia.com/cloudxr-sdk/latest/eula.html)，会写入 ~/.cloudxr/run/eula_accepted。"
    "不传此参数时，CloudXR runtime 启动时会改为在终端交互式询问。",
)
parser.add_argument(
    "--auto_cloudxr",
    action="store_true",
    help="用 isaacteleop 自带的 CloudXRLauncher 自动启动 runtime（官方路径）。"
    "默认关闭：本机 CloudXR SDK 6.1.0 不写 runtime_started 就绪哨兵，官方 launcher 的就绪"
    "检测必然 30s 超时，因此默认改由本脚本自行启动 runtime（见 _start_cloudxr_runtime）。",
)
parser.add_argument(
    "--cloudxr_ready_timeout",
    type=float,
    default=60.0,
    help="自行启动 runtime 时等待它就绪的秒数上限。",
)
parser.add_argument(
    "--xr_pos_scale",
    type=float,
    default=1.0,
    help="手柄 → 机械臂的位置缩放：1.0 = 手柄移动 1cm 机械臂也移动 1cm；"
    "调小(如 0.5)更精细，调大更省手臂行程。",
)
parser.add_argument(
    "--xr_rot_scale",
    type=float,
    default=1.0,
    help="手柄 → 机械臂的姿态旋转缩放（1.0 = 1:1）。",
)
parser.add_argument(
    "--xr_view",
    type=str,
    default=None,
    choices=("behind", "side_left", "side_right", "behind_left", "overhead"),
    help="VR 视角预设：你在虚拟场景里的站位与朝向。"
    "behind=站在机器人后方看桌面(默认)、side_left/side_right=站在两侧、"
    "behind_left=斜后方、overhead=旧的高处俯视。",
)
parser.add_argument(
    "--xr_anchor_pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="自定义 VR 视角位置（覆盖 --xr_view）。注意这是〈映射到头显原点的那一点〉，"
    "z 一般填 0（地面），你的头显高度会自动叠加。",
)
parser.add_argument(
    "--xr_anchor_rot",
    type=float,
    nargs=4,
    default=None,
    metavar=("QX", "QY", "QZ", "QW"),
    help="自定义 VR 视角朝向（四元数 xyzw，覆盖 --xr_view）。默认 (0,0,0,1) 面向 +X。",
)
parser.add_argument(
    "--xr_clamp_target",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="把 EE 目标限幅在工作空间安全盒内（默认开）。关掉后手柄可以把目标推到更远处，"
    "但超出可达范围时 IK 会让关节顶住限位。",
)
parser.add_argument(
    "--xr_squeeze_threshold",
    type=float,
    default=0.5,
    help="离合触发键按下的判定阈值（0~1），越大需要按得越深。",
)
parser.add_argument(
    "--xr_clutch_button",
    type=str,
    default="squeeze",
    choices=("squeeze", "trigger", "primary", "secondary", "thumbstick_click", "menu"),
    help="用哪个手柄键作为离合(clutch)触发键，默认 squeeze（侧键）。"
    "若你的手柄侧键在 squeeze 字段上读不到值（不同头显映射不同），"
    "按终端打印的各字段值换一个实际有响应的键。",
)
parser.add_argument(
    "--smoke_steps",
    type=int,
    default=0,
    help="Headless smoke test: step the env this many times with zero commands and exit.",
)
args, _ = parser.parse_known_args()

# XR 依赖隔离目录（isaacteleop + websockets>=14，均用 --no-deps 安装，见 README）
_XR_DEPS_DIR = pathlib.Path(
    os.environ.get("OPENARM_XR_DEPS", "/home/blanc/isaacsim/openarm_lab/.xr_deps")
)

# --- XR 依赖注入（必须在 import openarm_lab 之前！）---
# openarm_teleop_env_cfg 在 import 时会尝试 `from isaaclab_teleop import IsaacTeleopCfg`：
# 路径必须先就位，XR 配置才会被启用（否则该模块的 try/except 会静默降级为"无 XR"）。
# isaacteleop 追加在 sys.path 末尾，让 Isaac Sim 自带的 numpy/torch/scipy 优先。
if args.xr:
    _xr_paths = [
        (pathlib.Path(os.environ.get("ISAACLAB_TELEOP_SRC", "/home/blanc/projects/isaaclab/source/isaaclab_teleop")), True),
        (_XR_DEPS_DIR, False),
    ]
    for _p, _prepend in _xr_paths:
        if _p.exists() and str(_p) not in sys.path:
            if _prepend:
                sys.path.insert(0, str(_p))
            else:
                sys.path.append(str(_p))
        elif not _p.exists():
            print(f"[TELEOP] 警告: XR 依赖路径不存在 -> {_p}", flush=True)

# Isaac Lab 3.0 的 AppLauncher 只有显式请求 "kit" 可视化器才会打开 GUI，
# 否则即使 headless=False 也会被强制无头；这里按 --headless 决定是否请求 Kit GUI。
app_launcher = AppLauncher(
    {
        "headless": args.headless,
        "width": 1280,
        "height": 720,
        "enable_cameras": False,  # 遥操作不需要 scene 相机（只有 GUI 视口渲染）
        "visualizer": None if args.headless else "kit",
        # ⚠️ 关键：必须把 xr=True 交给 AppLauncher，Kit 才会加载 XR experience 与
        # omni.kit.xr.* / isaaclab.python.xr.openxr 扩展。否则：
        #   - GUI 里**没有 XR 面板**（Window 菜单找不到），也就无法点 "Start XR"；
        #   - headless 下也不会自动进入 XR 会话，teleop_device.advance() 永远返回 None。
        # 官方 teleop 脚本是用 AppLauncher.add_app_launcher_args(parser) 传递该标志的。
        "xr": args.xr,
    }
)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm_lab  # noqa: F401  (registers OpenArm-Teleop-v0)
from isaaclab.utils import math as math_utils
from isaaclab.utils.string import string_to_callable

# EE 工具点到指尖的偏移（必须与 openarm_teleop_env_cfg.EE_TO_FINGERTIP 一致）。
EE_TO_FINGERTIP = 0.11
EE_BODY_NAME = "openarm_right_link7"

# EE 目标位置的安全范围（env-local / robot root frame，单位 m）。
# 依据：机械臂基座在原点、桌面在 (0.30, 0.15, 0.38)，手臂伸出约 0.5-0.6 m。
# 目的：防止手柄相对位移把目标推到不可达处 —— 那时 IK 会持续饱和，把关节顶到限位
# （行程最窄的 joint6 只有 ±0.785 rad，最先顶住），看起来就是"某个腕关节狂摆、贴上下限"。
XR_TARGET_MIN = (0.02, -0.45, 0.20)
XR_TARGET_MAX = (0.68, 0.45, 0.80)

# teleop 动作张量末尾按钮字段的顺序（与 env cfg 里 SqueezeRetargeter._FIELDS 一致）
CLUTCH_BUTTON_ORDER = ("squeeze", "trigger", "primary", "secondary", "thumbstick_click", "menu")


def get_ee_frame_pose(env) -> tuple[torch.Tensor, torch.Tensor]:
    """当前末端目标 frame 在 **robot root frame** 下的位姿 (pos(3), quat(4))。

    与 IK 动作项内部的 ``_compute_frame_pose`` 保持完全一致（先转到 root frame，
    再叠加 body_offset），这样它可以直接作为绝对位姿命令的初值。
    """
    robot = env.scene["robot"]
    idx = robot.body_names.index(EE_BODY_NAME)
    pos_b, quat_b = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.torch,
        robot.data.root_quat_w.torch,
        robot.data.body_pos_w.torch[:, idx],
        robot.data.body_quat_w.torch[:, idx],
    )
    # 叠加工具偏移（沿 body 局部 +Z），与动作项的 body_offset 对应
    offset_pos = torch.tensor([0.0, 0.0, EE_TO_FINGERTIP], device=env.device).repeat(env.num_envs, 1)
    offset_rot = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1)
    return math_utils.combine_frame_transforms(pos_b, quat_b, offset_pos, offset_rot)


def _read_env_profile(path: str) -> dict[str, str]:
    """读取 CloudXR 的 ``.env`` profile（``KEY=VALUE`` 行；``#`` 注释与空行忽略）。"""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def _kill_stale_runtime(run_dir: str) -> None:
    """杀掉上一次运行残留的 CloudXR runtime 进程（若有）。

    为什么需要：runtime 的 IPC server 是独占的 —— 残留进程会让新 runtime 启动时报
    ``ERROR_IPC_SERVER_FAILED_TO_START`` 并干净退出（code 0），表现为"runtime 提前退出"。
    残留通常来自上一次运行被 Ctrl+C / timeout 强杀，atexit 清理没跑。

    安全性：只杀 ``/proc/<pid>/cmdline`` 里含 ``isaacteleop`` / ``cloudxr`` 的进程，
    避免误伤其它程序；pid 取自 runtime 自己写的 ``<run>/cloudxr.pid``。
    """
    import signal
    import time as _time

    pid_file = os.path.join(run_dir, "cloudxr.pid")
    if not os.path.isfile(pid_file):
        return
    try:
        with open(pid_file, encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().decode(errors="ignore")
    except OSError:
        return  # 进程已不存在
    if "isaacteleop" not in cmdline and "cloudxr" not in cmdline.lower():
        return  # 不是 CloudXR runtime，别动它
    print(f"[TELEOP] 清理残留的 CloudXR runtime 进程 pid={pid}", flush=True)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            break
        for _ in range(20):  # 最多等 5s
            _time.sleep(0.25)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            continue  # 还活着 → 升级为 SIGKILL
        break
    for stale in ("cloudxr.pid", "runtime_started"):
        try:
            os.remove(os.path.join(run_dir, stale))
        except OSError:
            pass


def _start_cloudxr_runtime(profile: str, ready_timeout: float):
    """自行启动 CloudXR runtime 进程，并等待它就绪（绕过 launcher 的哨兵检测）。

    背景见调用处的注释：本机 CloudXR SDK 6.1.0 不写 ``runtime_started`` 哨兵，
    导致 isaacteleop 的 ``CloudXRLauncher`` 就绪检测必然超时。这里直接以官方
    worker 方式（``python -c "from isaacteleop.cloudxr.runtime import run; run()"``）
    拉起 runtime，就绪判据改为**原生日志出现 "accepting connections"**。

    Args:
        profile: CloudXR ``.env`` profile 路径（决定 ``NV_DEVICE_PROFILE`` 等）。
        ready_timeout: 等待就绪的秒数上限。

    Returns:
        已启动的 ``subprocess.Popen``（已注册 atexit 清理）；失败时退出进程。
    """
    import subprocess
    import time as _time

    install_dir = os.path.abspath(os.environ["CXR_INSTALL_DIR"])
    run_dir = os.path.join(install_dir, "run")
    logs_dir = os.path.join(install_dir, "logs")
    os.makedirs(run_dir, mode=0o755, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    # 先清掉上次残留的 runtime：否则 IPC server 起不来（ERROR_IPC_SERVER_FAILED_TO_START）
    _kill_stale_runtime(run_dir)

    # SDK native 目录（bundled ssl / libcloudxr / openxr json 都在这里）
    from isaacteleop.cloudxr.runtime import get_sdk_path

    sdk_dir = get_sdk_path()

    env = dict(os.environ)
    env.update(
        {
            "CXR_INSTALL_DIR": install_dir,
            "CXR_HOST_VOLUME_PATH": install_dir,
            "NV_CXR_RUNTIME_DIR": run_dir,
            "NV_CXR_OUTPUT_DIR": logs_dir,
            "NV_CXR_FILE_LOGGING": "true",
            "NV_CXR_ENABLE_TENSOR_DATA": "true",
            "NV_CXR_ENABLE_PUSH_DEVICES": "0",
            "NV_ENABLE_POSE_WAIT": "0",
            "XRT_NO_STDIN": "true",
            "XR_RUNTIME_JSON": os.path.join(install_dir, "openxr_cloudxr.json"),
        }
    )
    # profile 文件里的设置（如 cloudxrjs -> NV_DEVICE_PROFILE=auto-webrtc）优先级最高
    env.update(_read_env_profile(profile))

    # ⚠️ 这些变量必须同时设进**主进程**（Isaac Sim / Kit）：
    # Kit 自己就是 OpenXR 客户端，要靠 XR_RUNTIME_JSON 才知道去连 CloudXR runtime；缺了它，
    # XR session 建立不起来 —— 表现为头显里只有 WebXR 客户端 UI 框、没有虚拟画面，且
    # teleop_device.advance() 永远返回 None（脚本一直打印"等待 XR session"）。
    # 官方路径下这些变量由 CloudXRLauncher 写进 ~/.cloudxr/run/cloudxr.env 再 source；
    # 我们绕过了 launcher，所以在主进程里显式设置。
    for _key in (
        "CXR_INSTALL_DIR",
        "CXR_HOST_VOLUME_PATH",
        "NV_CXR_RUNTIME_DIR",
        "NV_CXR_OUTPUT_DIR",
        "NV_CXR_FILE_LOGGING",
        "NV_CXR_ENABLE_TENSOR_DATA",
        "NV_CXR_ENABLE_PUSH_DEVICES",
        "NV_ENABLE_POSE_WAIT",
        "XRT_NO_STDIN",
        "XR_RUNTIME_JSON",
        "NV_DEVICE_PROFILE",
    ):
        if _key in env:
            os.environ[_key] = env[_key]
    print(f"[TELEOP] 主进程 OpenXR 变量已设置: XR_RUNTIME_JSON={os.environ.get('XR_RUNTIME_JSON')}", flush=True)

    # 原生库优先于 Isaac Sim 自带/系统库
    ssl_libs = [os.path.join(sdk_dir, n) for n in ("libcrypto_nvst.so.3", "libssl_nvst.so.3")]
    preload = ":".join([*(p for p in ssl_libs if os.path.isfile(p)), env.get("LD_PRELOAD", "")])
    env["LD_PRELOAD"] = preload.strip(":")
    env["LD_LIBRARY_PATH"] = sdk_dir + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else "")

    worker_code = (
        "import sys; sys.path = [p for p in sys.path if p]\n"
        "from isaacteleop.cloudxr.runtime import run\n"
        "run()\n"
    )
    stdout_log = os.path.join(logs_dir, "runtime_worker_stdout.log")
    print(f"[TELEOP] 自行启动 CloudXR runtime（profile={os.path.basename(profile)}）", flush=True)
    with open(stdout_log, "w", encoding="utf-8") as out:
        proc = subprocess.Popen(
            [sys.executable, "-c", worker_code],
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    import atexit

    def _cleanup() -> None:
        if proc.poll() is None:
            proc.terminate()

    atexit.register(_cleanup)

    # 就绪判据：扫描原生日志，出现下列任一标志即认为 runtime 已在运行。
    #   - "accepting connections"        —— nv_cxr_service_start 完成（理想信号）
    #   - "Signaling server started"     —— 信令服务已起（写在 cxr_streamsdk 日志里）
    #   - "RuntimeSystem::create completed"
    # 注意两点：① 这些行可能分别落在 cxr_server.* 与 cxr_streamsdk.* 两个文件里；
    # ② cxr_server 日志按块缓冲，未 flush 时文件停在 4096 字节边界 —— 因此两个都扫。
    _READY_MARKERS = (
        "accepting connections",
        "Signaling server started",
        "RuntimeSystem::create completed",
    )
    deadline = _time.time() + ready_timeout
    ready = False
    while _time.time() < deadline:
        if proc.poll() is not None:
            print(f"[TELEOP] CloudXR runtime 提前退出（code={proc.returncode}）", flush=True)
            break
        if os.path.isdir(logs_dir):
            for name in sorted(os.listdir(logs_dir), reverse=True):
                if not (name.startswith("cxr_server.") or name.startswith("cxr_streamsdk.")):
                    continue
                try:
                    with open(os.path.join(logs_dir, name), encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except OSError:
                    continue
                if any(m in content for m in _READY_MARKERS):
                    ready = True
                    break
        if ready:
            break
        _time.sleep(1.0)

    if not ready:
        print(
            "[TELEOP] CloudXR runtime 在 "
            f"{ready_timeout:.0f}s 内未就绪；详见 {logs_dir}（可继续，但头显可能连不上）",
            flush=True,
        )
    else:
        print("[TELEOP] CloudXR runtime 已就绪（原生日志显示信令服务在运行）", flush=True)

    # 让 session_lifecycle 不要再去自启 runtime（否则又会撞上哨兵超时）
    os.environ["ISAACLAB_CXR_SKIP_AUTOLAUNCH"] = "1"
    return proc


def _ensure_websockets_v14(xr_deps_dir: str) -> str:
    """确保能 import 到 ``websockets >= 14``（isaacteleop 的 WSS 代理需要新 asyncio API）。

    Isaac Sim 自带 websockets 12（只有旧 ``websockets.serve`` API），而
    ``isaacteleop/cloudxr/wss.py`` 直接 ``from websockets.asyncio.server import serve``
    且无回退 —— 缺了就 ``sys.exit("Missing dependency: websockets >= 14")``。
    因此把隔离目录 ``.xr_deps`` 前置到 sys.path，并丢弃已缓存的旧版本模块。

    Returns:
        实际生效的 websockets 版本号。
    """
    import sys

    if xr_deps_dir not in sys.path:
        sys.path.insert(0, xr_deps_dir)  # 只含 isaacteleop/websockets 等隔离包，不会覆盖 Isaac 的 numpy/torch
    for name in [n for n in sys.modules if n == "websockets" or n.startswith("websockets.")]:
        del sys.modules[name]
    import websockets
    from websockets.asyncio.server import serve  # noqa: F401  —— 只是验证新 API 存在

    return getattr(websockets, "__version__", "unknown")


def _start_wss_proxy(logs_dir: str, xr_deps_dir: str):
    """启动 CloudXR 的 WSS/TLS 代理（默认 48322 端口），供头显浏览器连入。

    **为什么必须自己启动**：这个代理原本由 isaacteleop 的 ``CloudXRLauncher`` 在
    runtime 就绪后拉起，但我们为了绕开它那套必然超时的 ``runtime_started`` 哨兵检测，
    改成了自行启动 runtime（见 ``_start_cloudxr_runtime``）—— 于是代理也不会被启动，
    头显访问 ``https://<PC IP>:48322/``（接受自签名证书那一步）会"无法连接"。

    这里直接复用 isaacteleop 的实现（``isaacteleop.cloudxr.wss.run``，默认把
    48322 上的 WebRTC/WS 流量转发到 runtime 的 49100 端口），与 launcher 的做法一致：
    在 daemon 线程里跑一个独立的 asyncio 事件循环。
    """
    import asyncio
    import threading
    import time as _time

    ws_version = _ensure_websockets_v14(xr_deps_dir)
    from isaacteleop.cloudxr.wss import run as wss_run

    ts = _time.strftime("%Y-%m-%dT%H%M%SZ", _time.gmtime())
    log_path = os.path.join(logs_dir, f"wss.{ts}.log")
    loop = asyncio.new_event_loop()
    stop_future = loop.create_future()

    def _run_wss() -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                wss_run(
                    log_file_path=log_path,
                    stop_future=stop_future,
                    backend_host="localhost",
                    backend_port=49100,  # runtime 的信令端口（见 cxr_server 日志 serverPort）
                )
            )
        except Exception:
            import traceback

            print("[TELEOP] WSS 代理线程异常退出:", flush=True)
            traceback.print_exc()
        finally:
            loop.close()

    thread = threading.Thread(target=_run_wss, name="cloudxr-wss-proxy", daemon=True)
    thread.start()

    # 等端口真正可连（头显就是连这个端口；失败时给出明确提示而不是让用户对着"无法连接"发愣）
    import socket

    deadline = _time.time() + 20.0
    ok = False
    while _time.time() < deadline:
        if not thread.is_alive():
            break
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", 48322)) == 0:
                ok = True
                break
        _time.sleep(0.5)
    print(
        f"[TELEOP] WSS 代理(48322) {'已就绪' if ok else '未就绪'} "
        f"| websockets {ws_version} | 日志 {log_path}",
        flush=True,
    )
    return stop_future, loop


def make_teleop_device(device_name: str, devices_cfg, callbacks: dict):
    """按名字构造遥操作设备（延迟导入，避免加载用不到的设备依赖）。

    * ``keyboard``   —— 无额外依赖，随时可用。
    * ``spacemouse`` —— 需要 ``hid`` 包（``pip install hidapi`` 后 ``import hid``）；
      未安装时会给出明确报错而不是崩溃在 import 阶段。
    * ``gamepad``    —— 需要 ``pygame``。

    不走 ``isaaclab.devices.teleop_device_factory``：那个工厂会 import
    ``isaaclab_teleop``（属于 Isaac Lab 的 ``teleop`` extra，只在接入 XR/CloudXR
    时才需要安装），阶段 0 用键盘/SpaceMouse 没必要为它装整套 teleop 依赖。
    """
    if device_name == "keyboard":
        from isaaclab.devices import Se3Keyboard

        device_cls = Se3Keyboard
    elif device_name == "spacemouse":
        try:
            from isaaclab.devices import Se3SpaceMouse
        except ModuleNotFoundError as e:  # 缺 hid 包
            raise SystemExit(
                "SpaceMouse 需要 'hid' 包：先运行 "
                f"'{os.environ.get('ISAAC_ROOT', '')}/kit/python/bin/python3 -m pip install hidapi' "
                f"（原始错误：{e}）"
            ) from e
        device_cls = Se3SpaceMouse
    elif device_name == "gamepad":
        from isaaclab.devices import Se3Gamepad

        device_cls = Se3Gamepad
    else:
        raise SystemExit(f"Unknown teleop device '{device_name}' (choose keyboard/spacemouse/gamepad)")

    device = device_cls(devices_cfg[device_name])
    for key, cb in callbacks.items():
        device.add_callback(key, cb)
    return device


def main() -> None:
    # --- 0) XR 模式：先清掉上次残留的 CloudXR runtime ---
    # 必须放在创建环境**之前**：残留 runtime 会占住 GPU 显存，导致 PhysX 初始化失败
    # （表现为创建 env 时 "Failed to get DOF velocities from backend"）。
    if args.xr:
        os.environ.setdefault("CXR_INSTALL_DIR", os.path.expanduser("~/.cloudxr"))
        _kill_stale_runtime(os.path.join(os.environ["CXR_INSTALL_DIR"], "run"))

    # --- 1) 从 gym spec 拿注册的 env cfg 类并实例化 ---
    spec = gym.spec(args.task)
    env_cfg = string_to_callable(spec.kwargs["env_cfg_entry_point"])()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    # --- 1b) VR 视角（XR 锚点）覆盖 ---
    # anchor_pos 是"映射到头显空间原点的那一点"：头显高度会自动叠加，故 z 通常填 0。
    if args.xr and getattr(env_cfg, "xr", None) is not None:
        if args.xr_view is not None:
            from openarm_lab.openarm_flip.openarm_teleop_env_cfg import XR_VIEW_PRESETS

            preset = XR_VIEW_PRESETS[args.xr_view]
            env_cfg.xr.anchor_pos = preset["pos"]
            env_cfg.xr.anchor_rot = preset["rot"]
            print(f"[TELEOP] VR 视角预设 '{args.xr_view}': pos={preset['pos']} rot={preset['rot']}", flush=True)
        if args.xr_anchor_pos is not None:
            env_cfg.xr.anchor_pos = tuple(args.xr_anchor_pos)
            print(f"[TELEOP] VR 视角位置(自定义): {env_cfg.xr.anchor_pos}", flush=True)
        if args.xr_anchor_rot is not None:
            env_cfg.xr.anchor_rot = tuple(args.xr_anchor_rot)
            print(f"[TELEOP] VR 视角朝向(自定义): {env_cfg.xr.anchor_rot}", flush=True)

    # --- 2) 创建环境（不包 rsl-rl wrapper：遥操作直接喂原始动作向量） ---
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()

    # 打印动作空间构成，便于确认 IK(7) + 夹爪(1) = 8 维的划分正确
    terms = env.action_manager.active_terms
    dims = [env.action_manager.get_term(name).action_dim for name in terms]
    print(f"[TELEOP] action terms: {list(zip(terms, dims))} (total {sum(dims)})", flush=True)
    print("[TELEOP] device gives 7 dims (6x SE3 delta + 1x gripper); script converts to "
          "8 dims (7x absolute pose + gripper)", flush=True)

    # --- 3) 无 GUI 冒烟 ---
    # (a) 零指令跑：验证"松手不下坠"（绝对目标模式的核心收益）
    # (b) 注入位移：验证 IK 真能驱动机械臂
    # （放在创建设备之前：headless 下没有输入子系统，冒烟不需要设备）
    if args.smoke_steps > 0:
        robot = env.scene["robot"]
        ee_idx = robot.body_names.index(EE_BODY_NAME)
        gripper = torch.ones(env.num_envs, 1, device=env.device)  # +1 = 张开

        def make_cmd(delta6: torch.Tensor) -> torch.Tensor:
            """把 6 维增量叠加到维护的 target 上，输出 8 维绝对位姿命令。"""
            nonlocal target_pos, target_quat
            target_pos, target_quat = math_utils.apply_delta_pose(target_pos, target_quat, delta6)
            return torch.cat([target_pos, target_quat, gripper], dim=-1)

        # target 初值 = 当前 EE frame 位姿（root frame）
        target_pos, target_quat = get_ee_frame_pose(env)

        hold_steps = max(2, args.smoke_steps // 2)
        move_steps = max(2, args.smoke_steps - hold_steps)

        # (a) 松手保持：零增量
        ee0 = robot.data.body_pos_w.torch[:, ee_idx].clone()
        for _ in range(hold_steps):
            env.step(make_cmd(torch.zeros(env.num_envs, 6, device=env.device)))
        ee_hold = robot.data.body_pos_w.torch[:, ee_idx].clone()
        hold_drift = (ee_hold - ee0)[0].norm().item()

        # (b) 注入位移：+x 前移、+z 抬升
        delta = torch.zeros(env.num_envs, 6, device=env.device)
        delta[:, 0] = 0.004  # 每步 +4 mm
        delta[:, 2] = 0.002  # 每步 +2 mm
        jp0 = robot.data.joint_pos.torch.clone()
        for _ in range(move_steps):
            env.step(make_cmd(delta))
        jp1 = robot.data.joint_pos.torch
        ee1 = robot.data.body_pos_w.torch[:, ee_idx]

        print(
            f"[TELEOP] 松手保持 {hold_steps} 步：EE 漂移 {hold_drift:.4f} m "
            f"(应远小于 <0.02，绝对目标模式下重力下垂被闭环消除)",
            flush=True,
        )
        print(
            f"[TELEOP] 施加位移 {move_steps} 步：关节最大变化 {(jp1 - jp0).abs().max().item():.4f} rad | "
            f"EE 位移 {(ee1 - ee_hold)[0].tolist()} (Δ={((ee1 - ee_hold)[0].norm().item()):.4f} m)",
            flush=True,
        )
        moved = (ee1 - ee_hold)[0].norm().item() > 0.01
        print(
            f"[TELEOP] 结论: IK {'WORKS' if moved else 'DID NOT MOVE — 检查 body_name / joint_names / IK 配置'} | "
            f"松手{'不漂移 OK' if hold_drift < 0.02 else '仍在漂移（检查 IK 模式 / 刚度）'}",
            flush=True,
        )
        env.close()
        return

    # --- 4) 准备设备回调（R 键 / 头显内按钮 → 重置环境） ---
    should_reset = False

    def _request_reset() -> None:
        nonlocal should_reset
        should_reset = True
        print("[TELEOP] reset requested", flush=True)

    callbacks = {"R": _request_reset}

    # --- 5) XR 分支（PICO 4 Ultra / Quest via CloudXR.js）---
    # IsaacTeleop 的重定向器直接输出 8 维**绝对位姿 + 夹爪**动作，与我们的动作项布局
    # 完全一致，因此不需要像键盘那样在脚本里维护 target —— 直接把设备输出喂给 env。
    if args.xr:
        if env_cfg.isaac_teleop is None:
            raise SystemExit(
                "[TELEOP] --xr 需要 isaaclab_teleop + isaacteleop；当前 env cfg 未启用 XR 配置。\n"
                "  安装（务必 --no-deps，避免 pip 升级 torch/numpy 破坏 Isaac Sim）:\n"
                "    $ISAAC_ROOT/kit/python/bin/python3 -m pip install --no-deps "
                "--target /home/blanc/isaacsim/openarm_lab/.xr_deps 'isaacteleop==1.4.142'\n"
                "  并确认 isaaclab_teleop 路径存在（见 xr_register.py 顶部说明）。"
            )
        from isaaclab_teleop import (
            CLOUDXR_AVP_ENV,
            CLOUDXR_JS_ENV,
            CLOUDXR_STANDALONE_ENV,
            create_isaac_teleop_device,
            poll_control_events,
        )

        profile = {
            "cloudxrjs": CLOUDXR_JS_ENV,
            "avp": CLOUDXR_AVP_ENV,
            "standalone": CLOUDXR_STANDALONE_ENV,
        }.get(args.cloudxr_env, args.cloudxr_env)

        # CXR_INSTALL_DIR 决定 CloudXR 的 run/logs 目录，launcher 与 runtime 必须一致
        os.environ.setdefault("CXR_INSTALL_DIR", os.path.expanduser("~/.cloudxr"))

        # --- 启动 CloudXR runtime ---
        # 默认**不走** isaacteleop 的 CloudXRLauncher：本机 CloudXR SDK 6.1.0
        # （apiVersion 1.4.329）启动后不会写 ``<run>/runtime_started`` 就绪哨兵，
        # 而 launcher 正是轮询该文件（30 s 超时），于是必然报
        #   "CloudXR runtime failed to start within 30s. Process exited with code 1."
        # —— 尽管原生日志显示 runtime 其实已经 "accepting connections"。
        # （尝试过补写该哨兵，但它同时是 runtime 的 lock file：预先存在会让 runtime
        #   认为已有实例而拒绝启动，反而更糟。）
        # 因此这里按官方支持的方式自行拉起 runtime 进程，并用
        # ISAACLAB_CXR_SKIP_AUTOLAUNCH=1 让 session_lifecycle 不再自启。
        _cloudxr_proc = None
        if not args.auto_cloudxr:
            _cloudxr_proc = _start_cloudxr_runtime(profile, args.cloudxr_ready_timeout)
            # runtime 起来后必须再启动 WSS 代理（48322）—— 否则头显访问该端口会"无法连接"，
            # 因为该代理原本由（被我们绕过的）CloudXRLauncher 启动。
            _start_wss_proxy(
                os.path.join(os.environ["CXR_INSTALL_DIR"], "logs"),
                str(_XR_DEPS_DIR),
            )

        # 首次运行需接受 CloudXR EULA（标记写入 <CloudXR install dir>/run/eula_accepted）。
        # 传 --accept_cloudxr_eula = 你确认接受该协议；否则库会在启动时于终端交互式询问。
        if args.accept_cloudxr_eula:
            try:
                from isaacteleop.cloudxr.runtime import check_eula

                check_eula(accept_eula=True)
                print(
                    "[TELEOP] 已记录 CloudXR EULA 接受标记（"
                    f"{os.path.join(os.environ['CXR_INSTALL_DIR'], 'run', 'eula_accepted')}）",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[TELEOP] 自动写 EULA 标记失败（{e}）。\n"
                    "         请改为在终端交互确认：直接跑 --xr（不加本参数），"
                    "看到 'Accept NVIDIA CloudXR EULA? [y/N]:' 时输入 y。",
                    flush=True,
                )

        teleop_device = create_isaac_teleop_device(
            env_cfg.isaac_teleop,
            sim_device=args.device,
            callbacks=callbacks,
            cloudxr_env_file=profile,
            auto_launch_cloudxr=True,
        )

        # 打印头显端需要的信息（CloudXR.js WebXR 客户端 + 本机局域网 IP）
        def _local_ip() -> str:
            """取本机局域网 IP（用于在头显里填 Server IP）。

            不用 gethostbyname(gethostname())：很多机器把它解析成 127.0.1.1。
            这里用 UDP "connect" 让内核挑选出口网卡（不实际发包）。
            """
            import socket

            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
            except Exception:
                try:
                    return socket.gethostbyname(socket.gethostname())
                except Exception:
                    return "<本机局域网 IP>"
            finally:
                s.close()

        ip = _local_ip()
        print(
            "\n[TELEOP] ================ XR 模式（PICO 4 Ultra / Quest）================\n"
            f"[TELEOP] CloudXR profile : {profile}\n"
            f"[TELEOP] 头显浏览器打开  : https://nvidia.github.io/IsaacTeleop/client/release-1.4.x\n"
            f"[TELEOP] Server IP 填    : {ip}\n"
            f"[TELEOP] 首次需接受自签证书: 先点页面上的 https://{ip}:48322/ 链接 "
            "→ Advanced → Proceed，然后回到客户端点 Connect\n"
            "[TELEOP] PICO 4 Ultra 需 PICO OS 15.4.4U+（必须 HTTPS 模式）\n"
            "[TELEOP] 头显内可控制：开始/暂停、重置环境\n"
            "[TELEOP] =============================================================\n",
            flush=True,
        )

        active = True
        auto_started = False
        step_count = 0
        last_report = 0

        # --- 离合(clutch)状态 ---
        # 需求：**只有按住手柄侧键(squeeze)时，机械臂才跟随手柄的相对位移**。
        # 松开即冻结目标 —— 这样可以把手柄挪回舒适位置再按住继续"接着推"，
        # 不必让手柄的绝对位置与机械臂工作空间对应（手柄空间远小于机械臂可达范围）。
        target_pos, target_quat = get_ee_frame_pose(env)  # 当前 EE 目标（root frame）
        clutching = False
        ref_dev_pos = ref_dev_quat = ref_ee_pos = ref_ee_quat = None
        SQUEEZE_THRESHOLD = args.xr_squeeze_threshold
        last_btn_state: dict[str, float] = {n: 0.0 for n in CLUTCH_BUTTON_ORDER}
        clamp_warned = [False]  # 限幅提示只打一次

        with teleop_device:
            while simulation_app.is_running():
                with torch.inference_mode():
                    action = teleop_device.advance()  # None = 头显尚未连接
                    ctrl = poll_control_events(teleop_device)
                    if ctrl.is_active is not None:
                        active = ctrl.is_active
                    if ctrl.should_reset:
                        env.reset()
                        teleop_device.reset(pause=True)
                        # 重置后目标重新对齐到当前 EE 位姿，并清掉离合参考
                        target_pos, target_quat = get_ee_frame_pose(env)
                        clutching = False
                        print("[TELEOP] environment reset (from headset)", flush=True)

                    # 自动"开始"：XR session 建立后若头显仍处于 STOPPED，主动请求开始，
                    # 免得用户对着画面找不到那个 START 按钮（机械臂完全不动最常见的成因）。
                    # 一旦开始过就不再干预 —— 用户随后在头显里按 PAUSE 仍会被尊重。
                    if action is not None and not auto_started and not active:
                        teleop_device.request_start()
                        active = True
                        auto_started = True
                        print("[TELEOP] 已自动请求开始（无需在头显里点 START）", flush=True)
                    elif ctrl.is_active:
                        auto_started = True

                    if action is not None:
                        # 统一成 (1, k) 形状，与 target_pos/target_quat 的 (N=1, ...) 对齐
                        a = action.detach().reshape(-1).float()
                        dev_pos = a[0:3].reshape(1, 3)
                        dev_quat = a[3:7].reshape(1, 4)
                        gripper = a[7:8].reshape(1, 1)
                        # 第 9~14 维是按钮组（见 env cfg：squeeze/trigger/primary/secondary/
                        # thumbstick_click/menu）。离合触发键可切换 —— 实测 PICO 4 Ultra 的
                        # 侧键不一定报在 squeeze 字段上，用 --xr_clutch_button 指定实际有反应的键。
                        btn = a[8:] if a.numel() >= 9 else torch.zeros(0, device=a.device)
                        btn_state = {
                            name: (float(btn[i]) if i < btn.numel() else 0.0)
                            for i, name in enumerate(CLUTCH_BUTTON_ORDER)
                        }
                        clutch_value = btn_state.get(args.xr_clutch_button, 0.0)
                        last_btn_state = btn_state
                        pressed = clutch_value > SQUEEZE_THRESHOLD

                        if pressed:
                            if not clutching:
                                # 侧键刚按下：以当前手柄位姿与当前 EE 目标为基准
                                ref_dev_pos = dev_pos.clone()
                                ref_dev_quat = dev_quat.clone()
                                ref_ee_pos = target_pos.clone()
                                ref_ee_quat = target_quat.clone()
                                clutching = True
                                print("[TELEOP] clutch 按下 → 开始跟随手柄相对位移", flush=True)
                            # 位置：EE 目标 = 基准 EE + 手柄的相对位移 × 缩放
                            target_pos = ref_ee_pos + (dev_pos - ref_dev_pos) * args.xr_pos_scale
                            # 目标位置限幅：手柄相对位移若把 EE 目标推出可达范围，IK 会持续
                            # 饱和、把关节顶到限位（行程最窄的 joint6 只有 ±0.785 rad），
                            # 表现为"某个腕关节大幅摆动、常贴上下限"。
                            # 这里把目标夹在工作空间内的安全盒子里，超出即削平。
                            if args.xr_clamp_target:
                                clamped = torch.clamp(
                                    target_pos,
                                    min=torch.tensor(XR_TARGET_MIN, device=target_pos.device),
                                    max=torch.tensor(XR_TARGET_MAX, device=target_pos.device),
                                )
                                if not torch.allclose(clamped, target_pos) and not clamp_warned[0]:
                                    print(
                                        "[TELEOP] EE 目标已限幅到工作空间安全范围 "
                                        f"{XR_TARGET_MIN} ~ {XR_TARGET_MAX}"
                                        "（要更大范围：--no-xr_clamp_target，或调小 --xr_pos_scale）",
                                        flush=True,
                                    )
                                    clamp_warned[0] = True
                                target_pos = clamped
                            # 姿态：EE 目标 = 基准 EE 姿态 ⊗ (基准手柄姿态⁻¹ ⊗ 当前手柄姿态)
                            #       旋转部分按 --xr_rot_scale 做 slerp（1.0 = 完全跟随）
                            dq = math_utils.quat_mul(
                                math_utils.quat_inv(ref_dev_quat), dev_quat
                            )
                            if args.xr_rot_scale != 1.0:
                                dq = math_utils.quat_slerp(
                                    torch.tensor([0.0, 0.0, 0.0, 1.0], device=dq.device).repeat(dq.shape[0], 1),
                                    dq,
                                    args.xr_rot_scale,
                                )
                            target_quat = math_utils.quat_mul(ref_ee_quat, dq)
                        elif clutching:
                            # 松开：目标冻结在当前位置（手柄可自由重新摆位）
                            clutching = False
                            print("[TELEOP] clutch 松开 → 机械臂保持不动", flush=True)

                        cmd = torch.cat([target_pos, target_quat, gripper], dim=-1)
                        if active:
                            env.step(cmd.repeat(env.num_envs, 1))
                        else:
                            env.sim.render()
                    else:
                        env.sim.render()  # 等待头显连接时保持视口刷新

                # 每 ~5 秒报告一次 XR 状态，便于判断"手柄到底有没有输入"：
                #   action=None            → 头显未连接/未进入沉浸模式
                #   数值随手柄移动变化      → 手柄追踪正常
                #   数值恒定不动            → 手柄未被追踪（未握持/未唤醒，或握的是另一只手柄）
                step_count += 1
                if step_count - last_report >= 150:
                    last_report = step_count
                    if action is None:
                        print("[XR] 等待 XR session（头显未连接或未进入沉浸模式）", flush=True)
                    else:
                        a = action.detach().reshape(-1).float()
                        if a.numel() >= 9:
                            btns = " ".join(
                                f"{n}={last_btn_state.get(n, 0.0):+.2f}"
                                for n in CLUTCH_BUTTON_ORDER
                            )
                            detail = (
                                f"pos=({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f}) "
                                f"gripper={a[7]:+.2f} | 按钮: {btns}"
                            )
                        else:
                            detail = f"dims={a.numel()}"
                        print(
                            f"[XR] active={active} clutching={clutching} | dims={a.numel()} | {detail}",
                            flush=True,
                        )
        env.close()
        return

    # --- 6) 本地设备分支（键盘 / SpaceMouse / 手柄）---
    teleop_device = make_teleop_device(args.teleop_device, env_cfg.teleop_devices.devices, callbacks)
    if args.teleop_device in ("keyboard", "spacemouse"):
        # 灵敏度：设备 cfg 里已有默认值，这里按 --sensitivity 再缩放一层
        for attr in ("pos_sensitivity", "rot_sensitivity"):
            if hasattr(teleop_device, attr):
                setattr(teleop_device, attr, getattr(teleop_device, attr) * args.sensitivity)
    print(f"[TELEOP] using device: {teleop_device}", flush=True)

    # 关键：维护一个**固定的绝对目标位姿** target。设备增量只在按键时累加到
    # target 上；松手时 target 不变，IK 每步都以它纠偏 —— 这样重力下垂不会像
    # 相对模式那样被逐步"固化"（相对模式实测 300 步掉 0.22 m）。
    target_pos, target_quat = get_ee_frame_pose(env)
    print(
        "\n[TELEOP] 键盘: W/S=前后 A/D=左右 Q/E=上下 | Z/X=roll T/G=pitch C/V=yaw | "
        "K=夹爪开合 | R=重置环境\n",
        flush=True,
    )
    teleop_device.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            # 设备输出 7 维：[dx,dy,dz, rx,ry,rz, gripper(±1)]
            device_cmd = teleop_device.advance().repeat(env.num_envs, 1)
            # 增量累加到固定 target（与相对模式下 apply_delta_pose 的手感一致）
            target_pos, target_quat = math_utils.apply_delta_pose(
                target_pos, target_quat, device_cmd[:, :6]
            )
            # 绝对位姿命令 [pos(3), quat(4)] + 夹爪(1) = 8 维
            actions = torch.cat([target_pos, target_quat, device_cmd[:, 6:]], dim=-1)
            env.step(actions)
            if should_reset:
                env.reset()
                teleop_device.reset()
                # 重置后把 target 重新同步到当前 EE 位姿，避免"追一个旧目标"
                target_pos, target_quat = get_ee_frame_pose(env)
                should_reset = False
                print("[TELEOP] environment reset", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
