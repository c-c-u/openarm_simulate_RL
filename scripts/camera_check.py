"""校验场景相机能输出有效图像（前视 + 左右腕部 3 个相机，RGB/深度）。

对每个相机打几帧（零动作），打印 rgb / distance_to_image_plane 的
shape 与 mean/std/min/max，用于排查渲染管线或相机安装问题。
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

_ISAAC_ROOT = pathlib.Path(
    os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64")
)
for _rel in (
    "extsDeprecated/omni.isaac.ml_archive/pip_prebundle",
    "exts/isaacsim.pip.newton/pip_prebundle",
):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--env_id", type=int, default=0, help="保存哪个环境（默认 env 0）的图片。")
parser.add_argument(
    "--save",
    type=str,
    default="",
    help="保存图片的目录（如 /tmp/cam）。每个相机存一张 rgb png（深度存归一化灰度图），"
    "便于肉眼确认画面里确实有机器人与盒子。",
)
args = parser.parse_known_args()[0]

app_launcher = AppLauncher(
    {"headless": True, "width": 320, "height": 240, "enable_cameras": True}
)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import openarm_lab  # noqa: F401
from isaaclab.utils.string import string_to_callable


def main() -> None:
    # 直接用默认的视觉任务 cfg（enable_image_obs=True -> 三个相机都在）
    spec = gym.spec("OpenArm-Flip-v0")
    env_cfg = string_to_callable(spec.kwargs["env_cfg_entry_point"])()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"
    env = gym.make("OpenArm-Flip-v0", cfg=env_cfg)
    env.reset()

    # 列出场景里注册的传感器（相机也是 sensor）
    names = list(env.unwrapped.scene.sensors.keys())
    print(f"CAMCHECK: sensors in scene: {names}", flush=True)

    # 先空动作跑几步，让各相机渲染器真正出帧
    action_dim = env.unwrapped.action_manager.total_action_dim
    for _ in range(10):
        env.step(torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device))

    import numpy as np
    # 逐个相机读输出并打印统计（任一相机失败都单独报错，不影响其余）
    for name in names:
        cam = env.unwrapped.scene.sensors[name]
        if not hasattr(cam.data, "output"):
            continue  # 接触传感器等非相机传感器，跳过
        try:
            rgb = cam.data.output["rgb"]
            depth = cam.data.output.get("distance_to_image_plane", None)
            if hasattr(rgb, "torch"):
                rgb = rgb.torch
            arr = np.asarray(rgb.cpu())
            print(
                f"CAMCHECK: {name}: rgb shape={arr.shape} dtype={arr.dtype} "
                f"mean={arr.mean():.1f} std={arr.std():.1f} min={arr.min():.0f} max={arr.max():.0f}",
                flush=True,
            )
            if depth is not None:
                d = np.asarray(depth.cpu()) if not hasattr(depth, "torch") else np.asarray(depth.torch.cpu())
                print(f"CAMCHECK: {name}: depth shape={d.shape} mean={d.mean():.3f}", flush=True)
            # 保存图片：rgb 直接存；depth 做 min-max 归一化后存灰度（仅为肉眼检查）
            if args.save and arr.ndim == 4:
                from PIL import Image

                out = pathlib.Path(args.save)
                out.mkdir(parents=True, exist_ok=True)
                img = arr[args.env_id]
                Image.fromarray(img.astype(np.uint8)).save(out / f"{name}_rgb.png")
                print(f"CAMCHECK: saved {out / f'{name}_rgb.png'}  shape={img.shape}", flush=True)
                if depth is not None:
                    dd = np.asarray(depth.cpu()) if not hasattr(depth, "torch") else np.asarray(depth.torch.cpu())
                    if dd.ndim == 4:
                        dimg = dd[args.env_id, :, :, 0] if dd.shape[-1] == 1 else dd[args.env_id]
                        dimg = np.asarray(dimg, dtype=np.float32)
                        finite = np.isfinite(dimg)
                        if finite.any():
                            lo, hi = np.percentile(dimg[finite], [1, 99])
                            norm = np.clip((dimg - lo) / max(hi - lo, 1e-6), 0, 1)
                            Image.fromarray((norm * 255).astype(np.uint8)).save(out / f"{name}_depth.png")
                            print(f"CAMCHECK: saved {out / f'{name}_depth.png'}  (1-99% 归一化)", flush=True)
        except Exception as e:
            print(f"CAMCHECK: {name} FAILED: {type(e).__name__}: {e}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
