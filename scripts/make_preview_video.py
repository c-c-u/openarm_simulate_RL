#!/usr/bin/env python3
# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""把 ``collect_vision_data.py`` 采到的一帧帧图像合成**预览视频**（带动画叠加）。

为什么不是 mp4
--------------
本机**没有 ffmpeg**（系统、conda、Isaac Sim 自带的 pip_prebundle 里都没有可用二进制），
pip 的 ``imageio-ffmpeg`` 也未安装，所以写不了 mp4。PIL 12 原生支持动画 **WebP** 与
**GIF**，两者浏览器都能播：

* ``preview_<cam>.webp`` —— 首选，24 位色 / 有损压缩，体积小、画质好
* ``preview_<cam>.gif``  —— 兜底，任何看图工具都能播，但只有 256 色

（想要真正的 mp4 可以 ``kit/python/bin/python3 -m pip install imageio-ffmpeg``，
它会带一个静态 ffmpeg 二进制；本脚本不擅自装。）

叠加内容
--------
* 顶部信息条：帧号 / 回合内步数 / ``label_up`` / 盒子轮廓可见度
* 盒子 8 角点的真值投影（绿框）—— 直接验证"图像与位姿真值对齐"，不是装饰
* 达标（label_up > cos15°）后顶部显示 **SUCCESS**

用法::

    python3 scripts/make_preview_video.py <run_dir>                 # 宿主 python3 即可
    python3 scripts/make_preview_video.py <run_dir> --fps 30        # 实时（策略 30 Hz）
    python3 scripts/make_preview_video.py <run_dir> --no_overlay    # 原始画面，无叠加
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

UP_COS = math.cos(math.radians(15))
SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def _font(size: int = 15):
    """尽量用 DejaVuSans，取不到就退回 PIL 自带点阵字体。"""
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if pathlib.Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def rot_of_quat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def project(points_w, cam_pos, cam_quat_ros, K):
    p_cam = (np.asarray(points_w) - np.asarray(cam_pos)) @ rot_of_quat(cam_quat_ros)
    z = p_cam[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    return np.stack([K[0, 0] * p_cam[:, 0] / safe + K[0, 2],
                     K[1, 1] * p_cam[:, 1] / safe + K[1, 2]], axis=-1)


def build(run_dir: pathlib.Path, cam: str, fps: float, scale: float, overlay: bool,
          formats: list[str], label: str = "", episode: int | None = None,
          max_frames: int | None = None) -> list[pathlib.Path]:
    info = json.loads((run_dir / "dataset_info.json").read_text(encoding="utf-8"))
    f = np.load(run_dir / "frames.npz")
    img_dir = run_dir / "images" / cam
    if not img_dir.exists():
        raise SystemExit(f"[VIDEO] 找不到相机目录 {img_dir}")
    # 可按回合过滤：多回合数据集常常只想要某一回合的连续片段当预览
    sel = np.arange(len(f["frame_index"]))
    if episode is not None:
        if "episode_index" not in f:
            raise SystemExit("[VIDEO] 该数据集没有 episode_index 字段，无法按回合过滤")
        sel = sel[f["episode_index"] == episode]
        if sel.size == 0:
            raise SystemExit(f"[VIDEO] 回合 {episode} 没有帧")

    n = len(sel)
    half = np.array(info.get("parcel_size_m", [0.09, 0.09, 0.07])) / 2.0
    corners_local = SIGNS * half
    pos_w = f["parcel_pos_w"] if "parcel_pos_w" in f else (
        f["parcel_pos_env"] + f["env_origin"] if "env_origin" in f else f["parcel_pos_env"])
    has_pose = all(k in f for k in (f"{cam}_pos_w", f"{cam}_quat_w_ros", f"{cam}_K"))
    font, font_b = _font(15), _font(19)

    frames: list[Image.Image] = []
    for i in sel[:max_frames] if max_frames else sel:
        fno = int(f["frame_index"][i])
        im = Image.open(img_dir / f"{fno:06d}.png").convert("RGB")
        if scale != 1.0:
            im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                           Image.LANCZOS)
        if overlay:
            d = ImageDraw.Draw(im)
            if has_pose:
                R = rot_of_quat(f["parcel_quat_w"][i])
                pw = pos_w[i] + corners_local @ R.T
                uv = project(pw, f[f"{cam}_pos_w"][i], f[f"{cam}_quat_w_ros"][i], f[f"{cam}_K"][i]) * scale
                pts = [(float(u), float(v)) for u, v in uv]
                for a in range(8):
                    for b in range(a + 1, 8):
                        if bin(a ^ b).count("1") == 1:
                            d.line([pts[a], pts[b]], fill=(0, 255, 0), width=2)
            up = float(f["label_up"][i])
            step = int(f["step_in_episode"][i]) if "step_in_episode" in f else fno
            vis = float(f[f"{cam}_parcel_visible_frac"][i]) if f"{cam}_parcel_visible_frac" in f else float("nan")
            d.rectangle([0, 0, im.width, 24], fill=(0, 0, 0))
            d.text((6, 4), f"frame {fno:3d} | step {step:3d} | label_up {up:+.2f} | "
                           f"vis {100 * vis:3.0f}%" if np.isfinite(vis) else
                           f"frame {fno:3d} | step {step:3d} | label_up {up:+.2f}",
                   fill=(255, 255, 0), font=font)
            if up > UP_COS:
                d.rectangle([im.width - 118, 0, im.width, 24], fill=(0, 110, 0))
                d.text((im.width - 110, 3), "SUCCESS", fill=(255, 255, 255), font=font_b)
        frames.append(im)

    out_paths = []
    dur = max(1, int(round(1000.0 / fps)))
    tag = f"_{label}" if label else ""
    if "webp" in formats:
        p = run_dir / f"preview_{cam}{tag}.webp"
        frames[0].save(p, save_all=True, append_images=frames[1:], duration=dur, loop=0,
                       quality=88, method=4)
        out_paths.append(p)
    if "gif" in formats:
        p = run_dir / f"preview_{cam}{tag}.gif"
        # GIF 只有 256 色，缩一档更耐看
        gif_frames = [im if im.width <= 480 else im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
                      for im in frames]
        gif_frames[0].save(p, save_all=True, append_images=gif_frames[1:], duration=dur, loop=0,
                           optimize=True)
        out_paths.append(p)
    print(f"[VIDEO] {n} 帧 @ {fps:g} fps ({n / fps:.1f} s, {'慢放' if fps < 30 else '实时'}，"
          f"策略原始 30 Hz)")
    for p in out_paths:
        print(f"[VIDEO] 已写出 {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    return out_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="把数据集合成预览视频（WebP/GIF）")
    ap.add_argument("run_dir", type=str)
    ap.add_argument("--cam", type=str, default="front")
    ap.add_argument("--fps", type=float, default=15.0, help="默认 15 = 相对策略 30 Hz 慢放 2 倍")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no_overlay", action="store_true", help="不加信息条与真值框")
    ap.add_argument("--formats", type=str, default="webp,gif")
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--episode", type=int, default=None, help="只取某个回合（0 起）")
    ap.add_argument("--max_frames", type=int, default=None, help="最多取多少帧")
    a = ap.parse_args()
    build(pathlib.Path(a.run_dir).expanduser().resolve(), a.cam, a.fps, a.scale,
          not a.no_overlay, [x.strip() for x in a.formats.split(",") if x.strip()], a.label,
          a.episode, a.max_frames)


if __name__ == "__main__":
    main()
