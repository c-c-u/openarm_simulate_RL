"""验证观测向量的实际排布（拼接顺序）是否与代码里的假设一致。

改动观测项（增删 ObsTerm / 调整顺序）后务必跑一次：排布错了不会报错，只会让
策略静默地"看错数据"、性能莫名下降。本脚本用物理恒等关系校验，而非肉眼比对。

用法::

    python.sh scripts/check_obs_layout.py --use_parcel_vel 0   # 62 维（默认观测）
    python.sh scripts/check_obs_layout.py --use_parcel_vel 1   # 68 维（旧模型）


62 维假设排布：
   0-17 joint_pos | 18-35 joint_vel | 36-38 parcel_pos | 39-42 parcel_quat
  43-45 label_normal | 46-61 last_action
68 维假设排布：
   0-17 joint_pos | 18-35 joint_vel | 36-38 parcel_pos | 39-42 parcel_quat
  43-45 parcel_lin_vel | 46-48 parcel_ang_vel | 49-51 label_normal | 52-67 last_action

校验项：
  * 标签法向分量应满足 obs[49:52].norm()≈1（68 维）/ obs[43:46].norm()≈1（62 维），
    且其 z 分量应等于由盒子四元数算出的 label_up = 1-2(x²+y²)；
  * 盒子位置分量应等于 env-local 的 root_pos_w；
  * last_action 段应与上一步实际下发的动作一致；
  * 四元数段应满足 norm≈1。
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
_ISAAC_ROOT = pathlib.Path(os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64"))
for _rel in ("extsDeprecated/omni.isaac.ml_archive/pip_prebundle", "exts/isaacsim.pip.newton/pip_prebundle"):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

parser = argparse.ArgumentParser()
parser.add_argument("--use_parcel_vel", type=int, default=0, help="1=68 维观测，0=62 维")
parser.add_argument("--steps", type=int, default=30)
args = parser.parse_args()

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": False})
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import openarm_lab  # noqa: F401
from isaaclab.utils.string import string_to_callable

spec = gym.spec("OpenArm-Flip-State-v0")
env_cfg = string_to_callable(spec.kwargs["env_cfg_entry_point"])(enable_cameras=False)
env_cfg.scene.num_envs = 4
env_cfg.sim.device = args.device if hasattr(args, "device") else "cuda:0"
env_cfg.use_parcel_vel = bool(args.use_parcel_vel)
env = gym.make("OpenArm-Flip-State-v0", cfg=env_cfg)
unwrapped = env.unwrapped
n = env_cfg.scene.num_envs

V = args.use_parcel_vel
dim = 68 if V else 62
IDX = dict(
    joint_pos=(0, 18),
    joint_vel=(18, 36),
    parcel_pos=(36, 39),
    parcel_quat=(39, 43),
    label_normal=(49, 52) if V else (43, 46),
    last_action=(52, 68) if V else (46, 62),
    parcel_lin_vel=(43, 46) if V else None,
)
print(f"\n[LAYOUT] 检查 {dim} 维观测（use_parcel_vel={V}）", flush=True)

obs = env.reset()[0]
obs_t = obs["policy"] if hasattr(obs, "keys") else obs
print(f"[LAYOUT] 实测 obs 维度 = {obs_t.shape[1]}（期望 {dim}）", flush=True)

parcel = unwrapped.scene["parcel"]
origins = unwrapped.scene.env_origins
prev_action = torch.zeros(n, 16, device=obs_t.device)

max_err = {k: 0.0 for k in ("label_norm", "label_z_eq_label_up", "parcel_pos", "quat_norm", "last_action")}
for step in range(args.steps):
    a = torch.randn(n, 16, device=obs_t.device).clamp(-1, 1)
    obs, _, _, _, _ = env.step(a)
    obs_t = obs["policy"] if hasattr(obs, "keys") else obs

    lo, hi = IDX["label_normal"]
    lab = obs_t[:, lo:hi]
    q = parcel.data.root_quat_w.torch
    qx, qy, _, _ = q.unbind(dim=-1)
    label_up = 1.0 - 2.0 * (qx * qx + qy * qy)

    max_err["label_norm"] = max(max_err["label_norm"], (lab.norm(dim=1) - 1).abs().max().item())
    max_err["label_z_eq_label_up"] = max(max_err["label_z_eq_label_up"], (lab[:, 2] - label_up).abs().max().item())

    plo, phi = IDX["parcel_pos"]
    pos_env = parcel.data.root_pos_w.torch - origins
    max_err["parcel_pos"] = max(max_err["parcel_pos"], (obs_t[:, plo:phi] - pos_env).abs().max().item())

    qlo, qhi = IDX["parcel_quat"]
    max_err["quat_norm"] = max(max_err["quat_norm"], (obs_t[:, qlo:qhi].norm(dim=1) - 1).abs().max().item())

    alo, ahi = IDX["last_action"]
    # 注意：上一步动作在观测里是"上一步的策略输出"，这里用随机动作近似比对会因
    # 动作缩放/裁剪而不精确，只做粗略的一致性检查（相关性）
    if step > 0:
        corr = torch.corrcoef(torch.stack([obs_t[:, alo:ahi].flatten(), prev_action.flatten()]))[0, 1]
        max_err["last_action"] = max(max_err["last_action"], float(1 - corr))
    prev_action = a.clone()

print("\n[LAYOUT] 恒等关系校验（误差越小越好）:", flush=True)
print(f"  label_normal 模长 - 1        : {max_err['label_norm']:.4e}   (应≈0)", flush=True)
print(f"  label_normal[2] - label_up   : {max_err['label_z_eq_label_up']:.4e}   (应≈0，决定性判据)", flush=True)
print(f"  parcel_pos - env-local 位置  : {max_err['parcel_pos']:.4e}   (应≈0)", flush=True)
print(f"  parcel_quat 模长 - 1         : {max_err['quat_norm']:.4e}   (应≈0)", flush=True)
print(f"  last_action 相关系数偏差     : {max_err['last_action']:.4e}   (应≈0)", flush=True)
ok = max_err["label_z_eq_label_up"] < 1e-5 and max_err["parcel_pos"] < 1e-4
print(f"\n[LAYOUT] 结论: {'排布与假设一致 ✓' if ok else '排布与假设不一致 ✗'}", flush=True)
env.close()
