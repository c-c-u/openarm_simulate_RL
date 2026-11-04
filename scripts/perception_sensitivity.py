"""感知精度需求实验：给"盒子位姿"观测注入误差，测策略成功率如何衰减。

用途：回答 sim-to-real / CNN 感知前端方案的核心问题 —— **感知要多准才够用**。
在仿真里直接扰动喂给策略的观测（不改变物理），扫描位置误差、姿态误差与延迟。

被扰动的观测分量（62 维排布；见 openarm_flip_env_cfg.py）：
    36-38 parcel_pos（盒子位置） | 39-42 parcel_quat（姿态, x,y,z,w） | 43-45 label_normal

模式说明：
  * ``posN``   ：位置加 σ=N 毫米的高斯噪声
  * ``rotN``   ：姿态施加 N 度的随机旋转扰动，并**按扰动后的姿态重算** label_normal
                 （模拟 CNN 直接输出六自由度位姿的情形）
  * ``labelN`` ：只把 label_normal 旋转 N 度（模拟单独回归"标签朝向"的情形）
  * ``delayN`` ：盒子位姿改用 N 个策略步之前的值（模拟感知/通信延迟，1 步 = 33 ms）

用法::

    python.sh scripts/perception_sensitivity.py --checkpoint models/openarm_flip_state62_best.pt \
        --num_envs 64 --episodes 2
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
_ISAAC_ROOT = pathlib.Path(os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64"))
for _rel in ("extsDeprecated/omni.isaac.ml_archive/pip_prebundle", "exts/isaacsim.pip.newton/pip_prebundle"):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

parser = argparse.ArgumentParser(description="Perception-accuracy sensitivity sweep for the parcel-flip policy.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="OpenArm-Flip-State-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=2, help="每个扰动模式的回合数。")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--modes",
    type=str,
    default="baseline,pos5,pos10,pos20,pos40,rot5,rot10,rot20,rot40,delay1,delay2,delay4",
    help="逗号分隔的扰动模式列表。",
)
args = parser.parse_known_args()[0]

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": False})
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import openarm_lab  # noqa: F401
from isaaclab.utils.string import string_to_callable
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from openarm_lab.openarm_flip.checkpoint_utils import make_env_cfg, obs_dim_from_checkpoint
from openarm_lab.openarm_flip.openarm_flip_env_cfg import TABLE_POS, TABLE_SIZE, TABLE_TOP

UP_COS = math.cos(math.radians(15))
DROP_TOL = 0.08

# 62 维排布中的盒子位姿分量
POS = slice(36, 39)
QUAT = slice(39, 43)
LABEL = slice(43, 46)


def _tensor_of(obs):
    if hasattr(obs, "keys") and "policy" in obs.keys():
        return obs["policy"]
    return obs


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton 积，四元数顺序 (x, y, z, w)。"""
    ax, ay, az, aw = a.unbind(dim=-1)
    bx, by, bz, bw = b.unbind(dim=-1)
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dim=-1,
    )


def _label_from_quat(q: torch.Tensor) -> torch.Tensor:
    """由姿态算标签面法向（= 旋转 (0,0,1)，即旋转矩阵第三列）。"""
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1
    )


def _rot_perturb(q: torch.Tensor, degrees: float, generator=None) -> torch.Tensor:
    """对姿态施加随机方向、固定角度的旋转扰动 q_delta * q。"""
    n = q.shape[0]
    axis = torch.randn(n, 3, device=q.device, generator=generator)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    half = math.radians(degrees) / 2.0
    s, c = math.sin(half), math.cos(half)
    delta = torch.cat([axis * s, torch.full((n, 1), c, device=q.device)], dim=-1)
    return _quat_mul(delta, q)


class Perturber:
    """按模式扰动观测里的盒子位姿分量（保持几何自洽）。"""

    def __init__(self, mode: str, history_len: int = 8, num_envs: int = 1, device=None):
        self.mode = mode
        self.hist: list[torch.Tensor] = []
        self.history_len = history_len
        # "恒定偏差"模式：偏移量只在初始化时抽一次，之后每步都用同一个 —— 模拟
        # 相机外参/标定误差这类**有偏**误差（比无偏噪声更危险，闭环无法自动纠正）
        self.bias_pos = None
        self.bias_quat_delta = None
        if mode.startswith("bias_pos"):
            sigma = float(mode[8:]) / 1000.0
            d = torch.randn(3, device=device)
            self.bias_pos = (d / d.norm().clamp_min(1e-8) * sigma)
        elif mode.startswith("bias_rot"):
            deg = float(mode[8:])
            axis = torch.randn(3, device=device)
            axis = axis / axis.norm().clamp_min(1e-8)
            half = math.radians(deg) / 2.0
            self.bias_quat_delta = torch.cat(
                [axis * math.sin(half), torch.tensor([math.cos(half)], device=device)]
            ).unsqueeze(0)

    def __call__(self, obs):
        t = _tensor_of(obs).clone()

        # 延迟模式：用 N 步之前的盒子位姿
        if self.mode.startswith("delay"):
            steps = int(self.mode[5:])
            self.hist.append(t[:, 36:46].clone())  # pos+quat 共 10 维
            if len(self.hist) > self.history_len:
                self.hist.pop(0)
            idx = max(0, len(self.hist) - 1 - steps)
            old = self.hist[idx]
            t[:, 36:46] = old
            t[:, LABEL] = _label_from_quat(t[:, QUAT])
            return self._write(obs, t)

        if self.mode == "baseline":
            return obs

        if self.mode.startswith("bias_pos"):
            t[:, POS] = t[:, POS] + self.bias_pos
        elif self.mode.startswith("bias_rot"):
            q = _quat_mul(self.bias_quat_delta.expand(t.shape[0], -1), t[:, QUAT])
            t[:, QUAT] = q
            t[:, LABEL] = _label_from_quat(q)
        elif self.mode.startswith("pos"):
            sigma = float(self.mode[3:]) / 1000.0  # mm -> m
            t[:, POS] = t[:, POS] + sigma * torch.randn_like(t[:, POS])
        elif self.mode.startswith("rot"):
            deg = float(self.mode[3:])
            q = _rot_perturb(t[:, QUAT], deg)
            t[:, QUAT] = q
            t[:, LABEL] = _label_from_quat(q)  # 姿态变了，法向按几何一致性重算
        elif self.mode.startswith("label"):
            deg = float(self.mode[5:])
            lab = t[:, LABEL]
            # 把法向绕随机轴旋转 deg 度（用 Rodrigues 公式，输入是单位向量）
            n = lab.shape[0]
            axis = torch.randn(n, 3, device=lab.device)
            axis = axis - (axis * lab).sum(-1, keepdim=True) * lab  # 取垂直于 lab 的分量
            axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            th = math.radians(deg)
            t[:, LABEL] = (lab * math.cos(th) + torch.cross(axis, lab, dim=-1) * math.sin(th))
        else:
            raise ValueError(f"unknown mode {self.mode}")
        return self._write(obs, t)

    def _write(self, obs, t):
        if hasattr(obs, "keys") and "policy" in obs.keys():
            obs["policy"] = t
            return obs
        return t


def main() -> None:
    import importlib.metadata as metadata

    dim = obs_dim_from_checkpoint(args.checkpoint)
    spec = gym.spec(args.task)
    env_cfg = make_env_cfg(
        string_to_callable(spec.kwargs["env_cfg_entry_point"]), args.checkpoint, enable_cameras=False
    )
    agent_cfg = string_to_callable(spec.kwargs["rsl_rl_cfg_entry_point"])()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, str(metadata.version("rsl-rl-lib")))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
    print(f"[SENS] checkpoint: {args.checkpoint} (obs dim {dim})", flush=True)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    n_envs = unwrapped.num_envs
    max_steps = int(unwrapped.max_episode_length)
    parcel = unwrapped.scene["parcel"]
    origins = unwrapped.scene.env_origins

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    print(f"[SENS] {n_envs} envs x {args.episodes} episodes per mode = {n_envs*args.episodes} 回合/模式", flush=True)
    results: dict[str, tuple[int, int]] = {}

    for mode in modes:
        pert = Perturber(mode, num_envs=n_envs, device=unwrapped.device)
        ok = tot = 0
        for ep in range(args.episodes):
            obs = env.reset()[0]
            pert.hist.clear()
            ever = torch.zeros(n_envs, dtype=torch.bool, device=unwrapped.device)

            def read_state():
                q = parcel.data.root_quat_w.torch
                p = parcel.data.root_pos_w.torch - origins
                qx, qy, _, _ = q.unbind(dim=-1)
                up = 1.0 - 2.0 * (qx * qx + qy * qy)
                ins = ((p[:, 0] - TABLE_POS[0]).abs() < TABLE_SIZE[0] / 2.0 - 0.03) & (
                    (p[:, 1] - TABLE_POS[1]).abs() < TABLE_SIZE[1] / 2.0 - 0.03
                )
                return up, ins & (p[:, 2] > TABLE_TOP - DROP_TOL)

            lu_cur, ot_cur = read_state()
            for step in range(max_steps):
                with torch.no_grad():
                    a = policy(pert(obs))
                obs, _, done, _ = env.step(a)
                lu_new, ot_new = read_state()
                d = done if done.dtype == torch.bool else done.bool()
                lu = torch.where(d, lu_cur, lu_new)
                ot = torch.where(d, ot_cur, ot_new)
                ever |= (lu > UP_COS) & ot
                lu_cur, ot_cur = lu_new, ot_new
            ok += int(ever.sum().item())
            tot += n_envs
        results[mode] = (ok, tot)
        print(f"  {mode:10s} {ok:4d}/{tot:<4d} = {100.0*ok/tot:5.1f}%", flush=True)

    base_rate = 100.0 * results["baseline"][0] / results["baseline"][1]
    print("\n" + "=" * 74)
    print(f"感知精度需求（基线 {base_rate:.1f}%，{results['baseline'][1]} 回合/模式）")
    print("=" * 74)
    print(f"  {'模式':10s} {'成功率':>8s} {'相对基线':>10s}   说明")
    desc = {
        "baseline": "无扰动",
        "pos5": "位置误差 σ=5 mm", "pos10": "位置误差 σ=10 mm",
        "pos20": "位置误差 σ=20 mm", "pos40": "位置误差 σ=40 mm",
        "rot5": "姿态误差 5°", "rot10": "姿态误差 10°",
        "rot20": "姿态误差 20°", "rot40": "姿态误差 40°",
        "delay1": "延迟 1 步 (33 ms)", "delay2": "延迟 2 步 (67 ms)", "delay4": "延迟 4 步 (133 ms)",
        "delay6": "延迟 6 步 (200 ms)",
        "pos80": "位置误差 σ=80 mm", "rot30": "姿态误差 30°", "rot60": "姿态误差 60°",
        "bias_pos10": "位置常偏 10 mm（标定误差）", "bias_pos20": "位置常偏 20 mm（标定误差）",
        "bias_pos40": "位置常偏 40 mm（标定误差）",
        "bias_rot10": "姿态常偏 10°（标定误差）", "bias_rot20": "姿态常偏 20°",
        "bias_rot40": "姿态常偏 40°",
    }
    for mode, (ok, tot) in results.items():
        rate = 100.0 * ok / tot
        delta = rate - base_rate
        se = math.sqrt(max(rate, 1e-6) * (100 - rate) / tot + base_rate * (100 - base_rate) / results["baseline"][1])
        flag = "  ← 显著下降" if delta < -2 * se else ""
        print(f"  {mode:10s} {rate:7.1f}% {delta:+9.1f}pp   {desc.get(mode, ''):22s}{flag}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
