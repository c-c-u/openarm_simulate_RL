# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""消融实验：盒子姿态的**面内转角**（绕标签轴的自转）对策略有多重要。

为什么要做这个实验
------------------
现有 62 维状态观测里既有 ``parcel_quat(4)`` 又有 ``label_normal(3)``，两者**信息冗余**
（label_normal 就是 R(q)·(0,0,1)）。若要把策略接到纯视觉前端上，CNN 能自然给出的只是
**标签法向**（3 维）—— 差的那一维自由度就是"绕标签轴的自转"（下称**面内转角 twist**）。

而且盒子 0.09×0.09 的正方形底面 + 标签纹理不参与物理 ⇒ 面内转角有 **90° 四重对称**：
直接让 CNN 回归四元数，四个等价解会让网络摇摆。所以必须先回答：

* **策略到底用不用这一维？** 不用 ⇒ 视觉前端只需供"位置 + 标签法向"6 维，方案最简。
* 用 ⇒ 必须靠标签纹理/角点打破对称，或干脆重训一个不依赖它的策略。

做法
----
把观测里的 ``parcel_quat`` 替换成"**保标签法向、改面内转角**"的四元数（``label_normal``
那一维不动，保持与 quat 自洽），其余观测不变，然后跑完整回合统计成功率：

============ ==========================================================
mode         含义
============ ==========================================================
baseline     原样
twist_zero   面内转角归零（= 视觉只给标签法向时最自然的假设）
twist_90     面内转角取 90° 的整数倍（**物理上等价**的另一组代表元）
twist_180    面内转角固定 +180°（确定性地给错）
twist_random 面内转角每步随机（测"用了多少"）
============ ==========================================================

**配对设计**：每个回合前 ``env.seed(seed + episode_idx)`` 重设随机种子，于是各模式拿到
**完全相同的初始条件**（否则观测噪声会消耗 RNG、让后续回合的条件漂移）。因此可以报告
不一致对（baseline 成功 / 该模式失败）的数量，比只比总成功率灵敏一个量级。

用法::

    $ISAAC_ROOT/python.sh scripts/twist_ablation.py --num_envs 128 --episodes 10
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

_ISAAC_ROOT = pathlib.Path(
    os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64")
)
for _rel in ("extsDeprecated/omni.isaac.ml_archive/pip_prebundle", "exts/isaacsim.pip.newton/pip_prebundle"):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

_REPO = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_CKPT = _REPO / "models" / "openarm_flip_state62_best.pt"

parser = argparse.ArgumentParser(description="面内转角（twist）消融")
parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CKPT))
parser.add_argument("--task", type=str, default="OpenArm-Flip-State-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--modes", type=str, default="baseline,twist_zero,twist_90,twist_180,twist_random")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": False})
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import openarm_lab  # noqa: F401,E402
from isaaclab.utils.string import string_to_callable  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from openarm_lab.openarm_flip.checkpoint_utils import make_env_cfg, obs_dim_from_checkpoint  # noqa: E402
from openarm_lab.openarm_flip.openarm_flip_env_cfg import (  # noqa: E402
    TABLE_POS,
    TABLE_SIZE,
    TABLE_TOP,
)

UP_COS = math.cos(math.radians(15))
DROP_TOL = 0.08
# 62 维排布：36-38 parcel_pos | 39-42 parcel_quat | 43-45 label_normal
QUAT_SL = slice(39, 43)
LABEL_SL = slice(43, 46)


# ---------------------------------------------------------------------------
# 四元数工具（全部 xyzw）
# ---------------------------------------------------------------------------
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """复合：R(a) @ R(b)。"""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def quat_z_to(n: torch.Tensor) -> torch.Tensor:
    """把世界 +Z 转到 ``n`` 的**最短弧**四元数（swing，面内转角恒为 0）。"""
    nx, ny, nz = n.unbind(-1)
    q = torch.stack([-ny, nx, torch.zeros_like(nx), 1.0 + nz], dim=-1)
    norm = q.norm(dim=-1, keepdim=True)
    degen = norm.squeeze(-1) < 1e-6  # n ≈ -Z：最短弧不唯一，约定为绕 X 轴 180°
    q = q / norm.clamp_min(1e-8)
    if degen.any():
        q = torch.where(
            degen.unsqueeze(-1),
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype).expand_as(q),
            q,
        )
    return q


def quat_about_z(theta: torch.Tensor) -> torch.Tensor:
    """绕局部 +Z 转 ``theta`` 的四元数，(N,) -> (N,4)。"""
    half = 0.5 * theta
    return torch.stack([torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)], dim=-1)


def label_normal_of(q: torch.Tensor) -> torch.Tensor:
    """R(q) @ (0,0,1) = 旋转矩阵第三列。"""
    x, y, z, w = q.unbind(-1)
    return torch.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=-1)


def ablate_twist(obs, mode: str, gen: torch.Generator | None = None):
    """按模式替换观测里的 parcel_quat，保持标签法向不变。"""
    if mode == "baseline":
        return obs
    t = obs["policy"] if (hasattr(obs, "keys") and "policy" in obs.keys()) else obs
    t = t.clone()
    q = t[:, QUAT_SL]
    n = label_normal_of(q)                 # 原标签法向（保持不变）
    q_swing = quat_z_to(n)                 # 面内转角归零的代表元
    if mode == "twist_zero":
        q_new = q_swing
    elif mode == "twist_90":
        # 物理等价的另一组代表元：绕标签轴多转 k*90°（盒子底面是正方形）
        k = torch.randint(0, 4, (q.shape[0],), device=q.device, generator=gen).float()
        q_new = quat_mul(q_swing, quat_about_z(k * (math.pi / 2.0)))
    elif mode == "twist_180":
        q_new = quat_mul(q_swing, quat_about_z(torch.full((q.shape[0],), math.pi, device=q.device)))
    elif mode == "twist_random":
        th = torch.rand(q.shape[0], device=q.device, generator=gen) * (2.0 * math.pi) - math.pi
        q_new = quat_mul(q_swing, quat_about_z(th))
    else:
        raise ValueError(mode)
    # 自检：替换后标签法向必须与原来一致（否则就不是"只改面内转角"了）
    t[:, QUAT_SL] = q_new
    t[:, LABEL_SL] = n
    if hasattr(obs, "keys") and "policy" in obs.keys():
        obs["policy"] = t
        return obs
    return t


def main() -> None:
    import importlib.metadata as metadata

    dim = obs_dim_from_checkpoint(args.checkpoint)
    if dim != 62:
        raise SystemExit(f"[TWIST] 需要 62 维 checkpoint（当前 {dim}）；索引排布按 62 维定义")

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
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    parcel = unwrapped.scene["parcel"]
    origins = unwrapped.scene.env_origins
    device = unwrapped.device
    n_envs = unwrapped.num_envs
    max_steps = int(unwrapped.max_episode_length)
    dt = unwrapped.step_dt

    # 先自检一遍消融是否真的保住了标签法向
    obs0, _ = env.reset()
    q0 = (obs0["policy"] if hasattr(obs0, "keys") else obs0)[:, QUAT_SL]
    for m in ("twist_zero", "twist_180", "twist_random"):
        o = ablate_twist(obs0, m)
        qq = (o["policy"] if hasattr(o, "keys") else o)[:, QUAT_SL]
        err = (label_normal_of(qq) - label_normal_of(q0)).abs().max().item()
        print(f"[TWIST] 自检 {m:12s}: 标签法向最大偏差 {err:.2e}  {'OK' if err < 1e-5 else '★ 异常'}", flush=True)

    print(f"[TWIST] {args.num_envs} envs × {args.episodes} 回合 × {max_steps} 步 | "
          f"模式 {args.modes} | 配对种子 {args.seed}", flush=True)

    def read_state():
        """(label_up, on_table) —— 与 eval_breakdown 同判据。"""
        q = parcel.data.root_quat_w.torch
        p = parcel.data.root_pos_w.torch - origins
        up = 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)
        ins = ((p[:, 0] - TABLE_POS[0]).abs() < TABLE_SIZE[0] / 2.0 - 0.03) & (
            (p[:, 1] - TABLE_POS[1]).abs() < TABLE_SIZE[1] / 2.0 - 0.03
        )
        return up, ins & (p[:, 2] > TABLE_TOP - DROP_TOL)

    results: dict[str, dict[str, torch.Tensor]] = {}
    init_is_down: torch.Tensor | None = None

    for mode in args.modes.split(","):
        mode = mode.strip()
        ever = torch.zeros(args.episodes, n_envs, dtype=torch.bool)
        final = torch.zeros(args.episodes, n_envs, dtype=torch.bool)
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed)
        for ep in range(args.episodes):
            # 配对设计：每回合前重设种子 -> 各模式拿到完全相同的初始条件
            unwrapped.seed(args.seed + ep)
            obs, _ = env.reset()
            if ep == 0:
                _q = parcel.data.root_quat_w.torch.clone()
                down = (1.0 - 2.0 * (_q[:, 0] ** 2 + _q[:, 1] ** 2)) < -0.5
                if init_is_down is None:
                    init_is_down = down.clone()
            lu_cur, ot_cur = read_state()
            ever_ep = torch.zeros(n_envs, dtype=torch.bool, device=device)
            last_ok = torch.zeros(n_envs, dtype=torch.bool, device=device)
            for _ in range(max_steps):
                with torch.no_grad():
                    actions = policy(ablate_twist(obs, mode, gen))
                obs, _, done, _ = env.step(actions)
                lu_new, ot_new = read_state()
                # 本步终止的 env 已被自动 reset，读到的是新回合初值 -> 用上一轮的值覆盖
                d = done if done.dtype == torch.bool else done.bool()
                lu_eff = torch.where(d, lu_cur, lu_new)
                ot_eff = torch.where(d, ot_cur, ot_new)
                last_ok = (lu_eff > UP_COS) & ot_eff
                ever_ep |= last_ok
                lu_cur, ot_cur = lu_new, ot_new
            ever[ep] = ever_ep.cpu()
            final[ep] = last_ok.cpu()
        results[mode] = {"ever": ever, "final": final}
        print(f"  [{mode}] 宽松(曾达标) {100 * ever.float().mean():5.1f}% | "
              f"严格(末帧仍达标) {100 * final.float().mean():5.1f}%", flush=True)

    env.close()

    # ------------------------------ 汇总（配对比较） ------------------------------
    base = results["baseline"]

    def wilson(k: int, n: int, z: float = 1.96):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return max(0.0, c - h), min(1.0, c + h)

    modes = [m.strip() for m in args.modes.split(",")]
    print("\n" + "=" * 96)
    print(f"面内转角(twist)消融 | checkpoint {pathlib.Path(args.checkpoint).name} | "
          f"{args.episodes}×{n_envs} = {args.episodes * n_envs} 个配对回合")
    print("=" * 96)
    print(f"{'模式':<14}{'指标':<6}{'成功率':>8}{'95% CI':>16}{'vs 基线':>10}"
          f"{'劣化对 b':>10}{'改善对 c':>10}   配对判读")
    for mode in modes:
        r = results[mode]
        for metric, mname in (("ever", "宽松"), ("final", "严格")):
            k = int(r[metric].sum())
            tot = r[metric].numel()
            lo, hi = wilson(k, tot)
            b = int((base[metric] & ~r[metric]).sum())
            c = int((~base[metric] & r[metric]).sum())
            drop = 100.0 * (base[metric].float().mean() - r[metric].float().mean())
            if mode == "baseline":
                verdict = "← 参照"
            elif abs(b - c) > 2.0 * math.sqrt(max(b + c, 1)):
                verdict = f"★ 该维被用到（{drop:+.1f}pp，显著）"
            else:
                verdict = f"该维基本没用到（{drop:+.1f}pp，不显著）"
            print(f"{mode:<14}{mname:<6}{100 * k / tot:7.1f}%{f'{100 * lo:.1f}-{100 * hi:.1f}%':>16}"
                  f"{drop:+9.1f}pp{b:10d}{c:10d}   {verdict}")
    print("=" * 96)
    print("判读：b = 基线成功但该模式失败的回合数，c = 反过来；主指标是『宽松』（过程中曾达标）。")
    print("      只有 b 明显大于 c（|b-c| > 2√(b+c)）才算该维真被策略用到。")

    if init_is_down is not None:
        print("\n【按初始姿态分组（宽松成功率）】")
        sw = ~init_is_down
        for mode in args.modes.split(","):
            mode = mode.strip()
            e = results[mode]["ever"]
            ds = e[:, init_is_down.cpu()].float().mean().item()
            ss = e[:, sw.cpu()].float().mean().item()
            print(f"  {mode:<14} 标签朝下(180°, 难) {100 * ds:5.1f}%   侧面(90°, 易) {100 * ss:5.1f}%")


if __name__ == "__main__":
    main()
    simulation_app.close()
