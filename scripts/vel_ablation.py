"""观测消融实验：回答"观测里各分量（尤其速度维）对策略有多重要"。

实验 B —— 观测敏感度分析：对 actor 网络求 |∂a/∂obs|，再乘以该维度的标准差
（即"该维变化 1 个标准差会引起多大动作变化"），按观测分量聚合排名。

实验 A —— 输入消融评估：在评估时对指定观测维做"置零 / 加噪"（模拟真机拿不到
该信息或估计不准），用与 eval_breakdown 相同的成功判据统计成功率，看掉多少。

观测 68 维的排布（StateObservationsCfg）：
    0-17  joint_pos(18) | 18-35 joint_vel(18) | 36-38 parcel_pos(3)
   39-42  parcel_quat(4) | 43-45 parcel_lin_vel(3) | 46-48 parcel_ang_vel(3)
   49-51  label_normal(3) | 52-67 last_action(16)

用法:
    python.sh scripts/vel_ablation.py --checkpoint <model_20000.pt> --num_envs 64 --episodes 3
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
for _rel in (
    "extsDeprecated/omni.isaac.ml_archive/pip_prebundle",
    "exts/isaacsim.pip.newton/pip_prebundle",
):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="OpenArm-Flip-State-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=3, help="每个消融模式的回合数。")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--sens_samples", type=int, default=10000, help="敏感度分析用的观测样本数。")
parser.add_argument("--skip_sens", action="store_true")
parser.add_argument("--skip_ablation", action="store_true")
parser.add_argument(
    "--modes",
    type=str,
    default="baseline,zero_parcel_vel,mean_parcel_vel,noisy_parcel_vel,zero_joint_vel,zero_all_vel",
    help="逗号分隔的消融模式：baseline / zero_parcel_vel / mean_parcel_vel / noisy_parcel_vel "
    "/ zero_joint_vel / zero_all_vel",
)
args = parser.parse_args()

# 观测归一化器的统计量（在 main() 里从 checkpoint 读入：
# 仿真里给某维"填常数 c"时，网络实际看到的是 (c - mean)/std ——
# 填 0 会落在分布外，填 mean 则恰好是中性值 0。两种做法行为不同，需要实测比较。）
NORM_MEAN: torch.Tensor | None = None
NORM_STD: torch.Tensor | None = None

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": False})
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import openarm_lab  # noqa: F401
from isaaclab.utils.string import string_to_callable
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from openarm_lab.openarm_flip.openarm_flip_env_cfg import TABLE_POS, TABLE_SIZE, TABLE_TOP

UP_COS = math.cos(math.radians(15))
DROP_TOL = 0.08

# 观测分量 -> (起始索引, 结束索引, 名称)
COMPONENTS = [
    (0, 18, "joint_pos     (关节位置)"),
    (18, 36, "joint_vel     (关节速度)"),
    (36, 39, "parcel_pos    (盒子位置)"),
    (39, 43, "parcel_quat   (盒子姿态)"),
    (43, 46, "parcel_lin_vel(盒子线速度)"),
    (46, 49, "parcel_ang_vel(盒子角速度)"),
    (49, 52, "label_normal  (标签法向)"),
    (52, 68, "last_action   (上一步动作)"),
]
PARCEL_VEL = (43, 49)
JOINT_VEL = (18, 36)


def _tensor_of(obs):
    """取出观测的底层 tensor（rsl-rl 5.x 的观测可能是 TensorDict 包装）。"""
    if hasattr(obs, "keys") and "policy" in obs.keys():
        return obs["policy"]
    return obs


def _with_tensor(obs, t):
    """把修改后的观测 tensor 写回（保持 TensorDict 结构，若原本是的话）。"""
    if hasattr(obs, "keys") and "policy" in obs.keys():
        obs["policy"] = t
        return obs
    return t


def apply_ablation(obs, mode: str):
    """按模式扰动观测（不改变环境状态，只改喂给策略的输入）。

    注意：这里的"置零/填常数"作用在**原始观测值**上；策略内部还会做
    (x - mean)/std 归一化，因此填 0 与填 mean 对网络而言并不等价
    （后者归一化后恰为 0，是训练分布的中心）。
    """
    if mode == "baseline":
        return obs
    t = _tensor_of(obs).clone()
    if mode == "zero_parcel_vel":
        t[:, PARCEL_VEL[0]:PARCEL_VEL[1]] = 0.0
    elif mode == "mean_parcel_vel":
        # 填训练分布的均值（真机拿不到速度时的"中性填充"）
        t[:, PARCEL_VEL[0]:PARCEL_VEL[1]] = NORM_MEAN[PARCEL_VEL[0]:PARCEL_VEL[1]].to(t.device)
    elif mode == "noisy_parcel_vel":
        sl = slice(PARCEL_VEL[0], PARCEL_VEL[1])
        # 真机速度估计的误差：按该维自身标准差的比例加噪（30%）
        std = t[:, sl].std(dim=0, keepdim=True).clamp_min(1e-3)
        t[:, sl] = t[:, sl] + 0.30 * std * torch.randn_like(t[:, sl])
    elif mode == "zero_joint_vel":
        t[:, JOINT_VEL[0]:JOINT_VEL[1]] = 0.0
    elif mode == "zero_all_vel":
        t[:, PARCEL_VEL[0]:PARCEL_VEL[1]] = 0.0
        t[:, JOINT_VEL[0]:JOINT_VEL[1]] = 0.0
    else:
        raise ValueError(mode)
    return _with_tensor(obs, t)


def main() -> None:
    import importlib.metadata as metadata

    global NORM_MEAN, NORM_STD
    _ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    NORM_MEAN = _ckpt["actor_state_dict"]["obs_normalizer._mean"].flatten()
    NORM_STD = _ckpt["actor_state_dict"]["obs_normalizer._std"].flatten()

    spec = gym.spec(args.task)
    env_cfg = string_to_callable(spec.kwargs["env_cfg_entry_point"])(enable_cameras=False)
    # 本脚本按 68 维排布（含盒子速度）做消融，需确认 checkpoint 与观测排布一致
    from openarm_lab.openarm_flip.checkpoint_utils import obs_dim_from_checkpoint

    _dim = obs_dim_from_checkpoint(args.checkpoint)
    if _dim != 68:
        raise SystemExit(
            f"[ABL] 该 checkpoint 观测维度为 {_dim}，不是 68 维（含盒子速度）。\n"
            "      本脚本的索引排布（PARCEL_VEL/JOINT_VEL）按 68 维定义；\n"
            "      若要看 62 维模型的消融，请改用其他模式或先重训。"
        )
    env_cfg.use_parcel_vel = True  # 68 维模型需要含盒子速度的观测排布
    agent_cfg = string_to_callable(spec.kwargs["rsl_rl_cfg_entry_point"])()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, str(metadata.version("rsl-rl-lib")))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
    print(f"[ABL] loading {args.checkpoint}", flush=True)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    max_steps = int(unwrapped.max_episode_length)
    dt = unwrapped.step_dt
    n_envs = unwrapped.num_envs

    # 直接用底层网络前向（纯 tensor 进出），绕过 TensorDict 包装以便求梯度
    actor = runner.alg.get_policy()
    normalizer = getattr(actor, "obs_normalizer", None)
    distribution = getattr(actor, "distribution", None)

    def actor_mean(o: torch.Tensor) -> torch.Tensor:
        """策略的确定性输出（分布均值）——与评估时用的前向一致。"""
        x = normalizer(o) if normalizer is not None else o
        y = actor.mlp(x)
        return distribution.deterministic_output(y) if distribution is not None else y

    # ==================================================================
    # 实验 B：观测敏感度（|∂a/∂obs| × std）
    # ==================================================================
    if not args.skip_sens:
        print("\n[B] 采集观测样本（策略在环境中运行，覆盖翻转全过程）...", flush=True)
        samples: list[torch.Tensor] = []
        obs = env.reset()[0]
        collected = 0
        ep = 0
        while collected < args.sens_samples and ep < 8:
            for step in range(max_steps):
                with torch.no_grad():
                    a = policy(obs)
                obs, _, done, _ = env.step(a)
                if step < 100:  # 只取回合前 100 步（翻转过程最具代表性）
                    o = _tensor_of(obs)
                    d = done if done.dtype == torch.bool else done.bool()
                    fresh = o[~d] if d.any() else o  # 丢掉本步已 reset 的环境
                    if fresh.numel() > 0:
                        samples.append(fresh.detach().clone())
                        collected += fresh.shape[0]
                if collected >= args.sens_samples:
                    break
            ep += 1
        O = torch.cat(samples, dim=0)[: args.sens_samples]  # (M, 68)
        print(f"[B] 样本形状 {tuple(O.shape)}；计算 |∂a/∂obs| ...", flush=True)

        n_act = actor_mean(O[:2]).shape[1]
        o = O.clone().requires_grad_(True)
        out = actor_mean(o)  # (M, n_act)
        grads = torch.zeros_like(O)
        for j in range(n_act):
            g = torch.autograd.grad(out[:, j].sum(), o, retain_graph=(j < n_act - 1))[0]
            grads += g.detach()
        grads /= n_act
        std = O.std(dim=0)
        sens = grads.abs().mean(dim=0) * std  # 每维敏感度
        total = sens.sum().item()

        print("\n" + "=" * 84)
        print("[B] 观测敏感度分析（|∂a/∂obs| × 该维标准差，越大表示策略越依赖）")
        print("=" * 84)
        print(f"  {'分量':30s} {'维数':>4s} {'该分量平均敏感度':>16s} {'该分量内最大':>12s} {'占总量比例':>10s}")
        print("  " + "-" * 80)
        for lo, hi, name in COMPONENTS:
            seg = sens[lo:hi]
            share = 100.0 * seg.sum().item() / total
            print(f"  {name:30s} {hi-lo:4d} {seg.mean().item():16.5f} {seg.max().item():12.5f} {share:9.1f}%")
        order = torch.argsort(sens, descending=True)[:12]
        print("\n  单维敏感度 Top-12（索引 -> 所属分量）:")
        for idx in order.tolist():
            comp = next(n for lo, hi, n in COMPONENTS if lo <= idx < hi)
            print(f"    obs[{idx:2d}] = {sens[idx].item():.5f}   ({comp.split('(')[0].strip()})")
        # 盒子速度 6 维整体占比
        pv = sens[PARCEL_VEL[0]:PARCEL_VEL[1]]
        jv = sens[JOINT_VEL[0]:JOINT_VEL[1]]
        print(f"\n  汇总: 盒子速度6维占总敏感度 {100*pv.sum().item()/total:.1f}% | "
              f"关节速度18维占 {100*jv.sum().item()/total:.1f}% | "
              f"其余44维占 {100*(total-pv.sum().item()-jv.sum().item())/total:.1f}%", flush=True)

    # ==================================================================
    # 实验 A：输入消融评估
    # ==================================================================
    if not args.skip_ablation:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        print("\n" + "=" * 84)
        print(f"[A] 输入消融评估（每模式 {n_envs} envs x {args.episodes} episodes = {n_envs*args.episodes} 回合）")
        print("=" * 84)
        results: dict[str, tuple[int, int]] = {}
        for mode in modes:
            ok = tot = 0
            for ep in range(args.episodes):
                obs = env.reset()[0]
                ever = torch.zeros(n_envs, dtype=torch.bool, device=unwrapped.device)
                lu_cur = ot_cur = None
                parcel = unwrapped.scene["parcel"]
                origins = unwrapped.scene.env_origins

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
                        a = policy(apply_ablation(obs, mode))
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
            print(f"  {mode:20s} {ok:4d}/{tot:<4d} = {100.0*ok/tot:5.1f}%", flush=True)

        base = results["baseline"]
        print("\n  ---- 相对 baseline 的变化 ----")
        for mode, (ok, tot) in results.items():
            if mode == "baseline":
                continue
            d = 100.0 * ok / tot - 100.0 * base[0] / base[1]
            print(f"  {mode:20s} {d:+6.1f} 个百分点")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
