"""细分评测：对指定 checkpoint 做多维度成功率统计（全程无头）。

与 ``eval_model.py`` 的区别：``eval_model.py`` 只给总体成功率；本脚本把结果
按**初始条件**拆开，用于回答"模型在什么情况下行、什么情况下不行"：

* 按盒子初始朝向：标签在侧面（90° 翻）vs 朝下（180° 翻），侧面再细分为
  标签朝向 ±X / ±Y 四个方向（由标签法向的水平分量判定）；
* 按盒子初始位置：离机器人的远近（env-local x）与左右（y）；
* 成功速度：首次达成"标签朝上且在桌上"所用的策略步数；
* 失败模式：掉下桌 / 被推出界 / 全程未成功；
* 区分"过程中曾成功"（宽松）与"回合结束时仍保持"（严格）两种判据；
* 各项给出 95% 置信区间（Wilson 区间，小样本比正态近似可靠）。

用法::

    python.sh scripts/eval_breakdown.py \
        --checkpoint logs/rsl_rl/openarm_flip/<run>/model_20000.pt \
        --num_envs 64 --episodes 4
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
    "extsDeprecated/omni.isaac.ml_archive/pip_prebundle",  # torch / torchvision
    "exts/isaacsim.pip.newton/pip_prebundle",  # newton physics bindings
):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))

parser = argparse.ArgumentParser(description="Breakdown evaluation of an OpenArm parcel-flip checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default=None, help="Gym task id；缺省时按 checkpoint 路径自动推断。")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=4, help="每个环境的回合数。")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--episode_length_s", type=float, default=None,
                    help="覆盖回合时长（缺省沿用任务配置 = 训练时的 10 s）。")
args = parser.parse_known_args()[0]

# 任务推断（与 play.py / eval_model.py 同一张表；内联于此以便启动 Isaac 前就能完成）
_STATE_TASK = "OpenArm-Flip-State-v0"
_VISION_TASK = "OpenArm-Flip-Play-v0"

_EXPERIMENT_TASKS = {"openarm_flip": "OpenArm-Flip-State-v0", "openarm_flip_img": "OpenArm-Flip-v0"}


def _resolve_task(checkpoint: str, task: str | None) -> str:
    """未显式指定 --task 时按 checkpoint 路径推断（策略结构必须与任务匹配）。"""
    if task is not None:
        return task
    for part in reversed(pathlib.Path(checkpoint).resolve().parts):
        if part in _EXPERIMENT_TASKS:
            print(f"[EVAL] inferred task '{_EXPERIMENT_TASKS[part]}' from checkpoint path ('{part}/')", flush=True)
            return _EXPERIMENT_TASKS[part]
    # 回退：按文件名/路径子串判断（支持 models/ 下的重命名模型，
    # 例如 models/openarm_flip_state62_best.pt）
    joined = "/".join(p.lower() for p in pathlib.Path(checkpoint).resolve().parts)
    name = pathlib.Path(checkpoint).name.lower()
    if "openarm_flip_img" in joined or "vision" in name or "cnn" in name:
        print("[EVAL] inferred task from filename -> vision task", flush=True)
        return _VISION_TASK
    if "openarm_flip" in joined or "state62" in name or "state68" in name:
        print("[EVAL] inferred task from filename -> state task", flush=True)
        return _STATE_TASK
    raise SystemExit(f"[EVAL] cannot infer task from '{checkpoint}'; pass --task explicitly.")


task = _resolve_task(args.checkpoint, args.task)
_needs_cameras = task in {"OpenArm-Flip-v0", "OpenArm-Flip-Play-v0"}

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": _needs_cameras})
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

import openarm_lab  # noqa: F401
from isaaclab.utils.string import string_to_callable
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from openarm_lab.openarm_flip.openarm_flip_env_cfg import TABLE_POS, TABLE_SIZE, TABLE_TOP

UP_COS = math.cos(math.radians(15))  # 判定"标签朝上"的阈值（与训练/评估一致）
DROP_TOL = 0.08  # 与 parcel_dropped / parcel_on_table 一致的垂直容差（m）
OOB = {"x": (-0.2, 0.9), "y": (-0.6, 1.0)}  # parcel_out_of_bound 的范围


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间。"""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bucket(label: str, k: int, n: int) -> str:
    """一行分组统计：k/n、百分比与 95% 置信区间。"""
    lo, hi = wilson(k, n)
    rate = 100.0 * k / n if n else 0.0
    return f"  {label:34s} {k:4d}/{n:<4d} {rate:6.1f}%   (95% CI {100 * lo:5.1f}-{100 * hi:5.1f}%)"


def main() -> None:
    import importlib.metadata as metadata

    spec = gym.spec(task)
    # 按 checkpoint 的观测维度选择 62/68 维观测（构造时传入才生效）
    from openarm_lab.openarm_flip.checkpoint_utils import make_env_cfg

    env_cfg = make_env_cfg(
        string_to_callable(spec.kwargs["env_cfg_entry_point"]),
        args.checkpoint,
        enable_cameras=_needs_cameras,
    )
    agent_cfg = string_to_callable(spec.kwargs["rsl_rl_cfg_entry_point"])()
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, str(metadata.version("rsl-rl-lib")))
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    if args.episode_length_s is not None:
        env_cfg.episode_length_s = args.episode_length_s

    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
    print(f"[EVAL] loading {args.checkpoint}", flush=True)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    unwrapped = env.unwrapped
    n_envs = unwrapped.num_envs
    max_steps = int(unwrapped.max_episode_length)
    dt = unwrapped.step_dt
    print(
        f"[EVAL] {n_envs} envs x {args.episodes} episodes | 回合 {max_steps} 步 ({max_steps * dt:.1f} s) "
        f"| 成功阈值 cos15°={UP_COS:.4f}",
        flush=True,
    )

    records: list[dict] = []
    for ep in range(args.episodes):
        obs = env.reset()[0]
        parcel = unwrapped.scene["parcel"]
        origins = unwrapped.scene.env_origins
        # --- 记录本回合的初始条件 ---
        quat0 = parcel.data.root_quat_w.torch.clone()
        pos0 = (parcel.data.root_pos_w.torch - origins).clone()
        x0, y0, z0, w0 = quat0.unbind(dim=-1)  # (x, y, z, w)
        # 标签法向 = q 旋转 (0,0,1)：R 的第三列
        lu0 = 1.0 - 2.0 * (x0 * x0 + y0 * y0)  # 法向 z 分量（= 任务目标量）
        lx = 2.0 * (x0 * z0 + w0 * y0)  # 法向 x 分量
        ly = 2.0 * (y0 * z0 - w0 * x0)  # 法向 y 分量
        is_down = lu0 < -0.5  # 标签朝下 = 180° 起手
        # 侧面起手时用法向的水平主方向细分四类：0=+X 1=-X 2=+Y 3=-Y
        side_dir = torch.where(
            lx.abs() >= ly.abs(),
            (lx <= 0).long(),
            2 + (ly <= 0).long(),
        )

        ever_success = torch.zeros(n_envs, dtype=torch.bool, device=unwrapped.device)
        first_step = torch.full((n_envs,), -1, dtype=torch.long, device=unwrapped.device)
        up_steps = torch.zeros(n_envs, dtype=torch.long, device=unwrapped.device)  # 朝上且on_table的步数
        last_label_up = torch.zeros(n_envs, device=unwrapped.device)
        last_on_table = torch.zeros(n_envs, dtype=torch.bool, device=unwrapped.device)
        dropped = torch.zeros(n_envs, dtype=torch.bool, device=unwrapped.device)
        oob = torch.zeros(n_envs, dtype=torch.bool, device=unwrapped.device)

        def read_state():
            """读取当前盒子状态（env-local 位置、label_up、是否在桌上）。"""
            q = parcel.data.root_quat_w.torch
            p = parcel.data.root_pos_w.torch - origins
            qx, qy, _, _ = q.unbind(dim=-1)
            up = 1.0 - 2.0 * (qx * qx + qy * qy)
            ins = ((p[:, 0] - TABLE_POS[0]).abs() < TABLE_SIZE[0] / 2.0 - 0.03) & (
                (p[:, 1] - TABLE_POS[1]).abs() < TABLE_SIZE[1] / 2.0 - 0.03
            )
            return up, ins & (p[:, 2] > TABLE_TOP - DROP_TOL), p

        # 初始状态：reset 之后立即读取
        lu_cur, ot_cur, pos_cur = read_state()

        for step in range(max_steps):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, done, _ = env.step(actions)
            lu_new, ot_new, pos_new = read_state()
            # 注意：本步终止的 env 已被环境自动 reset，上面读到的是"新回合初始状态"。
            # 对这些 env，终止前的真实状态是上一轮读到的值，故用 cur 覆盖。
            d = done if done.dtype == torch.bool else done.bool()
            lu_eff = torch.where(d, lu_cur, lu_new)
            ot_eff = torch.where(d, ot_cur, ot_new)
            pos_eff = torch.where(d.unsqueeze(-1), pos_cur, pos_new)

            now_ok = (lu_eff > UP_COS) & ot_eff
            newly = now_ok & ~ever_success
            first_step = torch.where(newly, torch.full_like(first_step, step), first_step)
            ever_success |= now_ok
            up_steps += now_ok.long()
            last_label_up, last_on_table = lu_eff, ot_eff
            dropped |= pos_eff[:, 2] < TABLE_TOP - DROP_TOL
            oob |= (pos_eff[:, 0] < OOB["x"][0]) | (pos_eff[:, 0] > OOB["x"][1]) | (pos_eff[:, 1] < OOB["y"][0]) | (
                pos_eff[:, 1] > OOB["y"][1]
            )
            lu_cur, ot_cur, pos_cur = lu_new, ot_new, pos_new

        for i in range(n_envs):
            records.append({
                "is_down": bool(is_down[i]),
                "side_dir": int(side_dir[i]),
                "x0": float(pos0[i, 0]),
                "y0": float(pos0[i, 1]),
                "ever": bool(ever_success[i]),
                "final": bool(last_label_up[i] > UP_COS and last_on_table[i]),
                "first_step": int(first_step[i]),
                "up_ratio": float(up_steps[i]) / max_steps,
                "dropped": bool(dropped[i]),
                "oob": bool(oob[i]),
                "final_label_up": float(last_label_up[i]),
            })
        print(f"  ... episode {ep + 1}/{args.episodes} 完成", flush=True)

    n = len(records)
    ever = sum(r["ever"] for r in records)
    final = sum(r["final"] for r in records)

    print("\n" + "=" * 80)
    print(f"模型: {args.checkpoint}")
    print(f"样本: {n} 回合 ({args.num_envs} envs x {args.episodes} episodes), 每回合 {max_steps} 步 ({max_steps * dt:.1f} s)")
    print("=" * 80)

    print("\n【总体成功率】")
    print(bucket("过程中曾达到（宽松）", ever, n))
    print(bucket("结束时仍保持（严格）", final, n))

    print("\n【按盒子初始朝向】")
    downs = [r for r in records if r["is_down"]]
    sides = [r for r in records if not r["is_down"]]
    print(bucket("标签朝下 180°（难）", sum(r["ever"] for r in downs), len(downs)))
    print(bucket("标签在侧面 90°（易）", sum(r["ever"] for r in sides), len(sides)))
    side_names = {0: "  └ 侧面: 标签朝 +X", 1: "  └ 侧面: 标签朝 -X", 2: "  └ 侧面: 标签朝 +Y", 3: "  └ 侧面: 标签朝 -Y"}
    for k in (0, 1, 2, 3):
        grp = [r for r in sides if r["side_dir"] == k]
        if grp:
            print(bucket(side_names[k], sum(r["ever"] for r in grp), len(grp)))

    print(f"\n【按盒子初始位置】(env-local；桌子中心 x={TABLE_POS[0]}, y={TABLE_POS[1]})")
    xm = sum(r["x0"] for r in records) / n
    ym = sum(r["y0"] for r in records) / n
    for lbl, sel in (
        (f"靠机器人侧 (x < {xm:.3f})", [r for r in records if r["x0"] < xm]),
        (f"靠外侧 (x >= {xm:.3f})", [r for r in records if r["x0"] >= xm]),
        (f"左侧 (y < {ym:.3f})", [r for r in records if r["y0"] < ym]),
        (f"右侧 (y >= {ym:.3f})", [r for r in records if r["y0"] >= ym]),
    ):
        if sel:
            print(bucket(lbl, sum(r["ever"] for r in sel), len(sel)))

    print("\n【失败模式】(按回合统计，可重叠)")
    print(f"  掉下桌          : {sum(r['dropped'] for r in records):4d} / {n}")
    print(f"  被推出界        : {sum(r['oob'] for r in records):4d} / {n}")
    never = [r for r in records if not r["ever"]]
    print(f"  全程未成功      : {len(never):4d} / {n}")
    if never:
        print(f"    └ 这些回合结束时 label_up 均值: {sum(r['final_label_up'] for r in never) / len(never):+.3f}")

    print("\n【成功用时 —— 从回合开始到首次达成'标签朝上且在桌上'】")
    ok = [r for r in records if r["first_step"] >= 0]
    steps = [r["first_step"] for r in ok]

    def pct(vals, q: float) -> float:
        s = sorted(vals)
        return float(s[min(len(s) - 1, int(q * len(s)))])

    if steps:
        mean_s = sum(steps) / len(steps)
        med = pctl_med = pct(steps, 0.5)
        print(f"  成功回合数: {len(steps)} / {n}")
        print(f"  平均用时  : {mean_s:.1f} 步 = {mean_s * dt:.2f} s")
        print(f"  中位用时  : {med:.0f} 步 = {med * dt:.2f} s")
        print(f"  分位数    : P10 {pct(steps,0.10):.0f} 步 ({pct(steps,0.10)*dt:.2f} s) | "
              f"P25 {pct(steps,0.25):.0f} 步 ({pct(steps,0.25)*dt:.2f} s) | "
              f"P75 {pct(steps,0.75):.0f} 步 ({pct(steps,0.75)*dt:.2f} s) | "
              f"P90 {pct(steps,0.90):.0f} 步 ({pct(steps,0.90)*dt:.2f} s)")
        print(f"  最快/最慢 : {min(steps)} 步 ({min(steps)*dt:.2f} s) / {max(steps)} 步 ({max(steps)*dt:.2f} s)")
        print("  用时分布  :")
        for lo, hi in ((1, 15), (15, 30), (30, 60), (60, 120), (120, 10**6)):
            grp = [x for x in steps if lo <= x < hi]
            label = f"{lo}-{hi} 步" if hi < 10**6 else f">= {lo} 步"
            bar = "#" * int(40 * len(grp) / len(steps))
            print(f"    {label:14s} {lo*dt:.2f}-{hi*dt if hi < 10**6 else max(steps)*dt:.2f} s : "
                  f"{len(grp):4d} ({100*len(grp)/len(steps):5.1f}%) {bar}")

    print("\n【成功用时 —— 按初始朝向】")
    for lbl, sel in (
        ("标签在侧面 90°", [r for r in ok if not r["is_down"]]),
        ("标签朝下 180°", [r for r in ok if r["is_down"]]),
    ):
        if sel:
            st = [r["first_step"] for r in sel]
            m = sum(st) / len(st)
            print(f"  {lbl:16s} {len(st):4d} 回合 | 平均 {m:6.1f} 步 = {m*dt:.2f} s | "
                  f"中位 {pct(st,0.5):.0f} 步 = {pct(st,0.5)*dt:.2f} s")

    print("\n【成功用时 —— 按是否保持到回合结束】")
    for lbl, sel in (
        ("成功后保持住", [r for r in ok if r["final"]]),
        ("成功后又被破坏", [r for r in ok if not r["final"]]),
    ):
        if sel:
            st = [r["first_step"] for r in sel]
            m = sum(st) / len(st)
            print(f"  {lbl:16s} {len(st):4d} 回合 | 平均 {m:6.1f} 步 = {m*dt:.2f} s | "
                  f"中位 {pct(st,0.5):.0f} 步 = {pct(st,0.5)*dt:.2f} s")

    print("\n【保持能力】(成功之后能不能稳住)")
    ratio = sum(r["up_ratio"] for r in records) / n
    print(f"  平均'朝上且未掉桌'时间占比: {100 * ratio:.1f}%  (成功回合 "
          f"{100 * sum(r['up_ratio'] for r in records if r['ever'])/max(1, ever):.1f}%"
          f" / 失败回合 {100 * sum(r['up_ratio'] for r in records if not r['ever'])/max(1, n - ever):.1f}%)")
    kept = [r for r in records if r["ever"] and r["final"]]
    print(f"  成功后保持到回合结束: {len(kept)}/{ever} = {100 * len(kept) / max(1, ever):.1f}%")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
