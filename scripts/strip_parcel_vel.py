"""把 68 维观测的 checkpoint "手术"成 62 维：删掉盒子线速度+角速度（索引 43-48）。

原理
----
策略内部先做观测归一化 ``n = (x - mean) / std``，第一层再算 ``h = W @ n + b``。
真机上拿不到盒子速度时我们给这 6 维填常数 ``c``（通常 0），此时它们对第一层的
贡献是一个**非零常数**::

    h_vel = W[:, 43:49] @ ((c - mean_vel) / std_vel)

所以直接删掉这 6 列权重**并不等价**，必须把这份贡献补进第一层 bias::

    b_new = b_old + W[:, 43:49] @ ((c - mean_vel) / std_vel)

补上之后，"62 维模型"与"68 维模型 + 速度维填 c"在数学上完全等价 —— 脚本会用
随机观测做一次前向对比来自证这一点（误差应在 1e-5 量级）。

同时处理：
* ``obs_normalizer`` 的 ``_mean/_var/_std/count`` 同步裁剪到 62 维；
* actor 与 critic 都裁（续训时两者观测维度必须一致）；
* ``optimizer_state_dict`` 里形状为 ``(hidden, 68)`` 的动量张量按同样索引裁剪，
  以便这个 checkpoint 还能直接用于续训（否则加载优化器会形状不匹配）。

用法::

    # 输出到 models/ 下并做数值验证
    python3 scripts/strip_parcel_vel.py \
        --checkpoint logs/rsl_rl/openarm_flip/<run>/model_20000.pt \
        --output models/openarm_flip_state62_best.pt --verify
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import torch

# 观测 68 维排布（StateObservationsFullCfg）：
#   0-17 joint_pos | 18-35 joint_vel | 36-38 parcel_pos | 39-42 parcel_quat
#   43-45 parcel_lin_vel | 46-48 parcel_ang_vel | 49-51 label_normal | 52-67 last_action
PARCEL_VEL_IDX = list(range(43, 49))  # 要删掉的 6 维

parser = argparse.ArgumentParser(description="Strip the parcel velocity dims (68 -> 62) from a checkpoint.")
parser.add_argument("--checkpoint", type=str, required=True, help="输入 checkpoint（68 维）。")
parser.add_argument("--output", type=str, required=True, help="输出的 62 维 checkpoint 路径。")
parser.add_argument("--fill", type=float, default=0.0,
                    help="真机上这 6 维将填入的常数值（默认 0）。bias 补偿按此值计算，"
                         "保证手术后的模型与'填该常数'完全等价。")
parser.add_argument("--norm_eps", type=float, default=1e-2,
                    help="rsl-rl EmpiricalNormalization 的 eps（默认 1e-2）。归一化实际是\n"
                         "  (x - mean) / (std + eps)，漏掉 eps 会让 bias 补偿与验证同时出错。")
parser.add_argument("--verify", action="store_true", help="用随机观测做前向对比，验证等价性。")
args = parser.parse_args()


def _flatten(t: torch.Tensor) -> torch.Tensor:
    """把 (1, N) / (N,) 统一成 (N,)，仅用于数值计算。"""
    return t.reshape(-1)


def _slice(t: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """按最后一维裁剪，**保持原有形状**（归一化器的 buffer 可能是 (1, N)）。"""
    return t.index_select(-1, keep).clone()


def _strip_state_dict(sd: dict, keep: torch.Tensor, fill: float, tag: str, norm_eps: float) -> dict:
    """裁剪一个模型 state_dict（输入层权重列 + 归一化器 + bias 补偿）。

    注意：``obs_normalizer._mean/_var/_std`` 在 checkpoint 里的形状是 ``(1, N)``，
    必须用 ``index_select(-1, ...)`` 保持二维形状，否则加载时会 shape mismatch。
    """
    out = dict(sd)
    W = sd["mlp.0.weight"]  # (hidden, N)
    b = sd["mlp.0.bias"]  # (hidden,)
    mean = sd["obs_normalizer._mean"]
    std = sd["obs_normalizer._std"]

    # 被删维度"填常数"后，归一化得到的值（用扁平化版本做数值计算）
    # 注意：rsl-rl 的归一化是 (x - mean) / (std + eps)，eps 默认 1e-2
    n_vel = (fill - _flatten(mean)[PARCEL_VEL_IDX]) / (
        _flatten(std)[PARCEL_VEL_IDX] + norm_eps
    )  # (6,)
    delta = W[:, PARCEL_VEL_IDX] @ n_vel  # (hidden,) 需要补进 bias 的常数贡献

    out["mlp.0.weight"] = _slice(W, keep)
    out["mlp.0.bias"] = (b + delta).clone()

    out["obs_normalizer._mean"] = _slice(mean, keep)
    out["obs_normalizer._var"] = _slice(sd["obs_normalizer._var"], keep)
    out["obs_normalizer._std"] = _slice(std, keep)
    # count 是标量（样本计数），不随维度变化；若意外与观测同维则一并裁剪
    cnt = sd.get("obs_normalizer.count")
    if cnt is not None:
        out["obs_normalizer.count"] = _slice(cnt, keep) if cnt.numel() == _flatten(mean).numel() else cnt.clone()

    print(f"  [{tag}] mlp.0.weight {tuple(W.shape)} -> {tuple(out['mlp.0.weight'].shape)}"
          f" | normalizer._mean {tuple(mean.shape)} -> {tuple(out['obs_normalizer._mean'].shape)}"
          f" | bias 补偿幅度 mean|delta|={delta.abs().mean().item():.6f}")
    return out


def _strip_optimizer(opt: dict, keep: torch.Tensor) -> dict:
    """裁剪 Adam 动量中形状为 (hidden, 68) 的张量（输入层权重的动量）。"""
    out = {"state": {}, "param_groups": opt.get("param_groups", [])}
    n_patched = 0
    for k, v in opt.get("state", {}).items():
        new_v = {}
        for name, val in v.items():
            if torch.is_tensor(val) and val.dim() == 2 and val.shape[1] == 68:
                new_v[name] = val[:, keep].clone()
                n_patched += 1
            else:
                new_v[name] = val.clone() if torch.is_tensor(val) else val
        out["state"][k] = new_v
    print(f"  [optimizer] 裁剪了 {n_patched} 个动量张量（形状 (hidden, 68)）")
    return out


def _manual_forward(sd: dict, x: torch.Tensor, norm_eps: float) -> torch.Tensor:
    """手工复现 actor 的确定性前向：normalizer -> mlp(elu) 三层。

    归一化必须与 rsl-rl 的 EmpiricalNormalization.forward 完全一致：
    ``(x - mean) / (std + eps)``（eps 默认 1e-2）—— 漏掉 eps 会让验证失去意义。
    """
    mean = sd["obs_normalizer._mean"]
    std = sd["obs_normalizer._std"]
    h = (x - mean) / (std + norm_eps)
    for i in (0, 2, 4, 6):
        h = h @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"]
        if i != 6:  # 最后一层无激活
            h = torch.nn.functional.elu(h)
    return h


def main() -> None:
    src = pathlib.Path(args.checkpoint)
    dst = pathlib.Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    keep = torch.tensor([i for i in range(68) if i not in PARCEL_VEL_IDX])
    print(f"[strip] {src.name}  (fill={args.fill})")

    new_ckpt = dict(ckpt)
    new_ckpt["actor_state_dict"] = _strip_state_dict(
        ckpt["actor_state_dict"], keep, args.fill, "actor", args.norm_eps
    )
    new_ckpt["critic_state_dict"] = _strip_state_dict(
        ckpt["critic_state_dict"], keep, args.fill, "critic", args.norm_eps
    )
    if "optimizer_state_dict" in ckpt:
        new_ckpt["optimizer_state_dict"] = _strip_optimizer(ckpt["optimizer_state_dict"], keep)
    # 记录手术信息，便于日后追溯
    infos = dict(ckpt.get("infos", {}) or {})
    infos["stripped_parcel_vel"] = {
        "source": str(src),
        "removed_obs_indices": PARCEL_VEL_IDX,
        "obs_dim": 62,
        "fill_value": args.fill,
        "note": "equivalent to the 68-dim model with parcel lin/ang vel filled with `fill_value`",
    }
    new_ckpt["infos"] = infos

    if args.verify:
        torch.manual_seed(0)
        x68 = torch.randn(64, 68)
        x68[:, PARCEL_VEL_IDX] = args.fill  # 模拟"填常数"
        out68 = _manual_forward(ckpt["actor_state_dict"], x68, args.norm_eps)
        out62 = _manual_forward(new_ckpt["actor_state_dict"], x68[:, keep], args.norm_eps)
        err = (out68 - out62).abs().max().item()
        scale = out68.abs().mean().item()
        print(f"  [verify] 前向最大误差 = {err:.3e}  (输出量级 {scale:.3f})  -> "
              f"{'等价 ✓' if err < 1e-4 else '不一致 ✗ 请检查归一化公式'}")
        if err >= 1e-4:
            raise SystemExit("手术后的模型与填常数版本不等价，已中止保存。")

    torch.save(new_ckpt, dst)
    print(f"[strip] 已保存 62 维模型 -> {dst}  ({dst.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    sys.exit(main())
