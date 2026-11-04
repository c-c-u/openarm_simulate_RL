"""脚本化位姿采样计划的单测（不需要 Isaac / GPU）。

为什么这些断言重要
------------------
采样计划决定了"数据集覆盖什么"，错了不会报错、只会训出偏置的模型：
* 面类型配额偏了 -> "已翻正"样本不够（这正是当前数据的病）；
* 网格有空格 -> 位置覆盖仍是一条走廊；
* rest_height 和 ``reset_parcel_root_state`` 不一致 -> 盒子被放到错误高度，落下来砸出
  非静止的 GT 帧。

用 ``python3 -m pytest openarm_lab/tests/test_pose_sampler.py`` 直接跑（宿主 python3 即可）。
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

# 直接按路径加载模块，**绕过包 __init__**：``openarm_lab/__init__.py`` 会 import isaaclab
# （只有 Isaac 进程里才有），而本模块是刻意做成零依赖的——这样"采样计划对不对"
# 可以在任何机器上用裸 python3 验证，不必启动仿真器。
_MOD = pathlib.Path(__file__).resolve().parents[1] / "tools" / "pose_sampler.py"
_spec = importlib.util.spec_from_file_location("pose_sampler", _MOD)
ps = importlib.util.module_from_spec(_spec)
sys.modules["pose_sampler"] = ps   # dataclass 解析类型注解时需要能在 sys.modules 里找到模块
_spec.loader.exec_module(ps)

FACES = ps.FACES
PlanConfig = ps.PlanConfig
PoseSpec = ps.PoseSpec
label_up_of = ps.label_up_of
plan_states = ps.plan_states
plan_summary = ps.plan_summary
rest_height = ps.rest_height

PARCEL = (0.09, 0.09, 0.07)


def test_plan_counts_and_faces():
    cfg = PlanConfig()
    specs = plan_states(200, cfg, seed=0)
    assert len(specs) == 200
    faces = {f: sum(1 for s in specs if s.face == f) for f in FACES}
    for f, p in cfg.face_probs.items():
        assert faces[f] == pytest.approx(200 * p, abs=3), f"{f} 配额偏差过大: {faces}"


def test_plan_respects_ranges():
    cfg = PlanConfig(x_range=(0.20, 0.40), y_range=(-0.02, 0.22))
    for s in plan_states(300, cfg, seed=1):
        assert cfg.x_range[0] <= s.xy[0] <= cfg.x_range[1]
        assert cfg.y_range[0] <= s.xy[1] <= cfg.y_range[1]
        assert 0.0 <= s.yaw_deg < 360.0


def test_grid_covers_all_cells():
    """分层网格：格数 ≤ 采样数时不允许出现空格子（否则位置覆盖有洞）。"""
    cfg = PlanConfig(grid=(5, 5))
    specs = plan_states(50, cfg, seed=2)
    summary = plan_summary(specs, cfg)
    assert summary["grid_cells"] == 25
    assert summary["grid_empty"] == 0


def test_grid_is_actually_stratified():
    """每格至少一个样本，且不是所有样本挤在同一格。"""
    cfg = PlanConfig(grid=(4, 3))
    specs = plan_states(24, cfg, seed=3)
    xy = np.array([s.xy for s in specs])
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[4, 3], range=[cfg.x_range, cfg.y_range])
    assert (H > 0).sum() == 12
    assert H.max() <= 2  # 24 个样本铺 12 格 -> 每格最多 2


def test_yaw_is_uniform():
    specs = plan_states(1000, PlanConfig(uniform_yaw=True), seed=4)
    yaw = np.array([s.yaw_deg for s in specs])
    # 均匀分布分 10 档，每档应接近 100（±45 容差足够严）
    hist = np.histogram(yaw, bins=10, range=(0, 360))[0]
    assert np.all(np.abs(hist - 100) < 45), hist


def test_rest_height_matches_events_convention():
    """必须与 ``events.py::reset_parcel_root_state`` 的 rest 计算一致。"""
    assert rest_height("side", PARCEL) == pytest.approx(PARCEL[0] / 2)   # 0.045
    assert rest_height("up", PARCEL) == pytest.approx(PARCEL[2] / 2)     # 0.035
    assert rest_height("down", PARCEL) == pytest.approx(PARCEL[2] / 2)   # 0.035
    with pytest.raises(ValueError):
        rest_height("tilted", PARCEL)


def test_label_up_expectation():
    assert label_up_of("up") == 1.0
    assert label_up_of("side") == 0.0
    assert label_up_of("down") == -1.0


def test_invalid_face_rejected():
    with pytest.raises(ValueError):
        PoseSpec(face="tilted", xy=(0.3, 0.1), yaw_deg=0.0)


def test_zero_and_normalization():
    assert plan_states(0, PlanConfig()) == []
    # 配额不归一也能用
    cfg = PlanConfig(face_probs={"up": 2.0, "side": 1.0, "down": 1.0})
    specs = plan_states(40, cfg, seed=5)
    assert sum(1 for s in specs if s.face == "up") == pytest.approx(20, abs=2)
