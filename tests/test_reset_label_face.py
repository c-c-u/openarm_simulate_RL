"""``reset_parcel_root_state`` 的 label_face 支持（脚本化采样用）的单测。

不需要 Isaac：``events.py`` 只依赖 torch 与 isaaclab 的三个符号，这里把 isaaclab
换成最小 stub，再用"假的 env / asset"接住 ``write_root_pose_to_sim_index`` 的输出，
就能真跑一遍姿态解算与落座高度。

**为什么必须测**：脚本化采样完全依赖"把盒子摆到指定面"这一个能力。姿态或高度算错
不会报错，只会产出"面类型与标注不符"或"悬空落下的 GT 帧"，训练时表现为模型学不会
某个面 —— 而且很难从 loss 上看出原因。

    python3 -m pytest openarm_lab/tests/test_reset_label_face.py
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# ---------------------------------------------------------------------------
# 用 stub 顶掉 isaaclab（只用到三个符号）
# ---------------------------------------------------------------------------


def _quat_from_matrix_x_y_z_w(m: torch.Tensor) -> torch.Tensor:
    """3x3 旋转矩阵 -> (x, y, z, w)。与 Isaac Lab 的约定一致。"""
    out = []
    for r in m.reshape(-1, 3, 3).cpu().numpy():
        t = r[0, 0] + r[1, 1] + r[2, 2]
        if t > 0:
            s = math.sqrt(t + 1.0) * 2
            w, x = 0.25 * s, (r[2, 1] - r[1, 2]) / s
            y, z = (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            w, x = (r[2, 1] - r[1, 2]) / s, 0.25 * s
            y, z = (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            w, x = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s
            y, z = 0.25 * s, (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            w, x = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s
            y, z = (r[1, 2] + r[2, 1]) / s, 0.25 * s
        out.append([x, y, z, w])
    return torch.tensor(out, dtype=torch.float32)


def _quat_xyzw_to_matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(dim=-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))


def _install_stubs() -> None:
    for name in ("isaaclab", "isaaclab.envs", "isaaclab.managers", "isaaclab.utils",
                 "isaaclab.utils.math"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["isaaclab.envs"].ManagerBasedEnv = object  # 仅用于类型注解
    sys.modules["isaaclab.managers"].SceneEntityCfg = (
        lambda name=None, **kw: types.SimpleNamespace(name=name))
    sys.modules["isaaclab.utils.math"].quat_from_matrix = _quat_from_matrix_x_y_z_w


def _load_events():
    _install_stubs()
    mod_path = pathlib.Path(__file__).resolve().parents[1] / "openarm_flip" / "mdp" / "events.py"
    spec = importlib.util.spec_from_file_location("oa_events", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oa_events"] = mod
    spec.loader.exec_module(mod)
    return mod


EV = _load_events()

PARCEL = (0.09, 0.09, 0.07)
TABLE_POS = (0.3, 0.15, 0.0)
TABLE_TOP = 0.30


# ---------------------------------------------------------------------------
# 假的 env / asset：接住 write_root_pose_to_sim_index
# ---------------------------------------------------------------------------


class _FakeAsset:
    def __init__(self):
        self.root_pose = None
        self.root_vel = None
        self.device = torch.device("cpu")   # 事件函数里会读 asset.device

    def write_root_pose_to_sim_index(self, root_pose=None, env_ids=None):
        self.root_pose = root_pose.detach().clone()

    def write_root_velocity_to_sim_index(self, root_velocity=None, env_ids=None):
        self.root_vel = root_velocity.detach().clone()


class _FakeScene(dict):
    def __init__(self, n: int = 8):
        super().__init__()
        self.env_origins = torch.zeros(n, 3)   # 事件里按 env_ids 索引，长度要够


class _FakeEnv:
    def __init__(self, asset=None, n: int = 8):
        self.scene = _FakeScene(n)
        self.asset = asset or _FakeAsset()
        # 事件函数里是 ``asset = env.scene[asset_cfg.name]``，所以要把假资产挂进场景
        self.scene["parcel"] = self.asset


def _place(face: str, n: int = 1, seed: int = 0):
    torch.manual_seed(seed)
    env = _FakeEnv(n=max(n, 1))
    EV.reset_parcel_root_state(
        env,
        torch.arange(n),
        asset_cfg=types.SimpleNamespace(name="parcel"),
        table_pos=TABLE_POS,
        table_size=(0.5, 0.7, 0.05),
        table_top=TABLE_TOP,
        parcel_size=PARCEL,
        parcel_xy=(0.2, 0.1),
        pos_jitter=0.0,
        side_prob=0.7,
        velocity_range=(0.0, 0.0),
        label_face=face,
    )
    pose = env.asset.root_pose
    pos, quat = pose[:, :3], pose[:, 3:]
    R = _quat_xyzw_to_matrix(quat)
    label_normal = R[..., 2]           # 局部 +Z 在世界系的方向
    return pos, R, label_normal, quat


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("face,expect_up", [("up", 1.0), ("side", 0.0), ("down", -1.0)])
def test_label_face_sets_orientation(face, expect_up):
    _, _, label_normal, _ = _place(face, n=8)
    assert np.allclose(label_normal[:, 2].numpy(), expect_up, atol=1e-4), (
        f"label_face={face} 的 label_up 应为 {expect_up}，实际 "
        f"{label_normal[:, 2].numpy()}")


@pytest.mark.parametrize("face,expect_rest", [("up", 0.035), ("down", 0.035), ("side", 0.045)])
def test_label_face_sets_rest_height(face, expect_rest):
    """静置高度：up/down 是局部 Z 竖直（sz/2），side 是翻转 90°（sx/2）。"""
    pos, _, _, _ = _place(face, n=8)
    z = pos[:, 2].numpy() - TABLE_TOP
    assert np.allclose(z, expect_rest, atol=1e-5), f"{face} 落座高度 {z}，应为 {expect_rest}"


def test_side_label_points_horizontally():
    """side 的标签法向必须是**水平单位向量**：z≈0，且水平分量模长为 1。

    注意不能断言"落在 ±X/±Y 上"——yaw 会把它绕 Z 转到任意方位角。
    """
    _, _, label_normal, _ = _place("side", n=16)
    ln = label_normal.numpy()
    assert np.allclose(ln[:, 2], 0.0, atol=1e-4)
    assert np.allclose(np.linalg.norm(ln[:, :2], axis=1), 1.0, atol=1e-4)
    # yaw 均匀 -> 方位角应散布在多个象限，而不是钉在 0/90/180/270
    ang = np.degrees(np.arctan2(ln[:, 1], ln[:, 0]))
    assert len(np.unique(np.round(ang))) > 8


def test_up_down_differ_by_180_deg_about_horizontal():
    """up 与 down 的标签法向应恰好反向（同一落座高度）。"""
    _, R_up, ln_up, _ = _place("up", n=4, seed=1)
    _, R_down, ln_down, _ = _place("down", n=4, seed=1)
    assert np.allclose(ln_up.numpy(), -ln_down.numpy(), atol=1e-4)


def test_position_is_centred_on_parcel_xy():
    pos, _, _, _ = _place("up", n=4)
    assert np.allclose(pos[:, 0].numpy(), 0.2, atol=1e-5)
    assert np.allclose(pos[:, 1].numpy(), 0.1, atol=1e-5)


def test_random_mode_still_works():
    """默认 label_face='random' 必须保持原行为（向后兼容）。"""
    pos, _, label_normal, _ = _place("random", n=64, seed=3)
    up = label_normal[:, 2].numpy()
    # 只有两种可能：侧躺(≈0) 或 朝下(≈-1)，绝不该出现"朝上"
    assert np.all(np.isclose(up, 0.0, atol=1e-4) | np.isclose(up, -1.0, atol=1e-4))
    assert not np.any(up > 0.5)
    # side_prob=0.7 -> 侧躺约占 70%
    frac_side = float(np.isclose(up, 0.0, atol=1e-4).mean())
    assert 0.4 < frac_side < 0.95, frac_side


def test_invalid_label_face_raises():
    with pytest.raises(ValueError):
        _place("tilted", n=1)


def test_velocity_is_zero_and_pose_shape():
    env = _FakeEnv(n=1)
    EV.reset_parcel_root_state(
        env, torch.tensor([0]), asset_cfg=types.SimpleNamespace(name="parcel"),
        table_pos=TABLE_POS, table_size=(0.5, 0.7, 0.05), table_top=TABLE_TOP,
        parcel_size=PARCEL, parcel_xy=(0.3, 0.15), pos_jitter=0.0, label_face="up",
    )
    assert env.asset.root_pose.shape == (1, 7)
    assert env.asset.root_vel.shape == (1, 6)
    assert torch.allclose(env.asset.root_vel, torch.zeros_like(env.asset.root_vel))
