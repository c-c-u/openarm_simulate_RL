# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset / randomization events for the parcel-flip task."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_matrix


def _align_rotation(label_dir: torch.Tensor, side: torch.Tensor, k: torch.Tensor,
                    up: torch.Tensor | None = None) -> torch.Tensor:
    """Rotation (N, 3, 3) mapping the parcel's local +Z onto ``label_dir``.

    ``side`` selects a horizontal label direction (label on a side face), ``k``
    picks which of the four sides. The box is left resting on a face: for side
    labels the box tips 90 deg (rest height = sx/2), for a down label it is
    rotated 180 deg (rest height = sz/2), for ``up`` it is the identity
    (label already points to world +Z, rest height = sz/2).
    """
    n = label_dir.shape[0]
    device = label_dir.device
    eye = torch.eye(3, device=device).expand(n, 3, 3)

    def rot_x(a: torch.Tensor) -> torch.Tensor:
        r = eye.clone()
        c, s = torch.cos(a), torch.sin(a)
        r[:, 1, 1] = c
        r[:, 1, 2] = -s
        r[:, 2, 1] = s
        r[:, 2, 2] = c
        return r

    def rot_y(a: torch.Tensor) -> torch.Tensor:
        r = eye.clone()
        c, s = torch.cos(a), torch.sin(a)
        r[:, 0, 0] = c
        r[:, 0, 2] = s
        r[:, 2, 0] = -s
        r[:, 2, 2] = c
        return r

    # k: 0 -> label +X (Ry(+90)), 1 -> label -X (Ry(-90)),
    #    2 -> label +Y (Rx(-90)), 3 -> label -Y (Rx(+90))
    half = torch.full((n,), math.pi / 2.0, device=device)
    _zero = torch.zeros(n, device=device)
    ax = torch.where(
        up if up is not None else torch.zeros(n, dtype=torch.bool, device=device),
        _zero,   # 标签朝上：不需要翻转
        torch.where(
            side & (k >= 2),
            torch.where(k == 2, -half, half),
            torch.where(side, _zero, torch.full((n,), math.pi, device=device)),
        ),
    )
    ay = torch.where(
        side & (k < 2),
        torch.where(k == 0, half, -half),
        torch.zeros(n, device=device),
    )
    return rot_x(ax) @ rot_y(ay)


def reset_parcel_root_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    table_pos: tuple[float, float, float],
    table_size: tuple[float, float, float],
    table_top: float,
    parcel_size: tuple[float, float, float],
    parcel_xy: tuple[float, float] | None = None,
    pos_jitter: float = 0.03,
    side_prob: float = 0.7,
    velocity_range: tuple[float, float] = (0.0, 0.0),
    label_face: str = "random",
) -> None:
    """Reset the parcel to rest on the table with a random label face and yaw.

    The label (local +Z) is placed on one of the four side faces with
    probability ``side_prob``, or facing down with probability ``1 - side_prob``.
    The yaw is uniform in [0, 2*pi). The parcel is dropped with zero velocity at
    the correct rest height for the chosen face.

    ``parcel_xy`` is the env-local (x, y) the parcel is centred on (defaults to
    the table centre); small jitter is added around it.

    ``label_face`` 强制指定标签朝向，用于**脚本化位姿采样**（``scripts/collect_posed_data.py``）：

    * ``"random"``（默认）：沿用 ``side_prob`` 的随机策略（向后兼容）；
    * ``"side"``  / ``"down"``：只放成侧面朝上 / 标签朝下；
    * ``"up"``：**标签朝上（已翻正）**——原实现无法产生这个状态，所以"翻正后"的样本
      过去只能从回合末尾捡，脚本化采样必须能主动摆出来。

    盒子都按"贴着桌面静置"的姿态放置（中心高度 = 对应半边长），因此不会落下穿模。
    """
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    device = asset.device

    sx, sy, sz = parcel_size
    base_x = parcel_xy[0] if parcel_xy is not None else table_pos[0]
    base_y = parcel_xy[1] if parcel_xy is not None else table_pos[1]

    side = torch.rand(n, device=device) < side_prob
    up = torch.zeros(n, dtype=torch.bool, device=device)
    if label_face == "up":
        up[:] = True
        side[:] = False
    elif label_face == "down":
        side[:] = False
    elif label_face == "side":
        side[:] = True
    elif label_face != "random":
        raise ValueError(f"label_face 必须是 random/side/down/up，收到 {label_face!r}")
    k = torch.randint(0, 4, (n,), device=device)
    yaw = torch.rand(n, device=device) * 2.0 * math.pi

    # label direction in world (pre-yaw): horizontal for side faces, -Z for down
    label_dir = torch.zeros(n, 3, device=device)
    label_dir[:, 0] = torch.where(side & (k == 0), 1.0, torch.where(side & (k == 1), -1.0, 0.0))
    label_dir[:, 1] = torch.where(side & (k == 2), 1.0, torch.where(side & (k == 3), -1.0, 0.0))
    label_dir[:, 2] = torch.where(side, 0.0, torch.where(up, 1.0, -1.0))

    align = _align_rotation(label_dir, side, k, up)  # (N, 3, 3)

    # yaw about world Z applied after the align rotation
    cy, sy_ = torch.cos(yaw), torch.sin(yaw)
    rz = torch.zeros(n, 3, 3, device=device)
    rz[:, 0, 0] = cy
    rz[:, 0, 1] = -sy_
    rz[:, 1, 0] = sy_
    rz[:, 1, 1] = cy
    rz[:, 2, 2] = 1.0
    rot = rz @ align  # (N, 3, 3)
    # Isaac Lab sim interface uses (x, y, z, w) quaternions throughout.
    quat = quat_from_matrix(rot)  # (N, 4) in (x, y, z, w) order

    # rest height: up/down -> 局部 Z 竖直（sz/2）；side +-X -> sx/2；side +-Y -> sy/2
    rest = torch.where(side, torch.where(k < 2, sx / 2.0, sy / 2.0), sz / 2.0)
    jitter = torch.rand(n, 2, device=device) * 2.0 * pos_jitter - pos_jitter
    # write_root_pose_to_sim_index expects WORLD positions: env origin + offset
    pos = env.scene.env_origins[env_ids].clone()
    pos[:, 0] += base_x + jitter[:, 0]
    pos[:, 1] += base_y + jitter[:, 1]
    pos[:, 2] += table_top + rest

    root_pose = torch.cat([pos, quat], dim=-1)
    asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros(n, 6, device=device),
        env_ids=env_ids,
    )


# -----------------------------------------------------------------------------
# 腕部相机位姿同步（interval 事件，每个策略步调用一次）
# -----------------------------------------------------------------------------

# 相机朝前的局部旋转：把相机光轴从 link7 的 +Z 翻到 -Z（工具轴指向指尖）
_QUAT_FLIP_X = (1.0, 0.0, 0.0, 0.0)  # 绕 X 轴 180°，(x, y, z, w)


def sync_wrist_cameras(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    camera_bindings: dict | None = None,
    local_offset: tuple[float, float, float] = (0.15, 0.0, 0.0),
    look_at_forward: float | None = 0.13,
    flip_forward: bool = False,
) -> None:
    """把每个腕部相机放到"眼在手"位姿上（按宿主 link 的世界位姿，每步同步）。

    为什么需要手动同步：Isaac Lab 在 fabric 模式下**不把 link 的变换回写 USD**，
    因此挂在 link 下的相机 prim 生成后位姿就被冻结（实测：机械臂移动 4.4 cm 时
    相机世界坐标位移 0.00 cm），图像全黑——相机停在了错误位置并朝向自身几何。
    这里改为每步用 link 的**物理**位姿主动设置相机世界位姿，绕开 USD 层级同步
    （思路同 openarm_sorting 的 ``WristCamera.update_pose``）。

    Args:
        env: 环境（interval 事件第一个参数）。
        env_ids: 间隔事件传入的环境索引（本函数更新全部环境，未使用）。
        robot_cfg: 机器人资产配置。
        camera_bindings: ``{相机名: 宿主 link body 名}`` 映射，例如
            ``{"wrist_left_cam": "openarm_left_link7"}``。相机不存在时自动跳过
            （纯状态任务 / ``--lean_cameras`` / ``--disable_cameras`` 下不会报错）。
        local_offset: 相机在 link 局部系中的安装偏移。默认 ``(0.15, 0, 0)`` ——
            沿局部 **+X** 15 cm。几何依据：夹爪开合方向 = 局部 -Y、工具轴/腕部转轴 =
            局部 +Z，垂直于这两者的即局部 X 轴；+X 在世界系 z 分量为正（"上方"一侧）。
        look_at_forward: 若给定，则相机**注视** link 局部工具轴前方这么远（米）的点
            —— 即从侧上方俯视夹爪/指尖（0.13 ≈ 指尖位置）。设为 None 则改用
            ``flip_forward`` 指定的固定朝向（相机与 link 同姿态，可选翻转 180°）。
        flip_forward: 仅在 ``look_at_forward is None`` 时生效：是否把光轴翻转 180°
            （绕局部 X）。按 ROS 约定相机 forward = 所在坐标系的 +Z。
    """
    from isaaclab.utils.math import quat_apply, quat_mul

    if not camera_bindings:
        return
    robot = env.scene[robot_cfg.name]
    # data.* 是 warp 数组：取 .torch 才是 torch 张量（device 也才是 torch.device）
    pos_all = robot.data.body_pos_w.torch  # (N, B, 3)
    quat_all = robot.data.body_quat_w.torch  # (N, B, 4) (x, y, z, w)
    device = pos_all.device
    offset = torch.tensor(local_offset, device=device, dtype=torch.float32).unsqueeze(0)  # (1, 3)
    flip = torch.tensor(_QUAT_FLIP_X, device=device, dtype=torch.float32).unsqueeze(0)  # (1, 4)

    for cam_name, body_name in camera_bindings.items():
        sensors = env.scene.sensors
        if cam_name not in sensors:
            continue  # 该相机未创建（如 --lean_cameras / 纯状态任务）
        cam = sensors[cam_name]
        if body_name not in robot.data.body_names:
            continue
        idx = robot.data.body_names.index(body_name)
        pos_w = pos_all[:, idx]  # (N, 3)
        quat_w = quat_all[:, idx]  # (N, 4)
        # 相机位置：link 原点 + 局部偏移旋到世界系
        cam_pos = pos_w + quat_apply(quat_w, offset.expand(pos_w.shape[0], -1))
        if look_at_forward is not None:
            # 注视点：沿 link 的工具轴（局部 +Z）前方 look_at_forward 米
            fwd_local = torch.tensor([0.0, 0.0, look_at_forward], device=device, dtype=torch.float32)
            target = pos_w + quat_apply(quat_w, fwd_local.expand(pos_w.shape[0], -1))
            cam.set_world_poses_from_view(eyes=cam_pos, targets=target)
        else:
            # 固定朝向：相机与 link 同姿态（可选翻转光轴），按 ROS 约定转换
            ori = quat_mul(quat_w, flip.expand(quat_w.shape[0], -1)) if flip_forward else quat_w
            cam.set_world_poses(positions=cam_pos, orientations=ori, convention="ros")


# -----------------------------------------------------------------------------
# 场景表面纹理（startup 事件）
# -----------------------------------------------------------------------------

# 程序化生成的纹理（tools/make_textures.py 产出，随仓库分发，无需下载）
_TEXTURE_DIR = Path(__file__).resolve().parents[2] / "assets" / "textures"


def _make_texture_material(
    stage,
    mat_path: str,
    tex_file: str,
    roughness: float = 0.85,
    metallic: float = 0.0,
    uv_scale: tuple[float, float] = (1.0, 1.0),
):
    """在 ``mat_path`` 创建一个 UsdPreviewSurface + UsdUVTexture 材质。

    纹理通过 ``st`` primvar 采样；``uv_scale`` 用来控制铺贴密度（>1 表示重复更多次）。
    材质放在共享路径（如 ``/World/Looks/...``）下，所有环境共用一份。
    """
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, mat_path)
    pbr = UsdShade.Shader.Define(stage, mat_path + "/PBR")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)

    st_reader = UsdShade.Shader.Define(stage, mat_path + "/stReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    tex = UsdShade.Shader.Define(stage, mat_path + "/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_file)
    tex.CreateInput("st", Sdf.ValueTypeNames.TexCoord2f).ConnectToSource(st_reader.ConnectableAPI(), "result")
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    pbr.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

    # uv_scale != 1 时插入一个 UsdTransform2d 缩放 UV（避免纹理在一个面上被拉伸过多）
    if uv_scale != (1.0, 1.0):
        xf = UsdShade.Shader.Define(stage, mat_path + "/UVScale")
        xf.CreateIdAttr("UsdTransform2d")
        xf.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
        xf.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(*uv_scale))
        xf.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xf.ConnectableAPI(), "result")
    return mat


def _resolve_geometry_prims(stage, pattern: str, env_regex_ns: str | None = None) -> list:
    """把 ``{ENV_REGEX_NS}`` 形式的路径展开，返回其下所有 Cube/Mesh prim 的路径。

    这样 ``targets`` 既可以写具体的几何 prim（``.../Parcel/Body``），也可以写
    上层节点（``.../Table``），由本函数自动找到真正的几何体。

    ``find_matching_prims`` 要求路径是全局的（以 ``/`` 开头），所以必须先把
    ``{ENV_REGEX_NS}`` 宏替换成真实命名空间（``env.scene.env_regex_ns``，
    形如 ``/World/envs/env_[^/]+``）。
    """
    from pxr import Usd

    from isaaclab.sim.utils import find_matching_prims

    if "{ENV_REGEX_NS}" in pattern:
        if not env_regex_ns:
            raise ValueError("pattern contains {ENV_REGEX_NS} but env_regex_ns was not provided")
        pattern = pattern.replace("{ENV_REGEX_NS}", env_regex_ns)

    out: list[str] = []
    for root in find_matching_prims(pattern, stage):
        root_path = root.GetPath()
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.GetTypeName() in ("Cube", "Mesh"):
                out.append(prim.GetPath().pathString)
        if root.GetTypeName() in ("Cube", "Mesh"):
            out.append(root_path.pathString)
    return out


def apply_scene_textures(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    material_bindings: list | None = None,
    uv_scales: dict | None = None,
    disable_ground_lights: bool = True,
) -> None:
    """startup 事件：给盒子/标签/桌子/地面贴上程序化纹理材质。

    为什么用事件而不是写进资产文件：材质放在 ``/World/Looks`` 下**全局共享**，
    否则每个环境都会复制一份材质层级（1024 环境 = 上千份材质，白白增加开销）。
    UV 则由资产本身提供（``assets/parcel/parcel.usda`` 里 author 了 ``primvars:st``）。

    Args:
        env: 环境。
        env_ids: startup 事件传入（未使用）。
        material_bindings: ``[{"looks_name":..., "texture":..., "targets":[...],
            "roughness":..., "metallic":..., "uv_scale":...}, ...]``。
        uv_scales: 已废弃，保留兼容（UV 缩放写在 material_bindings 里）。
    """
    from pxr import Sdf, UsdShade

    if not material_bindings:
        return
    stage = env.scene.stage
    looks_root = "/World/Looks"

    # Isaac 自带的 GroundPlane 资产里藏着一盏 SphereLight（会把场景额外照亮，是
    # "整体过亮"的一个来源）。这里统一关掉，让光照完全由 env cfg 里的三盏灯决定。
    if disable_ground_lights:
        from pxr import Usd

        gp = stage.GetPrimAtPath("/World/GroundPlane")
        n_off = 0
        if gp.IsValid():
            for prim in Usd.PrimRange(gp, Usd.TraverseInstanceProxies()):
                if prim.GetTypeName() in ("SphereLight", "DomeLight", "DistantLight", "DiskLight", "RectLight"):
                    attr = prim.GetAttribute("inputs:intensity")
                    if attr and attr.IsValid():
                        attr.Set(0.0)
                        n_off += 1
        if n_off:
            print(f"[TEXTURE] 已关闭 GroundPlane 资产自带的 {n_off} 盏灯（避免额外照亮）", flush=True)

    for spec in material_bindings:
        tex = spec.get("texture")
        if tex is None:
            continue
        tex_path = _TEXTURE_DIR / f"{tex}.png"
        if not tex_path.exists():
            print(f"[TEXTURE] 跳过 {spec.get('looks_name')}: 纹理不存在 {tex_path}", flush=True)
            continue
        mat_path = f"{looks_root}/{spec['looks_name']}"
        mat = _make_texture_material(
            stage,
            mat_path,
            str(tex_path),
            roughness=float(spec.get("roughness", 0.85)),
            metallic=float(spec.get("metallic", 0.0)),
            uv_scale=tuple(spec.get("uv_scale", (1.0, 1.0))),
        )
        n_bound = 0
        for pattern in spec.get("targets", []):
            paths = (
                _resolve_geometry_prims(stage, pattern, env_regex_ns=getattr(env.scene, "env_regex_ns", None))
                if "{" in pattern
                else [pattern]
            )
            for target in paths:
                prim = stage.GetPrimAtPath(target)
                if not prim.IsValid():
                    continue
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
                n_bound += 1
        print(f"[TEXTURE] {spec['looks_name']}: 绑定到 {n_bound} 个 prim（纹理 {tex}.png）", flush=True)
