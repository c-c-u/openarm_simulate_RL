# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""把机器人摆成指定姿态并截一张前视相机的图（同时输出数值验证）。

为什么需要"数值验证"
--------------------
截图脚本不能只把 PNG 丢出来就算完 —— 读图的人（包括 AI）未必能看图，也未必分得清
"手臂水平前伸"和"手臂略微下垂"。因此本脚本在截图的同时打印实测的末端位置：
两腕高度是否一致（水平）、是否在肩高（不下垂）、是否向前伸出（x 大）、是否左右对称。

姿态 ``forward``（双手水平前伸）
--------------------------------
两腕目标：``x=0.45``（前伸）、``z = 肩高 0.698``（水平）、``y = ±0.12``（对称微张）；
腕部朝向取"工具轴指向正前方 + 夹爪上方向朝上"，即 link7 局部 +Z（指向指尖，见
``WRIST_CAM_LOCAL`` 注释的标定）映射到世界 +X。夹爪保持张开（0.044）。

注意：本脚本**绕过 action manager**，直接给关节写位置目标（这样能摆出 RL 动作空间
摆不出的姿态），因此用 ``sim.step()`` 而不是 ``env.step()``。

踩过的坑（都验证过，别重犯）
----------------------------
1. **雅可比**用 ``robot.data.body_link_jacobian_w``（已按用户 body/joint 排序归一化，
   固定基座下 body 索引 = ``body_idx - 1``）。``root_physx_view.get_jacobians()``
   返回 **warp 数组**，不能按 ``[:, idx, :, joint_index_list]`` 索引。
2. **pose 模式的 IK 命令必须是 (N, 7) 的 ``[pos(3) | quat(4)]`` 拼接**，拆成两个参数
   会报 size 3 vs 7 的广播错（第二个参数会被当成 ``ee_pos``）。
3. 绕过 action manager 后**必须自己 ``robot.write_data_to_sim()``**，否则
   ``set_joint_position_target`` 只写内部缓冲、机械臂完全不动（表现为误差恒定不变）。
4. **不要用 PD 跟随做 IK**：重力 + 位置伺服会把手臂往下拽，实测 90 步只把误差从
   75 cm 收到 33 cm 且末端下垂。改用**运动学 IK**（``write_joint_state_to_sim``）
   几十步即收敛；剩下的几厘米残差来自**关节限位**，用两腕高度闭环修正压掉。
5. **不要在运行时用 ``set_world_poses`` 改相机机位**：fabric 模式下相机 prim 的 USD
   变换不同步，``data.pos_w`` 会读到陈旧值（投影算出来还在原地、图像却已变），
   无法验证。要用 ``--cam_pos/--cam_target`` 在 **``gym.make`` 之前**覆盖。
6. **臂展上限**：肩在 ``(0, ±0.093, 0.698)``，腕部目标离肩超过约 0.42 m 就接近极限，
   IK 收敛不良且两臂会落到不同局部解、高度差不齐（实测差 8 cm）。

用法::

    $ISAAC_ROOT/python.sh scripts/shot_pose.py                    # 场景相机原样
    $ISAAC_ROOT/python.sh scripts/shot_pose.py --aim_at_arms      # 额外再拍一张：相机临时对准手臂
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT = _REPO_ROOT / "models" / "openarm_flip_state62_best.pt"
_DEFAULT_OUT = pathlib.Path.home() / "openarm_shots"

parser = argparse.ArgumentParser(description="摆姿态 + 截前视相机图（附数值验证）")
parser.add_argument("--checkpoint", type=str, default=str(_DEFAULT_CHECKPOINT))
parser.add_argument("--pose", type=str, default="forward", choices=["forward", "forward_bent"],
                    help="forward = 双臂水平前伸；forward_bent = 前伸但肘部略弯")
parser.add_argument("--out", type=str, default=str(_DEFAULT_OUT))
parser.add_argument("--tag", type=str, default="arms_forward")
parser.add_argument("--trim_every", type=int, default=25,
                    help="每多少步做一次两腕高度闭环修正")
parser.add_argument("--steps", type=int, default=250, help="IK 迭代步数（运动学 IK；目标越接近臂展极限越难收敛）")
parser.add_argument("--target_x", type=float, default=0.36,
                    help="腕部前伸目标 x（米）。注意臂展：肩在 (0,±0.031,0.698)，"
                         "目标离肩超过 ~0.42 m 就接近极限，IK 会收敛不良、两臂不对称")
parser.add_argument("--target_y", type=float, default=0.13, help="腕部左右目标 |y|（米）")
parser.add_argument("--target_z", type=float, default=0.698, help="腕部高度目标 z（= 肩高，水平）")
parser.add_argument("--aim_at_arms", action="store_true",
                    help="把相机改成能拍到前伸手臂的机位（仅为出图；在构造期改，保证内外参一致）")
parser.add_argument("--cam_pos", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="覆盖前视相机位置（env-local）")
parser.add_argument("--cam_target", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="覆盖前视相机注视点（env-local）")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

_ISAAC_ROOT = pathlib.Path(
    os.environ.get("ISAAC_ROOT", "/home/blanc/isaacsim/isaac-sim-standalone-6.0.1-linux-x86_64")
)
for _rel in ("extsDeprecated/omni.isaac.ml_archive/pip_prebundle", "exts/isaacsim.pip.newton/pip_prebundle"):
    _pre = _ISAAC_ROOT / _rel
    if _pre.exists() and str(_pre) not in sys.path:
        sys.path.insert(0, str(_pre))
sys.path.insert(0, str(_REPO_ROOT.parent))

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "width": 320, "height": 240, "enable_cameras": True})
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from PIL import Image  # noqa: E402

import openarm_lab  # noqa: F401,E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.utils.string import string_to_callable  # noqa: E402
from openarm_lab.openarm_flip.checkpoint_utils import make_env_cfg  # noqa: E402
from openarm_lab.openarm_flip.openarm_teleop_env_cfg import (  # noqa: E402
    EE_TO_FINGERTIP,
    GRIPPER_OPEN_VALUE,
)

TASK = "OpenArm-Flip-State-v0"


def quat_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 旋转矩阵 -> 四元数 (x, y, z, w)。"""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2.0
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = 0.25 * s
        q[3] = (R[k, j] - R[j, k]) / s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    return tuple(float(v) for v in q)


# 工具轴（link7 局部 +Z）-> 世界 +X；夹爪"上方"（局部 +X）-> 世界 +Z。
# 于是指尖朝正前方、夹爪上下方向竖直 => 双臂水平前伸的自然姿态。
_R_TOOL_FORWARD = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def main() -> None:
    import importlib.metadata as metadata

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    spec = gym.spec(TASK)
    # 只创建前视相机（lean_cameras）+ 关掉图像观测，保持 62 维状态；分辨率仍是 FRONT_CAM_*
    env_cfg = make_env_cfg(
        string_to_callable(spec.kwargs["env_cfg_entry_point"]),
        checkpoint,
        enable_cameras=True,
        lean_cameras=True,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args.device

    # 相机机位覆盖**必须在 gym.make 之前**：运行时用 set_world_poses 改不行 ——
    # 相机 prim 在 fabric 模式下不会同步 USD 变换，data.pos_w/quat 会读到陈旧值，
    # 于是"投影算出来还在原地、图像却已经变了"，无法验证（踩过这个坑）。
    from openarm_lab.openarm_flip.openarm_flip_env_cfg import (  # noqa: E402
        _camera_look_at_quat_xyzw,
    )

    target_pos = (args.target_x, 0.0, args.target_z)
    if args.aim_at_arms or args.cam_pos is not None or args.cam_target is not None:
        cpos = tuple(args.cam_pos) if args.cam_pos is not None else (
            target_pos[0] + 0.30, target_pos[1] - 0.62, target_pos[2] + 0.32
        )
        ctgt = tuple(args.cam_target) if args.cam_target is not None else (0.26, 0.0, 0.70)
        env_cfg.scene.front_cam.offset.pos = cpos
        env_cfg.scene.front_cam.offset.rot = _camera_look_at_quat_xyzw(cpos, ctgt)
        print(f"[SHOT] 相机覆盖为 pos={tuple(round(v, 3) for v in cpos)} "
              f"target={tuple(round(v, 3) for v in ctgt)}", flush=True)

    env = gym.make(TASK, cfg=env_cfg)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    device = unwrapped.device
    sim = unwrapped.sim
    dt = unwrapped.step_dt
    env.reset()

    # 让渲染先跑几帧（RTX 降噪 + 相机数据就绪）
    zero = torch.zeros(1, unwrapped.action_manager.total_action_dim, device=device)
    for _ in range(10):
        env.step(zero)

    body_names = list(robot.data.body_names)
    print(f"[SHOT] 机器人 body 数 {len(body_names)}；末端 link7 索引:")
    arm_ids, ee_ids, finger_ids = {}, {}, {}
    for side in ("left", "right"):
        ids, _ = robot.find_joints([f"openarm_{side}_joint[1-7]"], preserve_order=True)
        arm_ids[side] = ids
        bid, _ = robot.find_bodies([f"openarm_{side}_link7"])
        ee_ids[side] = bid[0]
        fid, _ = robot.find_joints([f"openarm_{side}_finger_joint[1-2]"])
        finger_ids[side] = fid
        print(f"    {side:5s} link7 body_idx={bid[0]} ({body_names[bid[0]]}), arm joints={ids}")

    # ---------------- 臂展诊断（决定目标点是否可达）----------------
    print("\n[SHOT] reset 后的实测几何（env-local）：")
    for side in ("left", "right"):
        names = [f"openarm_{side}_link{i}" for i in (1, 2, 3, 4, 5, 6, 7)]
        row = []
        for nm in names:
            ids, _ = robot.find_bodies([nm])
            if ids:
                row.append((nm.split("_")[-1], np.asarray(robot.data.body_pos_w.torch[0, ids[0]].cpu())))
        if row:
            shoulder = row[0][1]
            wrist = row[-1][1]
            print(f"    {side:5s} " + "  ".join(f"{nm}=({q[0]:+.3f},{q[1]:+.3f},{q[2]:+.3f})" for nm, q in row))
            print(f"         肩->腕直线距离 {np.linalg.norm(wrist - shoulder):.3f} m "
                  f"（这就是当前姿态下的臂展；伸直后的最大臂展更大）")

    # ---------------- IK 目标 ----------------
    z_target = args.target_z
    ik, targets = {}, {}
    for side, sgn in (("left", +1.0), ("right", -1.0)):
        ik[side] = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                # adaptive_dls：靠近奇异位形时自动加大阻尼，比固定 lambda 更能收敛到零残差
                ik_method="adaptive_dls",
                ik_params={"lambda_min": 0.03, "lambda_max": 0.2, "k_val": 1.0},
            ),
            num_envs=1, device=device,
        )
        targets[side] = torch.tensor(
            [[args.target_x, sgn * args.target_y, z_target]], dtype=torch.float32, device=device
        )
        if args.pose == "forward_bent":
            targets[side][0, 2] += 0.04  # 肘部略弯：抬高一点，避免接近奇异位形
    target_quat = torch.tensor([quat_from_matrix(_R_TOOL_FORWARD)], dtype=torch.float32, device=device)

    print(f"[SHOT] 姿态 {args.pose}：腕部目标 x={args.target_x} y=±{args.target_y} z={z_target} (肩高)")

    # ---------------- IK 迭代（运动学：直接把 IK 解写入关节状态）----------------
    # 为什么不用 PD 跟随：截图要的是"摆到某个姿态"，不是动力学过程。一开始用
    # set_joint_position_target + write_data_to_sim 让它跟随，90 步只把误差从 75 cm
    # 收到 33 cm —— 重力 + 位置伺服把手臂往下拽，收敛太慢且末端下垂。直接写关节状态
    # （write_joint_state_to_sim）等效于运动学 IK，几十步内收敛且不受重力影响。
    jac_idx = {s: ee_ids[s] - 1 for s in ee_ids}  # 固定基座的 jacobian body 索引 = body_idx - 1
    for side in ("left", "right"):
        robot.write_joint_state_to_sim(
            torch.full((1, len(finger_ids[side])), GRIPPER_OPEN_VALUE, device=device),
            torch.zeros((1, len(finger_ids[side])), device=device),
            joint_ids=finger_ids[side],
        )
    def ee_z() -> dict:
        return {sd: float(robot.data.body_pos_w.torch[0, ee_ids[sd], 2]) for sd in ("left", "right")}

    for it in range(args.steps):
        # 高度闭环修正：两臂各自卡在不同关节限位上时，腕部高度会差几厘米。
        # 用实测高度差反向修目标 z（各承担一半），几个回合就能把差值压到毫米级。
        if it and it % args.trim_every == 0:
            z = ee_z()
            dz = z["left"] - z["right"]
            if abs(dz) > 5e-4:
                targets["left"][0, 2] -= 0.5 * dz
                targets["right"][0, 2] += 0.5 * dz
        for side in ("left", "right"):
            # 用 ArticulationData 的 link 雅可比（用户 body/joint 排序，已归一化）：
            # 不要用 root_physx_view.get_jacobians()，它返回 warp 数组且未做排序归一。
            jac = robot.data.body_link_jacobian_w.torch[:, jac_idx[side], :, arm_ids[side]]
            # pose 模式的 set_command 要 (N, 7) 的 [pos(3) | quat(4)] 拼接命令
            ik[side].set_command(torch.cat([targets[side], target_quat], dim=-1))
            cmd = ik[side].compute(
                robot.data.body_pos_w.torch[:, ee_ids[side]],
                robot.data.body_quat_w.torch[:, ee_ids[side]],
                jac,
                robot.data.joint_pos.torch[:, arm_ids[side]],
            )
            robot.write_joint_state_to_sim(
                cmd, torch.zeros_like(cmd), joint_ids=arm_ids[side]
            )
        sim.step()
        unwrapped.scene.update(dt)
        if it == 0 or (it + 1) % 50 == 0:
            e = [float((robot.data.body_pos_w.torch[0, ee_ids[s]] - targets[s][0]).norm()) for s in ("left", "right")]
            print(f"    iter {it + 1:3d}: 末端误差 左 {e[0] * 100:5.1f} cm 右 {e[1] * 100:5.1f} cm", flush=True)

    # ---------------- 数值验证 ----------------
    p = {s: robot.data.body_pos_w.torch[0, ee_ids[s]].detach().cpu().numpy() for s in ("left", "right")}
    q = {s: robot.data.body_quat_w.torch[0, ee_ids[s]].detach().cpu().numpy() for s in ("left", "right")}
    tool = {}
    for s in ("left", "right"):
        x, y, z, w = q[s]
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        tool[s] = R[:, 2]  # link7 局部 +Z = 指向指尖
    tip = {s: p[s] + EE_TO_FINGERTIP * tool[s] for s in ("left", "right")}

    print("\n[SHOT] 实测姿态验证（env-local，机器人原点在 (0,0,0)）")
    for s in ("left", "right"):
        print(f"    {s:5s} 腕 link7 = ({p[s][0]:+.3f}, {p[s][1]:+.3f}, {p[s][2]:+.3f})   "
              f"指尖 = ({tip[s][0]:+.3f}, {tip[s][1]:+.3f}, {tip[s][2]:+.3f})")
    resid = max(float(np.linalg.norm(p["left"] - np.asarray(targets["left"][0].cpu()))),
                float(np.linalg.norm(p["right"] - np.asarray(targets["right"][0].cpu()))))
    print(f"    IK 残差（到目标点）= {resid * 100:.1f} cm"
          f"{'  <- 偏大，目标可能接近臂展极限，可减小 --target_x' if resid > 0.02 else ''}")
    dz = abs(p["left"][2] - p["right"][2])
    print(f"    两腕高度差 = {dz * 1000:.1f} mm  -> {'水平 ✓' if dz < 0.01 else '不水平 ✗'}")
    print(f"    腕部高度 = {p['left'][2]:.3f} m vs 肩高 {args.target_z:.3f} m "
          f"(差 {(p['left'][2] - args.target_z) * 1000:+.1f} mm)")
    print(f"    前伸量 x = {p['left'][0]:.3f} m；指尖 x = {tip['left'][0]:.3f} m")
    print(f"    左右对称性 |y_l + y_r| = {abs(p['left'][1] + p['right'][1]) * 1000:.1f} mm")
    print(f"    工具轴（应为 +X 方向）: 左 {np.round(tool['left'], 3)} 右 {np.round(tool['right'], 3)}")

    # ---------------- 截图 ----------------
    out = pathlib.Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    def grab(name: str) -> np.ndarray:
        cam = unwrapped.scene.sensors["front_cam"]
        rgb = cam.data.output["rgb"]
        rgb = rgb.torch if hasattr(rgb, "torch") else rgb
        arr = np.asarray(rgb.cpu())[0]
        path = out / f"{args.tag}_{name}.png"
        Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)
        print(f"[SHOT] 已保存 {path}  ({arr.shape[1]}x{arr.shape[0]}, 均值 {arr.mean():.1f})")
        return arr

    grab("scene_cam")

    # 手臂在画面里的位置（相机固定看桌面时，水平前伸的手臂很可能在画面上方之外）
    def project(points_w, cam_pos_w, quat_ros, K):
        x, y, z, w = quat_ros
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        pc = (np.asarray(points_w) - np.asarray(cam_pos_w)) @ R
        return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], axis=-1)

    cam = unwrapped.scene.sensors["front_cam"]
    cam_pos = np.asarray(cam.data.pos_w.torch[0].cpu())
    cam_quat = np.asarray(cam.data.quat_w_ros.torch[0].cpu())
    K = np.asarray(cam.data.intrinsic_matrices.torch[0].cpu())
    W, H = cam.cfg.width, cam.cfg.height
    wrist_pts = np.array([p["left"], p["right"], tip["left"], tip["right"]])
    uv = project(wrist_pts, cam_pos, cam_quat, K)
    print("\n[SHOT] 手臂在该相机画面里的投影（u, v）与画面尺寸 %dx%d：" % (W, H))
    for name, xy in zip(("左腕", "右腕", "左指尖", "右指尖"), uv):
        inside = 0 <= xy[0] < W and 0 <= xy[1] < H
        print(f"    {name}: ({xy[0]:7.1f}, {xy[1]:7.1f})  {'在画面内' if inside else '★ 画面外'}")

    # 注：额外机位请用 --aim_at_arms / --cam_pos / --cam_target 重跑（构造期生效，可验证）
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
