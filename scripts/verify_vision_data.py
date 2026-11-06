#!/usr/bin/env python3
# Copyright (c) 2026 The OpenArm Lab Developers.
#
# SPDX-License-Identifier: BSD-3-Clause

"""校验 ``collect_vision_data.py`` 产出的数据集：图像与位姿真值是否**真的对齐**、取景是否够用。

为什么要单独校验
----------------
"图像 + 真值"数据集最危险的失败模式不是崩溃，而是**静默错位**：相机外参/内参或
坐标约定搞错时，训练 loss 照样下降，CNN 却学到错误的映射。所以必须用与渲染管线
无关的独立证据来验证。

三项检验
--------
1. **凸起点测（主判据）**：由 ``parcel_pos_w/quat_w`` 与相机内外参算出盒子 8 个角点的
   像素坐标；取投影中心那一行、在预测轮廓**外** 15 px 处各取一个左右参考点的深度。
   盒子比桌面凸出，故要求 ``depth(中心) < min(depth(左), depth(右)) − 20 mm``。
   成立即说明"投影中心确实落在一个凸出于周围表面的物体上"，也就是盒子 ——
   内参、外参、四元数约定、真值四者自洽。

   .. note::
      为什么不用更容易想到的两种"包围盒 IoU"：
      1. **颜色掩码**会误判 —— 盒子翻正后顶面是**白色标签**（R−B≈0，与桌面无法区分），
         只有翻正前才是棕色纸板（R−B≈68）。
      2. **深度凸起分割**只覆盖盒子的**顶面**（底面贴着桌面、相对两侧桌面并不凸出），
         顶面比底面远、投影更小，故其包围盒比 8 角点轮廓窄约 30 px，仍会产生虚假偏差。
      包围盒 IoU 仍会打印，仅作参考。
2. **深度一致性（辅判据）**：盒子**近表面**应比中心更靠近相机，故
   ``0 ≤ z_center − depth@(u,v) ≤ 半对角线``。单侧容差，避免把"中心 vs 表面"的
   正常偏差误判成错位。
3. **取景充足度**：逐帧盒子轮廓在画面内的可见面积占比。占比低说明取景过窄，
   位姿 CNN 拿不到完整证据 —— 这是**数据可用性**问题，不是正确性问题。

用法::

    python3 scripts/verify_vision_data.py <run_dir>       # 宿主 python3 即可，不需要 Isaac

旧版本数据集（无 ``parcel_pos_w`` / ``*_parcel_visible_frac`` 字段）也能校验：
缺的字段现场由投影重算。
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw

# 判定"该像素打在盒子上"的阈值：盒子是棕色 => R 明显大于 B；桌面/地面是中性灰
_BOX_RB_MIN = 20.0
# 遮挡超过此比例即认为该帧「测不了」，不参与判据 1/2 的统计
_OCCL_MAX = 0.2
_CORNER_SIGNS = np.array(
    [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
)
_CORNER_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]


def rot_of_quat(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) 四元数 -> 3x3 旋转矩阵。"""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def project(points_w: np.ndarray, cam_pos_w: np.ndarray, cam_quat_ros: np.ndarray, K: np.ndarray) -> np.ndarray:
    """世界点 (M,3) -> 像素 (M,2)，用 ROS 光学约定（x 右 / y 下 / z 前）。"""
    p_cam = (points_w - cam_pos_w) @ rot_of_quat(cam_quat_ros)
    z = p_cam[:, 2]
    safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    return np.stack([K[0, 0] * p_cam[:, 0] / safe + K[0, 2], K[1, 1] * p_cam[:, 1] / safe + K[1, 2]], axis=-1)


def bbox_of(uv: np.ndarray) -> tuple[float, float, float, float]:
    return (float(uv[:, 0].min()), float(uv[:, 0].max()), float(uv[:, 1].min()), float(uv[:, 1].max()))


def iou(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[2], b[2])
    ix1, iy1 = min(a[1], b[1]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


def visible_fraction(uv: np.ndarray, w: int, h: int) -> float:
    """角点投影包围盒落在画面内的面积占比。"""
    full = (uv[:, 0].max() - uv[:, 0].min()) * (uv[:, 1].max() - uv[:, 1].min())
    if full <= 1e-6:
        return 0.0
    cu0, cu1 = max(float(uv[:, 0].min()), 0.0), min(float(uv[:, 0].max()), float(w))
    cv0, cv1 = max(float(uv[:, 1].min()), 0.0), min(float(uv[:, 1].max()), float(h))
    return max(0.0, cu1 - cu0) * max(0.0, cv1 - cv0) / full


def _proud_point_test(
    depth_m: np.ndarray, u_c: float, v_c: float, pb: tuple, z_center: float,
    offsets: tuple = (15, 25, 35), margin_m: float = 0.02,
) -> float:
    """凸起点测：投影中心是否落在凸出于周围表面的物体（盒子）上。

    在投影中心的同一行、预测轮廓**左右外侧**各取若干个参考点（离开轮廓 15/25/35 px），
    分别求中位数作为两侧的背景深度，取**较大者**作为基准：要求
    ``depth(中心) < 基准 - margin_m``。

    为什么要取较大者而不是较小者：翻转过程中手指/夹爪经常紧贴盒子两侧，落在参考点上
    的东西比桌面**更近**；取较小者会误判失败（这是实际踩过的坑，失败率会虚高到 ~30%）。
    取较大者只需保证"至少有一侧是桌面"，即可证明投影中心确实站在一个凸起物上。
    """
    h, w = depth_m.shape
    vi = int(round(v_c))
    if not (0 <= vi < h):
        return float("nan")
    ui = int(round(u_c))
    if ui < 0 or ui >= w:
        return float("nan")
    d_c = float(depth_m[vi, ui])
    if d_c <= 0:
        return float("nan")
    sides = []
    for sign, edge in ((-1, pb[0]), (1, pb[1])):
        vals = []
        for off in offsets:
            uu = int(round(edge)) + sign * off
            if 0 <= uu < w:
                d = float(depth_m[vi, uu])
                if d > 0:
                    vals.append(d)
        if vals:
            sides.append(float(np.median(vals)))
    if not sides:
        return float("nan")
    return 1.0 if (max(sides) - d_c) > margin_m else 0.0


def _proud_object_bbox(
    depth_m: np.ndarray, pb: tuple, w: int, h: int, pad: int = 30, margin_m: float = 0.02
) -> tuple | None:
    """用深度把"比周围表面凸出的物体"分割出来，返回其包围盒。

    做法：在预测轮廓外扩 ``pad`` 像素的窗口内，取**左右两侧**环带深度的中位数作为
    背景参考；窗口内比背景近 ``margin_m`` 以上的像素即物体。这样对颜色、光照、
    以及"顶面是白标签还是棕纸板"都免疫。

    .. note::
       环带**只取左右**、不取上下：上侧环带落在更远的背景上、下侧落在更近的桌面上，
       两者混在一起会让背景参考产生偏差，进而把盒子的下半部分漏掉（表现为投影包围盒
       与分割包围盒之间有十几像素的**垂直**偏移）。左右环带与盒子大致同深度，无此偏差。
    """
    u0, u1 = int(max(pb[0] - pad, 0)), int(min(pb[1] + pad, w - 1))
    v0, v1 = int(max(pb[2] - pad, 0)), int(min(pb[3] + pad, h - 1))
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None
    win = depth_m[v0 : v1 + 1, u0 : u1 + 1]
    valid = win > 0
    if valid.sum() < 50:
        return None
    inner = np.zeros_like(valid)
    iu0, iu1 = int(max(pb[0], 0)) - u0, int(min(pb[1], w - 1)) - u0
    iv0, iv1 = int(max(pb[2], 0)) - v0, int(min(pb[3], h - 1)) - v0
    inner[max(iv0, 0) : iv1 + 1, max(iu0, 0) : iu1 + 1] = True
    side = np.zeros_like(valid)
    side[:, : max(iu0, 0)] = True
    side[:, iu1 + 1 :] = True
    ring = valid & ~inner & side
    if ring.sum() < 30:
        ring = valid & ~inner  # 左右太窄时退回整环
    if ring.sum() < 30:
        return None
    bg = float(np.median(win[ring]))
    mask = valid & (win < bg - margin_m)
    if mask.sum() < 20:
        return None
    ys, xs = np.nonzero(mask)
    return (float(xs.min() + u0), float(xs.max() + u0), float(ys.min() + v0), float(ys.max() + v0))


def _occlusion_ratio(depth_m: np.ndarray, uv: np.ndarray, z_center: float, half_diag_m: float) -> float:
    """预测轮廓内"比盒子近表面还近"的像素占比 —— 近似为被遮挡比例。

    盒子在``uv``（8 角点投影）围成的区域内，其最靠近相机的表面不会比
    ``z_center - half_diag`` 更近；若某像素明显更近，说明有东西（机械臂/夹爪）挡在
    盒子前面。
    """
    h, w = depth_m.shape
    u0, u1 = int(max(uv[:, 0].min(), 0)), int(min(uv[:, 0].max(), w - 1))
    v0, v1 = int(max(uv[:, 1].min(), 0)), int(min(uv[:, 1].max(), h - 1))
    if u1 - u0 < 2 or v1 - v0 < 2:
        return float("nan")
    patch = depth_m[v0 : v1 + 1, u0 : u1 + 1]
    valid = patch > 0
    if valid.sum() < 10:
        return float("nan")
    nearer = valid & (patch < z_center - half_diag_m - 0.01)
    return float(nearer.sum()) / float(valid.sum())


def depth_near(depth_mm: np.ndarray, u: float, v: float, r: int = 2) -> float:
    """取 (u, v) 邻域内最近的有效深度（米）；无有效值返回 nan。"""
    h, w = depth_mm.shape
    ui, vi = int(round(u)), int(round(v))
    u0, u1 = max(0, ui - r), min(w, ui + r + 1)
    v0, v1 = max(0, vi - r), min(h, vi + r + 1)
    if u0 >= u1 or v0 >= v1:
        return float("nan")
    patch = depth_mm[v0:v1, u0:u1]
    valid = patch[patch > 0]
    return float(valid.min()) / 1000.0 if valid.size else float("nan")


class FrameData:
    """逐帧采集到的可比较量。"""

    def __init__(self) -> None:
        self.iou: list[float] = []
        self.dev: list[tuple[float, float]] = []
        self.vis: list[float] = []
        self.depth_diff_mm: list[float] = []
        self.rgb_mean: list[float] = []
        self.uv: list[tuple[float, float] | None] = []
        self.occl: list[float] = []
        self.u_span_err: list[float] = []
        self.proud: list[float] = []
        self.n_occluded: int = 0


def main() -> None:
    parser = argparse.ArgumentParser(description="校验视觉-位姿数据集")
    parser.add_argument("run_dir", type=str, help="collect_vision_data.py 的输出目录")
    parser.add_argument("--sheet_cam", type=str, default="front", help="拼图用哪台相机")
    parser.add_argument("--sheet_n", type=int, default=25, help="拼图最多放几帧")
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir).expanduser().resolve()
    info = json.loads((run_dir / "dataset_info.json").read_text(encoding="utf-8"))
    f = np.load(run_dir / "frames.npz")
    n = len(f["frame_index"])
    cams = list(info["cameras"])
    size = info.get("parcel_size_m", [0.10, 0.10, 0.08])
    half = np.array(size, dtype=np.float64) / 2.0
    corners_local = _CORNER_SIGNS * half
    half_diag_mm = float(np.linalg.norm(half)) * 1000.0

    # 世界系盒子位置：新版有 parcel_pos_w；旧版用 env-local + 原点还原
    if "parcel_pos_w" in f:
        pos_w = f["parcel_pos_w"]
    elif "env_origin" in f:
        pos_w = f["parcel_pos_env"] + f["env_origin"]
    else:
        pos_w = f["parcel_pos_env"]  # 单 env、origin=(0,0,0)

    print("=" * 84)
    print(f"数据集: {run_dir}")
    print(f"  帧数 {n} | 相机 {cams} | 观测 {info['observation']['dim']} 维")
    print(f"  策略 {info['source_checkpoint']['path']}")
    print(f"  盒子尺寸 {size} m（半对角线 {half_diag_mm / 10:.1f} cm）")
    print("=" * 84)

    # ---------------------------------------------- 逐帧计算（三个判据共用角点投影）
    data: dict[str, FrameData] = {}
    for cam in cams:
        fd = FrameData()
        rgb_dir = run_dir / "images" / cam
        depth_dir = run_dir / "depth" / cam
        has_depth = depth_dir.exists()
        if not rgb_dir.exists():
            data[cam] = fd
            continue
        for i in range(n):
            fno = int(f["frame_index"][i])
            img = np.asarray(Image.open(rgb_dir / f"{fno:06d}.png"), dtype=np.float32)
            h_img, w_img = img.shape[0], img.shape[1]
            fd.rgb_mean.append(float(img.mean()))

            R = rot_of_quat(f["parcel_quat_w"][i])
            uv = project(pos_w[i] + corners_local @ R.T, f[f"{cam}_pos_w"][i],
                         f[f"{cam}_quat_w_ros"][i], f[f"{cam}_K"][i])
            pb = bbox_of(uv)
            z_px = float(f[f"{cam}_parcel_depth"][i])
            fd.vis.append(visible_fraction(uv, w_img, h_img))
            fd.uv.append(((pb[0] + pb[1]) / 2, (pb[2] + pb[3]) / 2))

            # 判据 1：只在"轮廓完整落在画面内"时比较，避免把取景问题当成对齐问题。
            # 分割用深度（盒子比周围桌面凸出），不用颜色（见模块 docstring 的说明）。
            clipped = pb[0] < 0 or pb[2] < 0 or pb[1] > w_img or pb[3] > h_img
            if not clipped and has_depth:
                dmap = np.asarray(Image.open(depth_dir / f"{fno:06d}.png"), dtype=np.float32) / 1000.0
                bb = _proud_object_bbox(dmap, pb, w_img, h_img)
                if bb is not None:
                    fd.iou.append(iou(pb, bb))
                    fd.dev.append(((pb[0] + pb[1]) / 2 - (bb[0] + bb[1]) / 2,
                                   (pb[2] + pb[3]) / 2 - (bb[2] + bb[3]) / 2))
                    # 水平跨度无偏（盒子左右侧面在任意高度都"凸出于"两侧桌面），
                    # 故以它作为对齐的主指标；垂直方向受"底面不凸出"影响有系统性偏差。
                    fd.u_span_err.append(0.5 * (abs(pb[0] - bb[0]) + abs(pb[1] - bb[1])))

            # 先算遮挡：盒子被机械臂/夹爪挡住的帧，判据 1/2 在物理上就测不了
            # （投影中心落在手臂上而不是盒子上），必须剔除而不是算作「对齐失败」。
            occl = float("nan")
            if has_depth:
                dmap = np.asarray(Image.open(depth_dir / f"{fno:06d}.png"), dtype=np.float32) / 1000.0
                zc = float(f[f"{cam}_parcel_depth"][i])
                occl = _occlusion_ratio(dmap, uv, zc, half_diag_mm / 1000.0)
                fd.occl.append(occl)
                if not (np.isfinite(occl) and occl <= _OCCL_MAX):
                    fd.n_occluded += 1

            measurable = has_depth and (not np.isfinite(occl) or occl <= _OCCL_MAX)
            if measurable:
                dmap = np.asarray(Image.open(depth_dir / f"{fno:06d}.png"), dtype=np.float32) / 1000.0
                # 判据 1 主指标：凸起点测 —— 投影中心应比同一行、轮廓外的左右参考点更近
                fd.proud.append(_proud_point_test(dmap, float(f[f"{cam}_parcel_uv"][i, 0]),
                                                 float(f[f"{cam}_parcel_uv"][i, 1]), pb, z_px))

            # 判据 2：深度
            if measurable and f"{cam}_parcel_depth" in f:
                z = float(f[f"{cam}_parcel_depth"][i])
                if np.isfinite(z) and z > 0:
                    d = depth_near(np.asarray(Image.open(depth_dir / f"{fno:06d}.png"), dtype=np.float32),
                                   float(f[f"{cam}_parcel_uv"][i, 0]), float(f[f"{cam}_parcel_uv"][i, 1]))
                    if np.isfinite(d):
                        fd.depth_diff_mm.append((z - d) * 1000.0)
        data[cam] = fd

    # --------------------------------------------------- 判据 1
    print("\n【判据 1】凸起点测：投影中心是否落在凸出于周围表面的物体（盒子）上")
    print(f"  {'相机':<12} {'可测帧':>7} {'剔除(遮挡)':>11} {'凸起成立率':>11} {'水平中心偏差中位':>17}")
    for cam in cams:
        fd = data[cam]
        proud = np.asarray([x for x in fd.proud if np.isfinite(x)])
        if proud.size:
            uu = np.abs(np.asarray(fd.dev))[:, 0] if fd.dev else np.array([np.nan])
            print(f"  {cam:<12} {proud.size:7d} {fd.n_occluded:11d} {100 * proud.mean():10.1f}% {np.median(uu):16.2f}px")
        else:
            print(f"  {cam:<12} {0:7d} {data[cam].n_occluded:11d} {'—':>11}")

    # --------------------------------------------------- 判据 2
    print("\n【判据 2】投影像素处深度 vs 盒子中心深度（正 = 表面更近；应落在 0 ~ 半对角线）")
    print(f"  {'相机':<12} {'可测帧':>7} {'中位偏差':>10} {'符合率':>8}   说明")
    for cam in cams:
        fd = data[cam]
        if fd.depth_diff_mm:
            a = np.asarray(fd.depth_diff_mm)
            ok = float(((a >= -5.0) & (a <= half_diag_mm)).mean()) * 100.0
            note = "正常（差值 ≈ 盒子半厚度，符合预期）" if ok > 80 else "异常：检查深度标定/投影"
            print(f"  {cam:<12} {len(a):7d} {np.median(a):9.1f}mm {ok:7.1f}%   {note}")
        else:
            print(f"  {cam:<12} {0:7d} {'—':>10}")

    # --------------------------------------------------- 判据 3
    print("\n【判据 3】取景充足度（盒子轮廓在画面内的可见面积占比）")
    print(f"  {'相机':<12} {'完整可见>95%':>13} {'部分50-95%':>11} {'截断<50%':>10} {'平均可见':>9}")
    for cam in cams:
        vis = np.asarray(data[cam].vis)
        if vis.size:
            print(f"  {cam:<12} {100.0 * (vis > 0.95).mean():12.1f}% "
                  f"{100.0 * ((vis > 0.5) & (vis <= 0.95)).mean():10.1f}% "
                  f"{100.0 * (vis <= 0.5).mean():9.1f}% {100.0 * vis.mean():8.1f}%")
        else:
            print(f"  {cam:<12} {'—':>13}")

    print("\n【判据 4】遮挡估计（预测轮廓内『比盒子近表面更近』的像素占比；被手臂/夹爪挡住的比例）")
    print(f"  {'相机':<12} {'可测帧':>7} {'遮挡中位':>10} {'>20% 遮挡帧':>12}")
    for cam in cams:
        oc = np.asarray([x for x in data[cam].occl if np.isfinite(x)])
        if oc.size:
            print(f"  {cam:<12} {oc.size:7d} {100 * np.median(oc):9.1f}% {100 * (oc > 0.2).mean():11.1f}%")
        else:
            print(f"  {cam:<12} {'—':>7}")

    print("\n【结论】")
    fv = np.asarray(data.get("front", FrameData()).vis)
    if fv.size:
        cut = 100.0 * (fv <= 0.5).mean()
        if cut > 20.0:
            print(f"  ! 前视取景过窄：{cut:.0f}% 的帧盒子被画面截断 —— 采集正式数据集前应先改相机"
                  "（后移 / 抬高 / 换更广焦距）或缩短采集窗口")
        else:
            print(f"  ✓ 前视取景充足（截断帧 {cut:.0f}%）")
    ok_cams = []
    for cam in cams:
        fd = data[cam]
        if not fd.u_span_err:
            continue
        dep_ok = 1.0
        if fd.depth_diff_mm:
            a = np.asarray(fd.depth_diff_mm)
            dep_ok = float(((a >= -5.0) & (a <= half_diag_mm)).mean())
        proud = np.asarray([x for x in fd.proud if np.isfinite(x)])
        proud_ok = float(proud.mean()) if proud.size else 0.0
        if proud_ok > 0.8 and dep_ok > 0.8:
            ok_cams.append(cam)
    if ok_cams:
        print(f"  ✓ 对齐校验通过（凸起成立率 >80% 且深度符合率 >80%）：{ok_cams} —— 图像与位姿真值可信")
    else:
        print("  ✗ 对齐校验未通过 —— 检查内外参/坐标约定")
    for cam in cams:
        vis = np.asarray(data[cam].vis)
        if vis.size and vis.mean() < 0.05 and cam != "front":
            print(f"  ! {cam} 平均可见 {100 * vis.mean():.1f}% —— 该相机看不到盒子（已知：腕部相机朝下看指尖）")

    # --------------------------------------------------- 真值覆盖度
    print("\n【真值覆盖度】")
    lu = f["label_up"]
    print(f"  label_up 范围 [{lu.min():+.3f}, {lu.max():+.3f}] | 均值 {lu.mean():+.3f} "
          f"| 达标(cos15°)占比 {100.0 * (lu > np.cos(np.radians(15))).mean():.0f}%")
    p = f["parcel_pos_env"]
    print(f"  盒子位置 x[{p[:, 0].min():.3f},{p[:, 0].max():.3f}] y[{p[:, 1].min():.3f},{p[:, 1].max():.3f}] "
          f"z[{p[:, 2].min():.3f},{p[:, 2].max():.3f}]")
    print(f"  在桌上 {100.0 * f['on_table'].mean():.0f}% | 回合边界帧 {int(f['reset_flag'].sum())} 帧")
    for cam in cams:
        m = np.asarray(data[cam].rgb_mean)
        if m.size:
            print(f"  {cam:<12} RGB 均值 {m.mean():6.1f}（min {m.min():.1f}）{'  !! 接近全黑' if m.mean() < 5 else ''}")

    # --------------------------------------------------- 拼图
    rgb_dir = run_dir / "images" / args.sheet_cam
    if rgb_dir.exists() and args.sheet_cam in data:
        step = max(1, n // args.sheet_n)
        idxs = list(range(0, n, step))[: args.sheet_n]
        im0 = Image.open(rgb_dir / f"{int(f['frame_index'][idxs[0]]):06d}.png")
        tw, th = im0.width // 2, im0.height // 2
        cols = 5
        rows = (len(idxs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * tw, rows * (th + 12)), (25, 25, 25))
        d = ImageDraw.Draw(sheet)
        cam = args.sheet_cam
        for k, i in enumerate(idxs):
            fno = int(f["frame_index"][i])
            im = Image.open(rgb_dir / f"{fno:06d}.png").resize((tw, th))
            di = ImageDraw.Draw(im)
            R = rot_of_quat(f["parcel_quat_w"][i])
            uv = project(pos_w[i] + corners_local @ R.T, f[f"{cam}_pos_w"][i],
                         f[f"{cam}_quat_w_ros"][i], f[f"{cam}_K"][i]) / 2.0
            pts = [(float(u), float(v)) for u, v in uv]
            for a, b in _CORNER_EDGES:
                di.line([pts[a], pts[b]], fill=(0, 255, 0), width=1)
            cx, cy = (k % cols) * tw, (k // cols) * (th + 12)
            sheet.paste(im, (cx, cy + 12))
            d.text((cx + 3, cy + 1),
                   f"#{fno} up={float(f['label_up'][i]):+.2f} vis={100 * data[cam].vis[i]:.0f}%",
                   fill=(255, 255, 0))
        out = run_dir / "contact_sheet.png"
        sheet.save(out)
        print(f"\n【拼图】{out}  （{len(idxs)} 帧 {cam}，绿框 = 盒子 8 角点投影）")
    print("=" * 84)


if __name__ == "__main__":
    main()
