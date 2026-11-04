"""程序化生成仿真用的纹理（自包含：不下载外部素材，便于迁移到其他机器）。

生成的纹理都放在 ``assets/textures/`` 下，随仓库分发（每张几十~几百 KB）：

* ``cardboard.png``  —— 瓦楞纸**表面**（仅 ``--with-cardboard`` 时生成；盒子当前用纯色，见下）
* ``table.png``      —— 哑光层压板台面（浅灰白 + 细密颗粒 + 轻微划痕，无木纹）
* ``floor.png``      —— 中性地板（避免纯色平面造成的"塑料感"）

用法::

    python3 tools/make_textures.py            # 全部重新生成
    python3 tools/make_textures.py --seed 7   # 指定随机种子（可复现）

注意：**快递盒当前使用纯色**（``parcel.usda`` 的 ``primvars:displayColor``），不绑纹理 ——
纸箱纹理试过几轮（单向瓦楞条纹 / 各向同性带通噪声）观感反而更假、还容易做出木纹感。
`make_cardboard()` 保留下来，用 ``--with-cardboard`` 可重新生成备用。

设计教训
--------
``cardboard`` 一开始在外表面画了**单向的"瓦楞条纹"**，结果看起来像**木纹** ——
单向平行条纹本来就是木纹的特征。真实的瓦楞芯只在纸板**截面**可见，外表面是压平的
牛皮纸，特征是"各向同性的短纤维 + 颗粒"。所以这里改用「多个随机方向的弱正弦叠加
+ 大量随机方向短纤维」来构造，并用 :func:`verify_isotropy` 自检方向性。
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "assets" / "textures"
SIZE = 512


def _fbm(rng: np.random.Generator, size: int, octaves: int = 5, base: int = 4) -> np.ndarray:
    """分形噪声（多个尺度的平滑噪声叠加），取值约在 [-1, 1]。"""
    acc = np.zeros((size, size), dtype=np.float32)
    amp = 1.0
    total = 0.0
    for o in range(octaves):
        n = base * (2**o)  # 该尺度的网格点数
        grid = rng.random((n, n)).astype(np.float32)
        img = Image.fromarray((grid * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
        acc += amp * (np.asarray(img, dtype=np.float32) / 255.0 * 2.0 - 1.0)
        total += amp
        amp *= 0.55
    return acc / total


def _band_noise(
    rng: np.random.Generator, size: int, sigma_low: float, sigma_high: float
) -> np.ndarray:
    """各向同性的带通噪声（单位方差）。

    做法：白噪声分别做两次**各向同性**高斯模糊，相减得到"某个尺度区间"的结构。
    这是构造纸张纤维/斑驳的正确方式 —— 高斯模糊没有方向偏好。

    反面教材：最早用"多个随机方向的正弦叠加"来造纤维，加了 48 个方向看似各向同性，
    但**下采样后**残留的方向性会被放大（实测纹理文件 1.55，渲染到盒面上变成 4.35，
    看起来就像木纹）。改成带通噪声后渲染前后都接近 1.0。
    """
    n = rng.normal(0, 1, (size, size)).astype(np.float32)
    q = np.clip(n * 0.25 + 0.5, 0, 1)
    im = Image.fromarray((q * 255).astype(np.uint8))
    lo = np.asarray(im.filter(ImageFilter.GaussianBlur(sigma_low)), dtype=np.float32) / 255.0
    hi = np.asarray(im.filter(ImageFilter.GaussianBlur(sigma_high)), dtype=np.float32) / 255.0
    band = lo - hi
    return band / (band.std() + 1e-6)


def verify_isotropy(im: Image.Image, bins: int = 18) -> float:
    """方向集中度 = 最强梯度方向占比 / 均匀分布占比（**梯度幅度加权**）。

    * ``≈1``   → 梯度方向均匀 = **各向同性**（纸面纤维应有的样子）
    * ``>>1``  → 梯度集中在单一方向 = 单向条纹（木纹 / 瓦楞条纹）

    用途：当初"快递盒看起来像木纹"就是因为外表面画了单向条纹；这个指标能把这类
    失误量化出来，改完纹理可自检（实测：单向条纹 ≈ 4~18，纸面纤维 ≈ 1.1）。

    实现要点：**必须用幅度加权**，不要先按幅度阈值筛选像素 —— 对理想的单向正弦条纹，
    梯度幅度处处相等，阈值筛选会挑出空集合，指标反而变成 0（曾踩此坑）。
    """
    a = np.asarray(im.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(a)
    mag = np.hypot(gx, gy).ravel()
    ang = np.arctan2(gy, gx).ravel() % np.pi
    hist, _ = np.histogram(ang, bins=bins, range=(0, np.pi), weights=mag)
    total = float(hist.sum())
    if total <= 0:
        return 1.0
    return float(hist.max() / total * bins)


def make_cardboard(rng: np.random.Generator) -> Image.Image:
    """瓦楞纸**表面**：牛皮纸色 + 各向同性纤维 + 颗粒 + 污渍（**不要**单向条纹）。"""
    size = SIZE
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    # 牛皮纸底色：偏灰黄的**浅**褐（真实纸箱多为浅牛皮色，不是深棕）
    base = np.array([0.81, 0.71, 0.57], dtype=np.float32) + rng.normal(0, 0.012, 3).astype(np.float32)

    # 1) 大尺度斑驳：纸张吸湿/受压造成的缓慢明暗不均（各向同性）
    blotch = _band_noise(rng, size, 10.0, 48.0) * 0.050

    # 2) 纤维尺度结构：纸面压平后的短纤维交织（各向同性带通噪声）
    fiber = _band_noise(rng, size, 0.8, 3.5) * 0.032

    # 3) 细颗粒（纸面粗糙感）
    speck = rng.normal(0, 0.018, (size, size)).astype(np.float32)

    shade = 1.0 + blotch + fiber + speck
    img = base[None, None, :] * shade[:, :, None]

    # 注意：**不要**用"单向条纹"（瓦楞条纹/折痕直线/单向正弦）—— 那是最强的方向性
    # 来源，会让纸面看起来像木纹。也不要依赖"多个随机方向正弦叠加"：加了 48 个方向
    # 看似各向同性，但下采样到盒面尺度后会残留方向性（实测渲染后指标 4.35）。
    # 各向同性带通噪声才是正确做法（渲染后 ≈1.0）。

    # 5) 少量污渍/磨损（圆斑，各向同性）
    #    注意：**不要**画"折痕/压痕"这类长直线 —— 最强调方向性来源，会让纸面又像木纹
    #    （实测：加两条淡压线就把方向集中度从 1.2 推到 1.5）。
    for _ in range(rng.integers(2, 5)):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(10, 42)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = np.exp(-(d / r) ** 2) * rng.uniform(0.02, 0.07)  # 浅色纸面污渍更淡
        img *= (1.0 - mask)[:, :, None]

    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def make_table(rng: np.random.Generator) -> Image.Image:
    """**哑光层压板**台面：均匀浅灰白 + 细密颗粒 + 极轻微不均 + 少量浅划痕。

    注意：早期版本做了木纹（grain）——快递分拣场景里真实台面通常是哑光 PVC /
    层压板 / 金属，不是木头，因此这里去掉任何木纹走向，只保留"哑光"的关键特征：
    **细密均匀的颗粒感**（配合材质 roughness≈0.9 才像哑光面）。
    """
    size = SIZE
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    # 浅灰白底色（哑光层压板）
    base = np.array([0.80, 0.79, 0.775], dtype=np.float32) + rng.normal(0, 0.008, 3).astype(np.float32)

    # 轻微的大尺度不均（真实台面不会绝对均匀；但必须很淡，不能形成花纹）
    uneven = _band_noise(rng, size, 12.0, 48.0) * 0.030

    # 哑光的关键：细密颗粒（各向同性）。
    # ⚠️ 尺度很重要：纹理 512 px 覆盖约 10 cm 台面（uv_scale=6），所以 1 px ≈ 0.2 mm。
    # 若颗粒用 sigma<3 px（<0.6 mm），在图像里桌面上只占零点几个像素，会被直接平均掉
    # ——实测台面区域 std 掉到 0.5，看起来是"死板纯色"。这里用 0.8~2.8 mm 的尺度。
    fine = _band_noise(rng, size, 4.0, 14.0) * 0.020
    uneven2 = _band_noise(rng, size, 2.0, 6.0) * 0.010
    speck = rng.normal(0, 0.010, (size, size)).astype(np.float32)  # 亚像素，仅作抗锯齿噪声

    shade = 1.0 + uneven + fine + uneven2 + speck
    img = base[None, None, :] * shade[:, :, None]

    # 少量很淡的划痕（哑光台面常见使用痕迹）
    for _ in range(rng.integers(3, 7)):
        y0, x0 = rng.integers(0, size, 2)
        ang = rng.uniform(0, np.pi)
        length = rng.integers(60, 260)
        bright = rng.uniform(-0.025, 0.025)
        for t in range(length):
            x = int(x0 + t * np.cos(ang)) % size
            y = int(y0 + t * np.sin(ang)) % size
            img[y, x] *= 1.0 + bright

    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def make_floor(rng: np.random.Generator) -> Image.Image:
    """中性地板：中灰底 + 低频斑驳 + 细颗粒（避免纯色平面的塑料感）。"""
    size = SIZE
    base = np.array([0.46, 0.45, 0.44], dtype=np.float32) + rng.normal(0, 0.01, 3).astype(np.float32)
    blotch = _fbm(rng, size, octaves=4, base=3) * 0.10
    speck = rng.normal(0, 0.02, (size, size)).astype(np.float32)
    shade = 1.0 + blotch + speck
    img = base[None, None, :] * shade[:, :, None]
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate procedural textures for the parcel-flip scene.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--with-cardboard",
        action="store_true",
        help="额外生成 cardboard.png（快递盒目前用纯色 displayColor，默认不需要；"
        "只有想给盒子换回纹理时才生成）。",
    )
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [("table", make_table), ("floor", make_floor)]
    if args.with_cardboard:
        targets.insert(0, ("cardboard", make_cardboard))
    for name, fn in targets:
        im = fn(rng)
        p = OUT_DIR / f"{name}.png"
        im.save(p, optimize=True)
        arr = np.asarray(im, dtype=np.float32)
        iso = verify_isotropy(im)
        print(f"  {p.relative_to(PROJECT_ROOT)}  {im.size}  "
              f"mean RGB={arr.reshape(-1, 3).mean(0).round(0)}  std={arr.reshape(-1, 3).std(0).round(1)}  "
              f"{p.stat().st_size / 1024:.1f} KB  方向集中度={iso:.2f}"
              f"（≈1 各向同性；越大越像单向条纹/木纹）")


if __name__ == "__main__":
    main()
