"""生成背景墙 USD 资产：``assets/scene/backdrop_wall.usda``。

为什么用脚本生成而不是手写
--------------------------
墙的尺寸（宽 × 高 × 厚）在 env cfg 里参与"墙立在哪、相机能不能看到"的几何推理，
手写的 USD 很容易和 cfg 里的常量脱节。这里把尺寸参数化，改尺寸只要重跑一次。

为什么是薄 Cube 而不是 Plane
---------------------------
原生 ``UsdGeom.Cube`` **没有 UV**（``primvars:st``），没有 UV 时 ``UsdUVTexture``
采样失败、表面渲染成灰/白 —— 这是本项目踩过的坑（见 ``assets/parcel/parcel.usda``
顶部注释）。所以这里和盒子一样：Cube + 手写 24 个 st 值（6 面 × 4 角，
``interpolation = faceVarying``）。用 Cube（有厚度）而不是单面 Plane，也顺带避免了
背面剔除/发光朝向的问题。

用法::

    python3 tools/make_backdrop_usd.py                        # 用默认尺寸生成
    python3 tools/make_backdrop_usd.py --width 2.2 --height 1.8  # 换尺寸（记得同步改
                                                              # env cfg 的 BACKDROP_* 常量）
"""

from __future__ import annotations

import argparse
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "assets" / "scene" / "backdrop_wall.usda"

# 与 env cfg 的 BACKDROP_WIDTH / BACKDROP_HEIGHT / BACKDROP_THICKNESS 保持一致
DEFAULT_WIDTH = 1.9
DEFAULT_HEIGHT = 1.6
DEFAULT_THICKNESS = 0.02

# 6 面 × 4 角的 UV（面顺序 -Z, +Z, -Y, +Y, -X, +X），与 parcel.usda 同一套约定
_UV = """        texCoord2f[] primvars:st = [
            (1, 0), (0, 0), (0, 1), (1, 1),   # -Z
            (0, 0), (0, 1), (1, 1), (1, 0),   # +Z（主视面：u→+X, v→+Y）
            (0, 0), (0, 1), (1, 1), (1, 0),   # -Y
            (1, 0), (0, 0), (0, 1), (1, 1),   # +Y
            (0, 0), (0, 1), (1, 1), (1, 0),   # -X
            (1, 0), (1, 1), (0, 1), (0, 0),   # +X
        ] (
            interpolation = "faceVarying"
        )
"""

_TEMPLATE = """#usda 1.0
(
    defaultPrim = "BackdropWall"
    metersPerUnit = 1
    upAxis = "Z"
)

# 背景墙资产（由 tools/make_backdrop_usd.py 生成，勿手改）。
#
# 局部坐标系：面板在 XY 平面内、法向 = 局部 +Z（与 GroundPlane 同约定），
# 因此摆放时用 tools/env_backdrop.plane_quat_xyzw(inward_normal) 把 +Z 转到
# "朝向工作区"的方向即可。
#
#   x 方向 = 墙宽（{width} m）
#   y 方向 = 墙高（{height} m）
#   z 方向 = 墙厚（{thickness} m，薄板；无碰撞体，只是可见几何）
#
# 材质不写在这里：墙面的真实场景照片由 `mdp/events.py: apply_scene_textures`
# 在 startup 时创建到 /World/Looks 下并统一绑定（全局共享一份，避免每个环境复制）。

def Xform "BackdropWall" (
    kind = "component"
)
{{
    def Cube "Panel"
    {{
        double size = 1
        float3 xformOp:scale = ({width}, {height}, {thickness})
        uniform token[] xformOpOrder = ["xformOp:scale"]
        color3f[] primvars:displayColor = [(0.72, 0.72, 0.72)]  # 没有照片时的中性灰
{uv}    }}
}}
"""


def build(width: float, height: float, thickness: float) -> str:
    """按尺寸渲染 USD 文本（模板里的花括号用 ``{{`` / ``}}`` 转义）。"""
    return _TEMPLATE.format(width=width, height=height, thickness=thickness, uv=_UV)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成背景墙 USD 资产")
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH, help="墙宽（米，局部 X）")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT, help="墙高（米，局部 Y）")
    parser.add_argument("--thickness", type=float, default=DEFAULT_THICKNESS, help="墙厚（米，局部 Z）")
    parser.add_argument("--out", type=pathlib.Path, default=OUT_PATH)
    args = parser.parse_args()

    text = build(args.width, args.height, args.thickness)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"已生成 {args.out}  ({args.width} x {args.height} x {args.thickness} m)")


if __name__ == "__main__":
    main()
