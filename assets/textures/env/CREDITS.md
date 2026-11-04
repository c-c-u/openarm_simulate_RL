# 真实场景背景照片（CC0）

`assets/textures/env/` 下的图片是给仿真场景"围墙"用的**真实场景照片**（墙面 / 地面材质特写），
用来缩小 sim→real 的外观差距，并作为外观域随机化素材：

| 文件 | 内容 | 尺寸 |
|---|---|---|
| `wall_painted_white.jpg` | 白色乳胶漆墙面（细颗粒） | 1024×1024 |
| `wall_plaster_grey.jpg` | 灰色抹灰墙面 | 1024×1024 |
| `wall_concrete.jpg` | 清水混凝土墙面 | 1024×1024 |
| `wall_painted_dark.jpg` | 深色漆面墙 | 1024×1024 |
| `floor_wood.jpg` | 木地板（可选：替换默认地面） | 1024×1024 |
| `floor_concrete.jpg` | 水泥地（可选：替换默认地面） | 1024×1024 |

## 来源与许可

全部来自 [ambientCG](https://ambientcg.com)，**CC0 1.0（公有领域）**，可商用、无需署名。
这里仍列出原始资产 ID 以便溯源（均为 1K-JPG 版本的 Color 贴图，已居中裁成正方形、
JPEG q=88 重新压缩）：

| 文件 | ambientCG 资产 | 下载地址 |
|---|---|---|
| `wall_painted_white.jpg` | `PaintedPlaster017` | https://ambientcg.com/view?id=PaintedPlaster017 |
| `wall_plaster_grey.jpg` | `Plaster002` | https://ambientcg.com/view?id=Plaster002 |
| `wall_concrete.jpg` | `Concrete034` | https://ambientcg.com/view?id=Concrete034 |
| `wall_painted_dark.jpg` | `PaintedPlaster010` | https://ambientcg.com/view?id=PaintedPlaster010 |
| `floor_wood.jpg` | `WoodFloor051` | https://ambientcg.com/view?id=WoodFloor051 |
| `floor_concrete.jpg` | `Concrete020` | https://ambientcg.com/view?id=Concrete020 |

## 换成自己现场拍的照片

直接往这个目录扔 `*.png / *.jpg / *.jpeg / *.webp` 即可（按文件名排序后循环使用），
或用环境变量指向别处：

```bash
export OPENARM_ENV_TEX_DIR=/data/real_room_photos
```

文件名以 `floor_` 开头的建议留给地面，其余当作墙面照片使用。
照片建议：**正视、无强反光、覆盖整面墙**（相机会把它铺满一面 1.9×1.6 m 的墙）。
