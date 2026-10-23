"""渲染场景截图（用 MJCF 相机做离屏渲染，供报告/视觉阶段使用）。

运行：python scripts/render_scene.py [--pose]
输出：outputs/figures/scene_<相机名>.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from armlab import robot_controller as RC  # noqa: E402
from armlab import scara_scene as SC  # noqa: E402
from armlab import spec as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "outputs" / "figures"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--cams", nargs="*", default=["cam_front", "cam_top", "cam_wrist"])
    ap.add_argument("--pose", action="store_true", help="先抓起物体再渲染（画面更有内容）")
    args = ap.parse_args()

    import mujoco

    spec = S.scara_teach_4rprr()
    model = SC.build_model(spec, object_pos=(0.30, -0.05, SC.TABLE_TOP + SC.OBJ_SIZE / 2),
                           place_center=(0.30, 0.12))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    if args.pose:
        ctrl = RC.ScaraController(model, data, spec)
        ctrl.pick(0.30, -0.05)
        ctrl.move_to(0.28, 0.02, 0.225, duration=0.8)

    FIG.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    for cam in args.cams:
        try:
            renderer.update_scene(data, camera=cam if SC.ids(model, "camera", cam) >= 0 else None)
            img = renderer.render()
        except Exception as exc:
            print(f"[warn] 相机 {cam} 渲染失败：{exc}")
            continue
        out = FIG / f"scene_{cam}.png"
        try:
            from PIL import Image
            Image.fromarray(img).save(out)
        except Exception:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.imsave(out, img)
        print(f"[ok] {out}")


if __name__ == "__main__":
    main()
