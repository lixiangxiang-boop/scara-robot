"""MuJoCo 实时演示：SCARA 用两指夹爪循环抓取-放置（物体位置每轮随机）。

运行：python scripts/demo_scara_viewer.py [--seed 0] [--cycles 0]
       --cycles 0 = 无限循环（关窗即退出）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from armlab import robot_controller as RC  # noqa: E402
from armlab import scara_scene as SC  # noqa: E402
from armlab import spec as S  # noqa: E402
from armlab.camera_view import MultiCameraView  # noqa: E402

PLACE_XY = (0.30, 0.12)          # 放置区固定（演示更直观）；物体起点每轮随机


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=0, help="0 = 无限循环")
    ap.add_argument("--cam-view", nargs="*", default=["cam_top", "cam_eye"],
                    help="同时显示哪些相机的实时视野（默认 俯视 + 第一人称）")
    ap.add_argument("--no-cam-view", action="store_true", help="不打开相机视野窗口")
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer

    rng = np.random.default_rng(args.seed)
    spec = S.scara_teach_4rprr()

    def random_obj_xy():
        return (0.30 + rng.uniform(-0.035, 0.035), -0.05 + rng.uniform(-0.035, 0.035))

    obj_xy = random_obj_xy()
    model = SC.build_model(
        spec,
        object_pos=(obj_xy[0], obj_xy[1], SC.TABLE_TOP + SC.OBJ_SIZE / 2),
        place_center=PLACE_XY,
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"[info] 场景就绪（两指夹爪）：物体起点=({obj_xy[0] * 1000:.0f}, "
          f"{obj_xy[1] * 1000:.0f}) mm", flush=True)

    cycles_done = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.28, 0.04, 0.16)
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 138
        viewer.cam.elevation = -18
        cam_view = None if args.no_cam_view else MultiCameraView(
            model, data, cams=tuple(args.cam_view), width=640, height=480)

        def sync():
            viewer.sync()
            if cam_view is not None:
                cam_view.update()

        ctrl = RC.ScaraController(model, data, spec, sync=sync)
        if cam_view is not None:
            obj_id = SC.ids(model, "body", "object")
            cam_view.set_target_fn(lambda: data.xpos[obj_id].copy())
        ctrl.set_place_hint(*PLACE_XY)
        while viewer.is_running():
            print(f"[cycle {cycles_done + 1}] 抓取中…", flush=True)
            ok_pick = ctrl.pick(*obj_xy)
            ok_place = ctrl.place(*PLACE_XY)
            obj = ctrl.object_position()
            err = float(np.hypot(obj[0] - PLACE_XY[0], obj[1] - PLACE_XY[1]))
            print(f"[cycle {cycles_done + 1}] {'成功' if (ok_pick and ok_place) else '失败'}"
                  f" → 放置误差 {err * 1000:.1f} mm", flush=True)
            cycles_done += 1
            if args.cycles and cycles_done >= args.cycles:
                break
            ctrl.move_to(obj_xy[0], obj_xy[1], 0.26, duration=0.8)
            # 下一轮：物体换到新的随机位置（手臂保持当前姿态重新规划）
            obj_xy = random_obj_xy()
            adr = SC.object_qpos_addr(model)
            data.qpos[adr:adr + 3] = (obj_xy[0], obj_xy[1],
                                      SC.TABLE_TOP + SC.OBJ_SIZE / 2)
            data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            print(f"[info] 新物体位置 ({obj_xy[0] * 1000:.0f}, {obj_xy[1] * 1000:.0f}) mm",
                  flush=True)
        if cam_view is not None:
            cam_view.close()
    print(f"[done] 共完成 {cycles_done} 个抓取循环", flush=True)


if __name__ == "__main__":
    main()
