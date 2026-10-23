"""双相机协同演示：俯视相机（全局） + J4 支架第一人称相机（局部）。

场景故意把物体放在**机械臂正下方**——俯视相机被大臂挡住，
演示两台相机如何配合：
    ① 俯视相机：负责"去哪"（全局定位物体/放置区）
    ② 第一人称（眼在手）：手臂到位后负责"对准多少"（局部精定位/确认抓取）
    ③ 若俯视被挡：可先把手 arm 挪开（主动视觉 look-then-move）让俯视恢复可见

窗口里每帧标注两台相机对该目标点的可见性：VISIBLE / BLOCKED。
运行：python scripts/demo_dual_camera.py [--cycles 1]
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
from armlab.camera_view import MultiCameraView, visible_from  # noqa: E402

# 物体放在大臂正下方（q=0 时大臂沿 +x 覆盖 x∈[0,0.225]）
OBJ_XY = (0.19, 0.0)
PLACE_XY = (0.40, 0.16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=1)
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer

    spec = S.scara_teach_4rprr()
    model = SC.build_model(spec,
                           object_pos=(OBJ_XY[0], OBJ_XY[1], SC.TABLE_TOP + SC.OBJ_SIZE / 2),
                           place_center=PLACE_XY)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    obj_id = SC.ids(model, "body", "object")

    def target():
        return data.xpos[obj_id].copy()

    def report(tag: str) -> None:
        t = target()
        vis_top = visible_from(model, data, "cam_top", t)
        vis_eye = visible_from(model, data, "cam_eye", t)
        print(f"[{tag}] 俯视 cam_top: {'可见' if vis_top else '被遮挡'} | "
              f"第一人称 cam_eye: {'可见' if vis_eye else '看不到'}", flush=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.30, 0.06, 0.16)
        viewer.cam.distance = 1.25
        viewer.cam.azimuth = 138
        viewer.cam.elevation = -20
        mc = MultiCameraView(model, data, cams=("cam_top", "cam_eye"),
                             width=640, height=480)
        mc.set_target_fn(target)

        def sync():
            viewer.sync()
            mc.update()

        ctrl = RC.ScaraController(model, data, spec, sync=sync)
        ctrl.set_place_hint(*PLACE_XY)

        print("===== 双相机协同演示（物体在机械臂正下方）=====")
        print("原理：俯视相机负责「去哪」（全局定位），第一人称相机负责「对准多少」（局部确认）；")
        print("      当物体正好被手臂挡住时，俯视失效 → 由第一人称接管。")
        report("初始：大臂压在物体上方")

        print("\n① 主动视觉：先把手 arm 挪开，让俯视相机重新看到物体")
        ctrl.move_to(0.40, -0.10, 0.26, duration=1.2)
        report("手臂挪开后")

        print("\n② 俯视相机给出全局目标 → 手臂盲开到位（此时第一人称开始有用）")
        ctrl.move_to(OBJ_XY[0], OBJ_XY[1], 0.26, duration=1.0)
        report("手臂到目标上方")
        ctrl.move_to(OBJ_XY[0], OBJ_XY[1], 0.18, duration=0.9)
        report("下探到抓取高度")

        print("\n③ 抓取（第一人称相机确认爪尖与物体）→ 抬起 → 放到放置区")
        ctrl.close_gripper()
        ctrl.move_to(OBJ_XY[0], OBJ_XY[1], 0.21, duration=0.6)
        report("抓紧抬起后")
        ok = ctrl.place(*PLACE_XY) if ctrl.move_to(*PLACE_XY, 0.21, duration=1.2) else False
        obj = ctrl.object_position()
        err = float(np.hypot(obj[0] - PLACE_XY[0], obj[1] - PLACE_XY[1]))
        print(f"\n放置误差 {err * 1000:.1f} mm → {'成功' if err < 0.025 else '失败'}")
        report("放置完成后")
        ctrl.home()
        mc.close()
    print("[done]")


if __name__ == "__main__":
    main()
