"""交互式控制器：在 MuJoCo 窗口旁边用命令行操作 SCARA（抓取/放置/移动）。

启动：python scripts/scara_console.py
命令（在控制台输入）：
    pick <x> <y>        抓取 (x,y) 处的物体（单位 m，可省小数，如 pick 0.30 -0.05）
    place <x> <y>       放到 (x,y)
    place               放到当前放置区（默认 0.30 0.12）
    move <x> <y> <z>    末端移动到笛卡尔点
    open / close        张开 / 闭合夹爪
    home                回零位（全缩回 + 张开）
    spawn <x> <y>       把物体挪到 (x,y)（重新摆一个物体）
    where               打印当前末端/物体位置
    auto [n]            自动演示 n 轮（默认 3）
    quit                退出

实现：主线程跑 MuJoCo 物理与渲染同步，另开一个守护线程读键盘输入 → 命令队列，
因此**窗口始终不卡**，命令一条条执行。
"""

from __future__ import annotations

import queue
import sys
import threading
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
from armlab.camera_view import CameraView  # noqa: E402

PLACE_XY = (0.30, 0.12)


def reader(q: "queue.Queue[str]") -> None:
    while True:
        try:
            line = input()
        except EOFError:
            break
        q.put(line.strip())


def main() -> None:
    import mujoco
    import mujoco.viewer

    rng = np.random.default_rng()
    spec = S.scara_teach_4rprr()
    obj_xy = (0.30 + rng.uniform(-0.03, 0.03), -0.05 + rng.uniform(-0.03, 0.03))
    model = SC.build_model(spec,
                           object_pos=(obj_xy[0], obj_xy[1], SC.TABLE_TOP + SC.OBJ_SIZE / 2),
                           place_center=PLACE_XY)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(__doc__)
    print(f"[info] 物体初始位置 ({obj_xy[0]:.3f}, {obj_xy[1]:.3f})；"
          f"放置区 {PLACE_XY}", flush=True)

    cmds: "queue.Queue[str]" = queue.Queue()
    threading.Thread(target=reader, args=(cmds,), daemon=True).start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.28, 0.04, 0.16)
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 138
        viewer.cam.elevation = -18
        cam_view = CameraView(model, data, cam_name="cam_top", width=800, height=600)

        def sync():
            viewer.sync()
            cam_view.update()

        ctrl = RC.ScaraController(model, data, spec, sync=sync)
        cam_view.set_tcp_fn(ctrl.tcp)
        ctrl.set_place_hint(*PLACE_XY)
        while viewer.is_running():
            try:
                line = cmds.get(timeout=0.02)
            except queue.Empty:
                ctrl.hold(0.02)               # 空转时保持位姿并刷新窗口
                continue
            parts = line.split()
            if not parts:
                continue
            cmd, a = parts[0].lower(), parts[1:]
            try:
                if cmd in ("quit", "exit", "q"):
                    break
                elif cmd in ("help", "?", "h"):
                    print(__doc__)
                elif cmd == "pick":
                    ctrl.pick(float(a[0]), float(a[1]))
                elif cmd == "place":
                    if len(a) >= 2:
                        ctrl.place(float(a[0]), float(a[1]))
                    else:
                        ctrl.place()
                elif cmd == "move":
                    ctrl.move_to(float(a[0]), float(a[1]), float(a[2]))
                elif cmd == "open":
                    ctrl.open_gripper()
                elif cmd == "close":
                    ctrl.close_gripper()
                elif cmd == "home":
                    ctrl.home()
                elif cmd == "spawn":
                    x, y = float(a[0]), float(a[1])
                    adr = SC.object_qpos_addr(model)
                    data.qpos[adr:adr + 3] = (x, y, SC.TABLE_TOP + SC.OBJ_SIZE / 2)
                    data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    print(f"[ok] 物体已摆到 ({x:.3f}, {y:.3f})")
                elif cmd == "where":
                    print(f"[info] 末端 {np.round(ctrl.tcp(), 3)}，"
                          f"物体 {np.round(ctrl.object_position(), 3)}")
                elif cmd == "auto":
                    n = int(a[0]) if a else 3
                    for k in range(n):
                        ox, oy = (0.30 + rng.uniform(-0.035, 0.035),
                                  -0.05 + rng.uniform(-0.035, 0.035))
                        adr = SC.object_qpos_addr(model)
                        data.qpos[adr:adr + 3] = (ox, oy, SC.TABLE_TOP + SC.OBJ_SIZE / 2)
                        data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
                        data.qvel[:] = 0.0
                        mujoco.mj_forward(model, data)
                        print(f"[auto {k + 1}/{n}] 抓 ({ox:.3f}, {oy:.3f}) → 放 {PLACE_XY}")
                        if not (ctrl.pick(ox, oy) and ctrl.place(*PLACE_XY)):
                            print("[warn] 本轮失败（目标可能不可达）")
                        obj = ctrl.object_position()
                        print(f"[auto {k + 1}/{n}] 放置误差 "
                              f"{np.hypot(obj[0] - PLACE_XY[0], obj[1] - PLACE_XY[1]) * 1000:.1f} mm")
                else:
                    print(f"[warn] 未知命令：{cmd}（输入 help 查看帮助）")
                    print(__doc__)
            except (IndexError, ValueError) as exc:
                print(f"[warn] 命令参数有误：{line!r} → {exc}")
        cam_view.close()
    print("[done] 已退出控制器")


if __name__ == "__main__":
    main()
