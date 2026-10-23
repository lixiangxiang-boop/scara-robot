"""策略现场演示：在 MuJoCo 窗口里看**学到的策略**自己抓取-放置（含相机视野）。

与 `python/scripts/demo_scara_viewer.py`（专家控制器演示）对应：
    专家演示 = 时间驱动的完整状态机（"剧本"）
    策略演示 = 相位分类策略（19 维观测 → 8 相位）+ 运动原语 + 抓取安全闩锁
两者共用同一套场景、CTC 控制器与相机，可直接对照。

运行：python learning/demo_policy_viewer.py [--cycles 0] [--seed 0] [--no-cam-view]
      --cycles 0 = 无限循环（关 MuJoCo 窗口退出）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from armlab import scara_scene as SC               # noqa: E402
from armlab import spec as S                       # noqa: E402
from armlab.camera_view import MultiCameraView     # noqa: E402
from learning import rollout as RU                 # noqa: E402
from learning.policy_io import Policy              # noqa: E402

PLACE_XY = (0.30, 0.12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cycles", type=int, default=0, help="0 = 无限循环")
    ap.add_argument("--cam-view", nargs="*", default=["cam_top", "cam_eye"],
                    help="同时显示哪些相机的实时视野（默认 俯视 + 第一人称）")
    ap.add_argument("--no-cam-view", action="store_true")
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer

    spec = S.scara_teach_4rprr()
    rng = np.random.default_rng(args.seed)
    policy = Policy()
    print(f"[info] 相位策略：验证准确率 "
          f"{policy.meta.get('val_phase_accuracy', float('nan')) * 100:.2f}%，"
          f"训练数据 {policy.meta.get('data_episodes', '?')} 条专家轨迹", flush=True)

    def random_scenario():
        return {"obj_xy": (0.30 + rng.uniform(-0.035, 0.035),
                           -0.05 + rng.uniform(-0.035, 0.035)),
                "place_xy": PLACE_XY, "obj_mass": 0.05,
                "q_jitter": np.zeros(3), "fric_scale": 1.0}

    model, data, ctrl, fast = RU.make_env(spec, random_scenario())
    cycles = 0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = (0.28, 0.04, 0.16)
        viewer.cam.distance = 1.15
        viewer.cam.azimuth = 138
        viewer.cam.elevation = -18
        cam_view = None if args.no_cam_view else MultiCameraView(
            model, data, cams=tuple(args.cam_view), width=640, height=480)
        obj_id = SC.ids(model, "body", "object")
        if cam_view is not None:
            cam_view.set_target_fn(lambda: data.xpos[obj_id].copy())

        def hook(step, model_, data_, ctrl_, phase, t_phase):
            viewer.sync()
            if cam_view is not None:
                cam_view.update()

        ctrl.sync = lambda: hook(0, model, data, ctrl, 0, 0.0)   # home() 时也刷新画面
        while viewer.is_running():
            ctrl.home(duration=1.0)                     # 先把手臂收回零位
            sc = random_scenario()
            print(f"[cycle {cycles + 1}] 物体 ({sc['obj_xy'][0] * 1000:.0f}, "
                  f"{sc['obj_xy'][1] * 1000:.0f}) mm → 策略执行中…", flush=True)
            r = RU.run_rollout(spec, policy, sc, env=(model, data, ctrl, fast), hook=hook)
            print(f"[cycle {cycles + 1}] {'成功' if r['success'] else '失败'} → "
                  f"放置误差 {r['place_error_m'] * 1000:.1f} mm（"
                  f"{r['steps']} 拍，末相位 {r['final_phase']}）", flush=True)
            cycles += 1
            if args.cycles and cycles >= args.cycles:
                break
        if cam_view is not None:
            cam_view.close()
    print(f"[done] 共完成 {cycles} 个策略抓取循环", flush=True)


if __name__ == "__main__":
    main()
