"""SCARA 抓取-放置批量评测（MuJoCo，两指夹爪 + CTC 控制器）。

与实时演示共用同一个控制器（`armlab.robot_controller.ScaraController`）：
    pick  = 张开 → 高位接近 → 下探 → 闭合 → 抬起
    place = 搬运 → 下探 → 张开 → 全部缩回

运行：python scripts/run_scara_pick_place.py [--rand 5]
输出：outputs/data/scara_pick_place.json、outputs/figures/scara_pick_place.png
"""

from __future__ import annotations

import argparse
import json
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

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs" / "data"
FIG = ROOT / "outputs" / "figures"


def run_once(seed: int, verbose: bool = True):
    import mujoco

    rng = np.random.default_rng(seed)
    spec = S.scara_teach_4rprr()
    obj_xy = (0.30 + rng.uniform(-0.035, 0.035), -0.05 + rng.uniform(-0.035, 0.035))
    place_xy = (0.30 + rng.uniform(-0.02, 0.02), 0.12 + rng.uniform(-0.02, 0.02))

    model = SC.build_model(spec, object_pos=(obj_xy[0], obj_xy[1], SC.TABLE_TOP + SC.OBJ_SIZE / 2),
                           place_center=place_xy)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ctrl = RC.ScaraController(model, data, spec)
    ctrl.record = True
    ctrl.set_place_hint(*place_xy)
    ok_pick = ctrl.pick(*obj_xy)
    ok_place = ctrl.place(*place_xy)

    obj = ctrl.object_position()
    err = float(np.hypot(obj[0] - place_xy[0], obj[1] - place_xy[1]))
    on_table = bool(abs(obj[2] - (SC.TABLE_TOP + SC.OBJ_SIZE / 2)) < 0.012)
    at_rest = bool(np.linalg.norm(data.qvel[-6:]) < 0.05)
    peak = np.max(np.abs(np.asarray(ctrl.history["tau"])), axis=0) if ctrl.history["tau"] else np.zeros(4)
    res = {
        "seed": seed, "object_start": list(obj_xy), "place_center": list(place_xy),
        "object_final": obj.tolist(), "place_error_m": err,
        "on_table": on_table, "at_rest": at_rest,
        "success": bool(ok_pick and ok_place and err < 0.025 and on_table and at_rest),
        "peak_torque_Nm": peak.tolist(), "cycle_time_s": float(ctrl.history["t"][-1]),
        "grasp": "两指夹爪（摩擦）",
    }
    if verbose:
        print(f"  物体终点 ({obj[0] * 1000:.1f}, {obj[1] * 1000:.1f}, {obj[2] * 1000:.1f}) mm，"
              f"放置误差 {err * 1000:.1f} mm，{'成功' if res['success'] else '失败'}")
    return res, ctrl.history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rand", type=int, default=5)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    print("===== SCARA 抓取-放置评测（两指夹爪 + CTC）=====")
    results, last_hist = [], None
    for seed in range(args.rand):
        print(f"[工况 {seed + 1}/{args.rand}]")
        res, hist = run_once(seed)
        results.append(res)
        last_hist = hist
    n_ok = sum(1 for r in results if r["success"])
    print(f"\n成功率：{n_ok}/{len(results)}")
    print(f"峰值力矩（J1/J2/J3/J4）：{np.max([r['peak_torque_Nm'] for r in results], axis=0).round(2)}")
    print(f"平均放置误差：{np.mean([r['place_error_m'] for r in results]) * 1000:.1f} mm")

    (DATA / "scara_pick_place.json").write_text(
        json.dumps({"results": results, "success_rate": n_ok / len(results),
                    "gripper": "two-finger friction grasp"}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for i, r in enumerate(results):
        axes[0].plot(r["object_start"][0], r["object_start"][1], "o", ms=6, color="C3",
                     label="起点" if i == 0 else None)
        axes[0].annotate("", xy=(r["object_final"][0], r["object_final"][1]),
                         xytext=(r["object_start"][0], r["object_start"][1]),
                         arrowprops=dict(arrowstyle="->", color="C0", alpha=0.6))
        axes[0].plot(r["object_final"][0], r["object_final"][1], "s", ms=6, color="C2",
                     label="终点" if i == 0 else None)
    axes[0].add_patch(plt.Rectangle((0.30 - 0.05, 0.12 - 0.05), 0.1, 0.1, fill=False,
                                    ls="--", ec="C2", lw=1.5, label="放置区"))
    axes[0].plot(0, 0, "k+", ms=12, label="基座(J1)")
    axes[0].set_aspect("equal"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8)
    axes[0].set_title("物体水平位移（俯视，m）"); axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]")
    q = np.asarray(last_hist["q"]); t = np.asarray(last_hist["t"])
    grip = np.asarray(last_hist["grip"])
    axes[1].plot(t, np.degrees(q[:, 0]), label="q1 [°]")
    axes[1].plot(t, np.degrees(q[:, 1]), label="q2 [°]")
    axes[1].plot(t, q[:, 2] * 1000, label="q3 [mm]")
    axes[1].plot(t, np.degrees(q[:, 3]), label="q4 [°]")
    axes[1].plot(t, grip * 1000, label="夹爪开度 [mm]")
    axes[1].set_xlabel("时间 [s]"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    axes[1].set_title("关节与夹爪（末次工况）")
    fig.suptitle(f"SCARA 抓取-放置（两指夹爪）：成功率 {n_ok}/{len(results)}", y=1.02)
    fig.savefig(FIG / "scara_pick_place.png", dpi=130, bbox_inches="tight")
    print(f"图：{FIG / 'scara_pick_place.png'}")


if __name__ == "__main__":
    main()
