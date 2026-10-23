"""离屏渲染策略执行过程（不用开窗口，输出图片供报告/复核）。

每一幕取关键时刻截图（俯视 + 自由视角）：接近 → 下探 → 抓取(夹爪闭合) → 搬运 → 放置完成

运行：python learning/render_policy_rollout.py [--episodes 3] [--seed 500]
输出：outputs/figures/policy_roll_<k>_<phase>_<cam>.png、outputs/data/policy_rollout.json
"""

from __future__ import annotations

import argparse
import json
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
from learning import rollout as RU                 # noqa: E402
from learning.policy_io import Policy              # noqa: E402

FIG = ROOT / "outputs" / "figures"
DATA = ROOT / "outputs" / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--cams", nargs="*", default=["cam_top", "free"])
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=640)
    args = ap.parse_args()

    import mujoco
    from PIL import Image

    spec = S.scara_teach_4rprr()
    policy = Policy()
    rng = np.random.default_rng(args.seed)
    FIG.mkdir(parents=True, exist_ok=True)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = (0.28, 0.04, 0.16)
    cam.distance = 1.15
    cam.azimuth = 138
    cam.elevation = -18

    out = []
    for k in range(args.episodes):
        sc = RU.sample_scenario(rng)
        env = RU.make_env(spec, sc)
        model, data = env[0], env[1]
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        taken = set()

        def shoot(tag: str) -> None:
            for c in args.cams:
                if c == "free":
                    renderer.update_scene(data, camera=cam)
                else:
                    if SC.ids(model, "camera", c) < 0:
                        continue
                    renderer.update_scene(data, camera=c)
                Image.fromarray(renderer.render()).save(
                    FIG / f"policy_roll_{k + 1}_{tag}_{c}.png")

        def hook(step, model_, data_, ctrl_, phase, t_phase):
            obj_id = SC.ids(model_, "body", "object")
            p_obj = data_.xpos[obj_id]
            p_tcp = ctrl_.tcp()
            if "approach" not in taken and phase == 0 and p_tcp[2] < 0.28:
                taken.add("approach"); shoot("approach")
            if "descend" not in taken and phase == 1:
                taken.add("descend"); shoot("descend")
            if "grasp" not in taken and ctrl_.gripper < 0.004:
                taken.add("grasp"); shoot("grasp")
            if "carry" not in taken and phase == 4:
                taken.add("carry"); shoot("carry")
            if "placed" not in taken and phase >= 6 and \
                    abs(p_obj[2] - (SC.TABLE_TOP + SC.OBJ_SIZE / 2)) < 0.015:
                taken.add("placed"); shoot("placed")

        r = RU.run_rollout(spec, policy, sc, env=env, hook=hook)
        for tag in ("approach", "descend", "grasp", "carry", "placed"):
            if tag not in taken:
                shoot(tag)                        # 补拍（失败时也能看到实际状态）
        renderer.close()
        out.append({"episode": k + 1, "success": bool(r["success"]),
                    "place_error_mm": r["place_error_m"] * 1000,
                    "final_phase": r["final_phase"], "frames": sorted(taken)})
        print(f"[{k + 1}/{args.episodes}] {'成功' if r['success'] else '失败'}，"
              f"放置误差 {r['place_error_m'] * 1000:.1f} mm，取帧 {sorted(taken)}", flush=True)
    (DATA / "policy_rollout.json").write_text(
        json.dumps({"n": len(out), "success": sum(r["success"] for r in out),
                    "results": out}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] 帧：{FIG}\\policy_roll_*.png")


if __name__ == "__main__":
    main()
