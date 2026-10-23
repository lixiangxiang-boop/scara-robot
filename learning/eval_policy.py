"""闭环评估：相位策略在 MuJoCo 里真正执行抓取-放置（与专家头对头）。

控制栈：策略(50 Hz，输出相位) → 运动原语（MoveL + 五次多项式）→ CTC(1 kHz) → MuJoCo
（执行细节与安全闩锁见 learning/rollout.py）

运行：python learning/eval_policy.py [--episodes 20]
输出：outputs/data/policy_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from armlab import robot_controller as RC          # noqa: E402
from armlab import scara_scene as SC               # noqa: E402
from armlab import scara_task as ST                # noqa: E402
from armlab import spec as S                       # noqa: E402
from armlab.scara_fast import ScaraFastModel       # noqa: E402
from learning.policy_io import Policy              # noqa: E402
from learning.rollout import (build_scenario_model, run_rollout,   # noqa: E402
                              sample_scenario)

DATA = ROOT / "outputs" / "data"


def load_meta() -> dict:
    """读取数据 meta（相位名等元信息）。"""
    p = DATA / "demos_meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def expert_episode(spec, sc: dict) -> dict:
    """专家（时间驱动的 PickPlaceTask）：同一批场景的对照。"""
    import mujoco

    model, data, q0 = build_scenario_model(spec, sc, mujoco)
    fast = ScaraFastModel(spec, friction_smooth=0.02)
    ctrl = RC.ScaraController(model, data, spec, dyn=fast)
    task = ST.PickPlaceTask(spec, obj_xy=sc["obj_xy"], place_xy=sc["place_xy"], model=fast)
    task.reset(q0)
    while not task.done:
        tau, grip, _ = task.step(ctrl.q, ctrl.dq)
        ctrl._step(tau, grip)
    obj_id = SC.ids(model, "body", "object")
    p_obj = data.xpos[obj_id].copy()
    err = float(np.hypot(p_obj[0] - sc["place_xy"][0], p_obj[1] - sc["place_xy"][1]))
    on_table = bool(abs(p_obj[2] - (SC.TABLE_TOP + SC.OBJ_SIZE / 2)) < 0.015)
    return {"success": bool(err < 0.025 and on_table), "place_error_m": err,
            "object_final": p_obj.tolist(), "on_table": on_table}


def evaluate(policy, spec, rng, n_ep, use_expert=False, verbose=True):
    results = []
    t0 = time.time()
    for ep in range(n_ep):
        sc = sample_scenario(rng)
        r = expert_episode(spec, sc) if use_expert else run_rollout(spec, policy, sc)
        results.append({"episode": ep, "success": bool(r["success"]),
                        "place_error_m": float(r["place_error_m"]),
                        "object_final": r["object_final"], "on_table": r["on_table"]})
        if verbose and (ep + 1) % 5 == 0:
            n_ok = sum(x["success"] for x in results)
            print(f"  {ep + 1}/{n_ep}：成功率 {n_ok}/{ep + 1}，"
                  f"平均误差 {np.mean([x['place_error_m'] for x in results]) * 1000:.1f} mm，"
                  f"用时 {time.time() - t0:.0f} s", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--no-expert", action="store_true")
    args = ap.parse_args()

    spec = S.scara_teach_4rprr()
    policy = Policy()
    print(f"===== 策略闭环评估（{args.episodes} 个未见场景，相位分类 + 运动原语）=====",
          flush=True)
    rng = np.random.default_rng(args.seed)
    print("\n[1/2] 学到的策略：", flush=True)
    pol = evaluate(policy, spec, rng, args.episodes)
    n_ok = sum(r["success"] for r in pol)
    print(f"  → 成功率 {n_ok}/{len(pol)}，"
          f"平均放置误差 {np.mean([r['place_error_m'] for r in pol]) * 1000:.1f} mm", flush=True)

    exp = None
    if not args.no_expert:
        rng2 = np.random.default_rng(args.seed)
        print("\n[2/2] 专家（同一批场景）：", flush=True)
        exp = evaluate(None, spec, rng2, args.episodes, use_expert=True, verbose=False)
        print(f"  → 成功率 {sum(r['success'] for r in exp)}/{len(exp)}，"
              f"平均放置误差 {np.mean([r['place_error_m'] for r in exp]) * 1000:.1f} mm",
              flush=True)

    out = {"policy": {"success_rate": n_ok / len(pol), "results": pol},
           "expert": ({"success_rate": sum(r["success"] for r in exp) / len(exp),
                       "results": exp} if exp else None),
           "n_episodes": len(pol), "seed": args.seed,
           "control": "phase policy (50 Hz) + motion primitive (MoveL) + CTC (1 kHz)"}
    (DATA / "policy_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"\n[ok] {DATA / 'policy_eval.json'}")


if __name__ == "__main__":
    main()
