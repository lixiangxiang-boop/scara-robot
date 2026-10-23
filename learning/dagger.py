"""DAgger（数据集聚合）：让策略在**自己访问到的状态**上拿到专家相位标签，逐轮纠偏。

为什么需要它（BC 的复合误差）：
    BC 只在专家状态上训练；策略一旦偏离，就进入"没见过的状态"→ 相位判断更容易错 →
    误差滚雪球。DAgger 把策略实际访问到的状态收集起来，标上专家相位，再重训。

每一轮：
    1) **采样**：用当前策略闭环跑 N 条（执行栈与评估完全一致 run_rollout）；
       每 20 ms 用一个**状态驱动的相位状态机**（`PhaseSequencer`，与执行器同构：
       到达路点 + 驻留时间 ⇒ 推进）给当前状态打专家标签；
       末端离专家计划超过 40 mm 的样本直接丢弃（已跑飞的状态标签不可信）。
    2) **聚合 + 重训**：新数据并入原专家数据，重新训练相位分类器。

运行：python learning/dagger.py --rounds 2 --episodes 24 [--epochs 120]
输出：更新 learning/checkpoints/bc_policy.pt；outputs/data/dagger_report.json
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
from armlab import scara_task as ST                # noqa: E402
from armlab import spec as S                       # noqa: E402
from armlab.scara_fast import ScaraFastModel       # noqa: E402
from learning import eval_policy as EV             # noqa: E402
from learning import rollout as RU                 # noqa: E402
from learning.policy_io import Policy              # noqa: E402
from learning.train_bc import train_core           # noqa: E402

DATA = ROOT / "outputs" / "data"
CKPT = ROOT / "learning" / "checkpoints"
PROJ_EVERY = 10            # 专家轨迹投影网格（10 ms，用于"跑飞"判据）
PROJ_MAX_ERR = 0.04        # 偏离专家计划 > 40 mm 视为"已跑飞"，该样本标签不可信 → 丢弃

# 观测噪声增广（按维度给不同量级：关节角 0.015 rad、位置 3 mm、夹爪 1 mm …）
OBS_NOISE = np.array([0.015, 0.015, 2e-3, 0.015,
                      0.03, 0.03, 0.02, 0.03,
                      0.002, 0.002, 0.002, 0.01,
                      0.002, 0.002, 0.002, 0.01,
                      0.0005, 0.002, 0.002, 0.02])


def collect_episode(spec, policy, sc: dict, max_proj_err: float = PROJ_MAX_ERR):
    """用策略闭环跑一条并返回 (obs, act_onehot, 成功?, 保留样本数)。"""
    import mujoco

    # 专家计划（提供参考轨迹用于投影）
    model, data, q0 = RU.build_scenario_model(spec, sc, mujoco)
    fast = ScaraFastModel(spec, friction_smooth=0.02)
    ctrl = RC.ScaraController(model, data, spec, dyn=fast)
    task = ST.PickPlaceTask(spec, obj_xy=sc["obj_xy"], place_xy=sc["place_xy"], model=fast)
    task.reset(q0)
    n_ref = len(task.traj.t)
    grid = np.arange(0, n_ref, PROJ_EVERY)
    P_ref = np.array([ctrl.kin.fk(task.traj.q[i], link="link4")[1] for i in grid])
    seg_arr = np.asarray(task.seg_of_sample)
    seq = ST.PhaseSequencer(sc["obj_xy"], sc["place_xy"])
    state = {"tau": 0}

    def label_fn(step, data_, ctrl_, obs):
        p_tcp = ctrl_.tcp()
        # 1) 状态驱动的专家相位（与执行器同构，含到达/驻留规则）
        ph = seq.update(p_tcp, RU.POLICY_DT)
        # 2) 跑飞判据：末端离专家计划太远 → 丢弃
        lo = max(state["tau"] - 100, 0)
        lo = (lo // PROJ_EVERY) * PROJ_EVERY
        hi = min(state["tau"] + 1200, n_ref - 1)
        sel = np.arange(lo, hi + 1, PROJ_EVERY)
        d = np.linalg.norm(P_ref[sel // PROJ_EVERY] - p_tcp, axis=1)
        i_best = int(sel[int(np.argmin(d))])
        state["tau"] = i_best
        if float(d.min()) > max_proj_err:
            return -1
        return ph

    res = RU.run_rollout(spec, policy, sc, label_fn=label_fn)
    obs = res["obs"]
    lab = res["phase_label"]
    keep = lab >= 0
    obs, lab = obs[keep], lab[keep]
    act = np.zeros((len(lab), 8), dtype=np.float32)
    act[np.arange(len(lab)), lab] = 1.0
    return obs, act, bool(res["success"]), int(len(lab))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=24, help="每轮策略采样条数")
    ap.add_argument("--eval-episodes", type=int, default=12, help="每轮后闭环评估条数")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--obs-noise", type=float, default=-1.0,
                    help=">0 时用统一噪声；默认 -1 = 使用按维度的 OBS_NOISE")
    ap.add_argument("--seed", type=int, default=2000)
    args = ap.parse_args()
    obs_noise = OBS_NOISE if args.obs_noise < 0 else args.obs_noise

    spec = S.scara_teach_4rprr()
    base = np.load(DATA / "demos_pick_place.npz")
    base_obs, base_act, base_ep = base["obs"], base["act"], base["episode"]
    next_ep_id = int(base_ep.max()) + 1
    report = {"rounds": [], "base_samples": int(len(base_obs))}
    t00 = time.time()
    # 备份"纯 BC"权重（DAgger 每轮会覆盖 bc_policy.pt，备份保证可对比/可回退）
    bc_backup = CKPT / "bc_policy_bc.pt"
    if (CKPT / "bc_policy.pt").exists():
        import shutil
        shutil.copy2(CKPT / "bc_policy.pt", bc_backup)
        shutil.copy2(str(CKPT / "bc_policy.pt") + ".json", str(bc_backup) + ".json")
        print(f"[info] 已备份纯 BC 权重 → {bc_backup.name}")

    for rnd in range(1, args.rounds + 1):
        print(f"\n===== DAgger 第 {rnd}/{args.rounds} 轮：策略采样 {args.episodes} 条 =====",
              flush=True)
        policy = Policy()
        rng = np.random.default_rng(args.seed + 100 * rnd)
        new_obs, new_act, new_ep, n_ok, n_kept = [], [], [], 0, 0
        t0 = time.time()
        for k in range(args.episodes):
            sc = RU.sample_scenario(rng)
            obs, act, ok, kept = collect_episode(spec, policy, sc)
            if len(obs) == 0:                    # 该条完全跑飞 → 丢弃
                continue
            new_obs.append(obs)
            new_act.append(act)
            new_ep.append(np.full(len(obs), next_ep_id + k, dtype=np.int32))
            n_ok += int(ok)
            n_kept += kept
            if (k + 1) % 6 == 0:
                print(f"  采样 {k + 1}/{args.episodes}：策略自身成功率 {n_ok}/{k + 1}，"
                      f"保留样本 {n_kept}，用时 {time.time() - t0:.0f} s", flush=True)
        obs_all = np.concatenate([base_obs] + new_obs).astype(np.float32)
        act_all = np.concatenate([base_act] + new_act).astype(np.float32)
        ep_all = np.concatenate([base_ep] + new_ep)
        print(f"  聚合数据集：{len(obs_all)} 样本（新增 {sum(len(o) for o in new_obs)}），"
              f"重训 {args.epochs} epochs …", flush=True)
        res = train_core(obs_all, act_all, ep_all, epochs=args.epochs, seed=args.seed,
                         obs_noise=obs_noise, out_pt=CKPT / "bc_policy.pt", log=print)
        print("  闭环评估（未见场景）…", flush=True)
        rng_eval = np.random.default_rng(1000)
        ev = EV.evaluate(Policy(), spec, rng_eval, args.eval_episodes, verbose=False)
        n_succ = sum(r["success"] for r in ev)
        err = float(np.mean([r["place_error_m"] for r in ev]))
        print(f"  第 {rnd} 轮结果：采样成功率 {n_ok}/{args.episodes}，"
              f"闭环评估 {n_succ}/{len(ev)}，平均放置误差 {err * 1000:.1f} mm", flush=True)
        report["rounds"].append({
            "round": rnd, "sample_success": f"{n_ok}/{args.episodes}",
            "new_samples_kept": int(n_kept),
            "eval_success": f"{n_succ}/{len(ev)}",
            "eval_success_rate": n_succ / len(ev),
            "eval_mean_place_err_mm": err * 1000,
            "val_phase_accuracy": res["meta"]["val_phase_accuracy"],
            "samples": int(len(obs_all))})

    report["total_time_s"] = round(time.time() - t00, 1)
    (DATA / "dagger_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"\n[ok] DAgger 报告：{DATA / 'dagger_report.json'}")


if __name__ == "__main__":
    main()
