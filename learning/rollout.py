"""策略闭环执行（评估 / 演示 / DAgger 共用同一份执行逻辑）。

分层结构
--------
    任务级（学到的）：策略看 19 维观测 → 输出当前相位（8 选 1）
    执行级（离线原语）：`PickPlaceTask.phase_segment(相位, 当前状态)` —— MoveL 五次多项式，
        与专家控制器同一套生成器，末端精度 1 mm 量级

一个相位开始时，用**当前实际状态**重新锚定该相位的原语，然后按时间采样执行
（与专家做法一致），相位切换由策略在 50 Hz 决定。

安全闩锁（执行层的常识逻辑，防止手指在错误位置闭合）：
    - 只有相位 ≥ grasp 且末端到过抓取点 15 mm 以内，才允许夹爪闭合；
    - 相位 ≥ release 时张开并复位闩锁。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from armlab import robot_controller as RC          # noqa: E402
from armlab import scara_scene as SC               # noqa: E402
from armlab import scara_task as ST                # noqa: E402
from armlab.scara_fast import ScaraFastModel       # noqa: E402
from learning.policy_io import build_obs           # noqa: E402

POLICY_DT = 0.02            # 策略频率 50 Hz
T_MAX = 10.0                # 单条上限（相位驱动比时间驱动略慢）
GRASP_IDX = 2               # PHASES 里 "grasp" 的下标
RELEASE_IDX = 6             # PHASES 里 "release" 的下标
LATCH_NEAR = 0.010          # 末端进入距抓取点 10 mm 即允许合爪（防止在半空合爪）


def sample_scenario(rng: np.random.Generator) -> dict:
    """随机场景采样（数据工厂/DAgger/评估共用同一分布）。"""
    return {"obj_xy": (0.30 + rng.uniform(-0.04, 0.04), -0.05 + rng.uniform(-0.04, 0.04)),
            "place_xy": (0.30 + rng.uniform(-0.03, 0.03), 0.12 + rng.uniform(-0.03, 0.03)),
            "obj_mass": float(rng.uniform(0.03, 0.08)),
            "q_jitter": np.deg2rad(rng.uniform(-6, 6, 3)),
            "fric_scale": float(rng.uniform(0.7, 1.4))}


def build_scenario_model(spec, sc: dict, mujoco):
    """按场景建立 MuJoCo 模型与初始状态（含摩擦缩放与初始关节抖动）。"""
    model = SC.build_model(spec,
                           object_pos=(sc["obj_xy"][0], sc["obj_xy"][1],
                                       SC.TABLE_TOP + SC.OBJ_SIZE / 2),
                           place_center=sc["place_xy"], object_mass=sc["obj_mass"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    model.dof_damping[:4] *= sc["fric_scale"]
    model.dof_frictionloss[:4] *= sc["fric_scale"]
    q0 = np.array([sc["q_jitter"][0], sc["q_jitter"][1], 0.0, sc["q_jitter"][2]])
    data.qpos[:4] = q0
    mujoco.mj_forward(model, data)
    return model, data, q0


def make_env(spec, sc: dict):
    """建一套可复用的仿真环境（viewer 演示用：模型只建一次，之后用 reset_env 重置）。"""
    import mujoco

    model, data, _ = build_scenario_model(spec, sc, mujoco)
    fast = ScaraFastModel(spec, friction_smooth=0.02)
    ctrl = RC.ScaraController(model, data, spec, dyn=fast)
    ctrl.set_place_hint(*sc["place_xy"])
    return model, data, ctrl, fast


def reset_env(model, data, sc: dict, *, reset_arm: bool = True) -> np.ndarray:
    """把物体/关节状态重置为场景 sc 的初始值（不重建模型）。"""
    import mujoco

    if reset_arm:
        data.qvel[:] = 0.0
        adr = SC.object_qpos_addr(model)
        data.qpos[adr:adr + 3] = (sc["obj_xy"][0], sc["obj_xy"][1],
                                  SC.TABLE_TOP + SC.OBJ_SIZE / 2)
        data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
        q0 = np.array([sc["q_jitter"][0], sc["q_jitter"][1], 0.0, sc["q_jitter"][2]])
        data.qpos[:4] = q0
    else:
        q0 = np.asarray(data.qpos[:4], dtype=float).copy()
    mujoco.mj_forward(model, data)
    return q0


def run_rollout(spec, policy, sc: dict, *, hook: Optional[Callable] = None,
                label_fn: Optional[Callable] = None, max_steps: Optional[int] = None,
                env=None) -> dict:
    """执行一条策略闭环（以相位驱动，非时间驱动）。

    hook(step, model, data, ctrl, phase, t_phase)：每拍回调（用于渲染/记录）；
    label_fn(step, data, ctrl, obs) -> int：若给出，则每拍记录一个"专家标签"（DAgger 用）。
    """
    import mujoco

    if env is None:
        model, data, q0 = build_scenario_model(spec, sc, mujoco)
        fast = ScaraFastModel(spec, friction_smooth=0.02)
        ctrl = RC.ScaraController(model, data, spec, dyn=fast)
        ctrl.set_place_hint(*sc["place_xy"])
    else:
        model, data, ctrl, fast = env
        reset_env(model, data, sc)
    task = ST.PickPlaceTask(spec, obj_xy=sc["obj_xy"], place_xy=sc["place_xy"], model=fast)
    obj_id = SC.ids(model, "body", "object")
    grasp_wp = np.array([sc["obj_xy"][0], sc["obj_xy"][1], ST.GRASP_Z])

    phase = 0
    t_phase = 0.0
    seg = task.phase_segment(phase, ctrl.q)      # 相位原语（该相位开始时锚定一次）
    grip_cmd = SC.GRIP_OPEN
    latched = False
    n_sub = int(round(POLICY_DT / ctrl.dt))
    obs_rec, lab_rec, phase_rec = [], [], []
    n_steps = int(max_steps or round(T_MAX / POLICY_DT))
    p_hist = []
    step = 0
    for step in range(n_steps):
        R_tcp, p_tcp = ctrl.kin.fk(ctrl.q, link="link4")
        yaw_tcp = float(np.arctan2(R_tcp[1, 0], R_tcp[0, 0]))
        p_obj = data.xpos[obj_id].copy()
        R_obj = data.xmat[obj_id].reshape(3, 3)
        yaw_obj = float(np.arctan2(R_obj[1, 0], R_obj[0, 0]))
        obs = build_obs(ctrl.q, ctrl.dq, p_tcp, yaw_tcp, p_obj, yaw_obj,
                        ctrl.gripper, sc["place_xy"],
                        progress=min(t_phase / max(ST.PHASES[phase].duration, 1e-6), 1.0))
        if label_fn is not None:
            obs_rec.append(obs)
            lab_rec.append(int(label_fn(step, data, ctrl, obs)))

        # 单调推进约束（执行层的"顺序表"）：相位只允许保持或前进一段，
        # 不允许跳变/回调 —— 消除"网络在两个类之间每 40 ms 抖一次"导致的原语互相抵消
        # （实测：悬停点 descend↔retract 抖动 → 机械臂原地爬行 9 s 不动）。
        ph = int(policy.phase(obs))
        ph = min(max(ph, phase), phase + 1, len(ST.PHASES) - 1)
        # 停滞兜底（看门狗）：本段已跑满标称时长、末端 1.2 s 没动、策略又没推进
        # ⇒ 强制推进一段。等价于把 PhaseSequencer 的"到达+驻留"超时规则放在执行层，
        # 保证任何情况下序列都会走完（避免策略在边界卡死把整条任务拖到超时）。
        p_hist.append(p_tcp.copy())
        if len(p_hist) > 60:
            p_hist.pop(0)
        if (ph == phase and phase < len(ST.PHASES) - 1
                and t_phase > ST.PHASES[phase].duration
                and len(p_hist) == 60
                and float(np.max(np.linalg.norm(np.asarray(p_hist) - p_tcp, axis=1))) < 0.002):
            ph = phase + 1
        if ph != phase:                          # 相位切换 → 用当前状态重新锚定原语
            phase, t_phase = ph, 0.0
            seg = task.phase_segment(phase, ctrl.q)
        phase_rec.append(phase)

        # ---- 夹爪：接近即合、直到释放才张开（安全闩锁）----
        # 为什么不用"相位 == grasp 才合"：策略在相位边界可能只停 1~2 拍，
        # 而合爪需要持续下指令；实测出现过"到达抓取点 6 mm、手指全程没合"的失败。
        # 改成"到过抓取点附近就闩上、一直夹到 release"后，抓取不再依赖策略在
        # grasp 相位的停留时长（这也更接近真实抓手的工作方式）。
        if phase >= RELEASE_IDX:
            latched = False
        elif float(np.linalg.norm(p_tcp - grasp_wp)) < LATCH_NEAR:
            latched = True
        g_target = 0.0 if latched else SC.GRIP_OPEN
        grip_cmd += (g_target - grip_cmd) * min(POLICY_DT / 0.1, 1.0)

        # ---- 执行相位原语的本拍 20 ms 切片（CTC 跟踪，与专家一致）----
        for k in range(n_sub):
            t_loc = min(t_phase + k * ctrl.dt, float(seg.t[-1]))
            q_d, dq_d, ddq_d = seg.sample(t_loc)
            ddq_cmd = ddq_d + ctrl.kd * (dq_d - ctrl.dq) + ctrl.kp * (q_d - ctrl.q)
            tau = fast.inverse_dynamics(ctrl.q, ctrl.dq, ddq_cmd, friction=True)
            tau = np.clip(tau, -ctrl.tau_max, ctrl.tau_max)
            ctrl._step(tau, grip_cmd)
        t_phase += POLICY_DT
        if hook is not None:
            hook(step, model, data, ctrl, phase, t_phase)
        # 缩回段结束即收工（省时间）
        if phase == len(ST.PHASES) - 1 and t_phase > ST.PHASES[-1].duration + 0.2:
            break

    p_obj = data.xpos[obj_id].copy()
    err = float(np.hypot(p_obj[0] - sc["place_xy"][0], p_obj[1] - sc["place_xy"][1]))
    on_table = bool(abs(p_obj[2] - (SC.TABLE_TOP + SC.OBJ_SIZE / 2)) < 0.015)
    ok = bool(err < 0.025 and on_table)
    out = {"success": ok, "place_error_m": err, "object_final": p_obj.tolist(),
           "on_table": on_table, "steps": step + 1, "final_phase": int(phase),
           "phases": phase_rec}
    if label_fn is not None:
        out["obs"] = np.asarray(obs_rec, np.float32)
        out["phase_label"] = np.asarray(lab_rec, np.int64)
    return out
