"""控制器：计算力矩控制（CTC）与 PD+重力补偿（任意自由度）。"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .trajectory import Trajectory


def _as_vec(x, n: int) -> np.ndarray:
    return np.full(n, float(x)) if np.isscalar(x) else np.asarray(x, dtype=float).reshape(n)


def compute_torque(model, q: Sequence[float], dq: Sequence[float],
                   q_d: Sequence[float], dq_d: Sequence[float], ddq_d: Sequence[float],
                   kp: float | Sequence[float] = 100.0, kd: float | Sequence[float] = 20.0,
                   mode: str = "ctc", friction: bool = False) -> np.ndarray:
    """关节力矩。

    mode='ctc'  ：τ = M(q)·[q̈d + Kd·ė + Kp·e] + C·q̇ + G（计算力矩控制/反馈线性化）
    mode='pd_g' ：τ = Kp·e + Kd·ė + G（PD + 重力补偿，工程常用简化形式）
    """
    n = model.n
    e = _as_vec(q_d, n) - np.asarray(q, dtype=float).reshape(n)
    de = _as_vec(dq_d, n) - np.asarray(dq, dtype=float).reshape(n)
    kp_v, kd_v = _as_vec(kp, n), _as_vec(kd, n)
    if mode == "ctc":
        ddq_cmd = _as_vec(ddq_d, n) + kd_v * de + kp_v * e
        return model.inverse_dynamics(q, dq, ddq_cmd, friction=friction)
    if mode == "pd_g":
        return kp_v * e + kd_v * de + model.gravity(q)
    raise ValueError(f"未知控制器模式: {mode}")


def tracking_simulation(model, traj: Trajectory, kp: float | Sequence[float] = 100.0,
                        kd: float | Sequence[float] = 20.0, mode: str = "ctc",
                        friction: bool = False, tau_max: Optional[float | Sequence[float]] = None,
                        integrator: str = "rk4",
                        substeps: int = 1,
                        ki: float | Sequence[float] = 0.0,
                        i_limit: Optional[float | Sequence[float]] = None,
                        friction_ff_on_reference: bool = False,
                        fk_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                        q0: Sequence[float] | None = None) -> Dict[str, np.ndarray]:
    """闭环跟踪仿真。

    控制器周期 = 轨迹采样周期 dt；被控对象积分步长 = dt / substeps
    （力矩在控制周期内零阶保持，模拟真实数字控制器）。
    ki > 0 时加入积分项（消除库仑摩擦造成的静差；i_limit 为抗饱和限幅）。
    """
    n = model.n
    dt = float(traj.t[1] - traj.t[0])
    h = dt / max(int(substeps), 1)
    N = len(traj.t)
    q = np.zeros((N, n))
    dq = np.zeros((N, n))
    tau_hist = np.zeros((N, n))
    q[0] = np.asarray(traj.q[0], dtype=float) if q0 is None else np.asarray(q0, dtype=float)
    tau_lim = None if tau_max is None else _as_vec(tau_max, n)
    ki_v = _as_vec(ki, n)
    i_lim = None if i_limit is None else _as_vec(i_limit, n)
    i_acc = np.zeros(n)

    for i in range(N - 1):
        ti = float(traj.t[i])
        qi, dqi = q[i].copy(), dq[i].copy()
        qd, dqd, ddqd = traj.sample(ti)
        tau = compute_torque(model, qi, dqi, qd, dqd, ddqd, kp=kp, kd=kd, mode=mode,
                             friction=friction and not friction_ff_on_reference)
        if friction and friction_ff_on_reference:
            # 摩擦前馈改用**参考速度**（与 Simulink/工程实现一致：避免零速附近 sign 抖动，
            # 且不会把测量噪声带进力矩）
            tau = tau + model.friction_torque(dqd)
        if np.any(ki_v > 0):                      # 积分项（抗饱和）
            i_acc = i_acc + (traj.q[i] - qi) * dt
            if i_lim is not None:
                i_acc = np.clip(i_acc, -i_lim, i_lim)
            tau = tau + ki_v * i_acc
        if tau_lim is not None:
            tau = np.clip(tau, -tau_lim, tau_lim)
        tau_hist[i] = tau

        def f(qq: np.ndarray, dqq: np.ndarray):
            return dqq, model.forward_dynamics(qq, dqq, tau, friction=friction)

        q_next, dq_next = qi, dqi
        for _ in range(max(int(substeps), 1)):
            if integrator == "euler":
                k1v, k1a = f(q_next, dq_next)
                q_next = q_next + h * k1v
                dq_next = dq_next + h * k1a
            else:
                k1v, k1a = f(q_next, dq_next)
                k2v, k2a = f(q_next + 0.5 * h * k1v, dq_next + 0.5 * h * k1a)
                k3v, k3a = f(q_next + 0.5 * h * k2v, dq_next + 0.5 * h * k2a)
                k4v, k4a = f(q_next + h * k3v, dq_next + h * k3a)
                q_next = q_next + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
                dq_next = dq_next + h / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a)
        q[i + 1], dq[i + 1] = q_next, dq_next

    qd_end, dqd_end, ddqd_end = traj.sample(float(traj.t[-1]))
    tau_hist[-1] = compute_torque(model, q[-1], dq[-1], qd_end, dqd_end, ddqd_end,
                                  kp=kp, kd=kd, mode=mode, friction=friction)
    out = {
        "t": traj.t.copy(), "q": q, "dq": dq, "tau": tau_hist,
        "e": traj.q - q, "de": traj.dq - dq,
    }
    if fk_fn is not None:
        out["xy"] = np.array([fk_fn(qq) for qq in q])
    return out
