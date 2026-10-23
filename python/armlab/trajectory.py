"""轨迹生成：关节空间五次多项式、笛卡尔直线（2 轴）、路径点链。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .kinematics import Planar2R


@dataclass
class Trajectory:
    """采样好的轨迹（任意自由度）。"""

    t: np.ndarray                 # (N,)
    q: np.ndarray                 # (N, n)
    dq: np.ndarray                # (N, n)
    ddq: np.ndarray               # (N, n)
    xy: Optional[np.ndarray] = None       # (N, 2) 末端实际笛卡尔轨迹（可选）
    xy_ref: Optional[np.ndarray] = None   # (N, 2) 期望路径（直线时与 xy 不同）
    meta: dict = field(default_factory=dict)
    # 精确采样函数（解析轨迹专用）：sample_fn(t) -> (q, dq, ddq)
    # 没有它时退化为对采样序列做线性插值（会产生 O(dt) 参考误差）
    sample_fn: Optional[Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None

    @property
    def n(self) -> int:
        return int(self.q.shape[1])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """采样 (q, dq, ddq)。解析轨迹用闭式采样，否则线性插值。"""
        t = float(np.clip(t, self.t[0], self.t[-1]))
        if self.sample_fn is not None:
            return self.sample_fn(t)
        q = np.array([np.interp(t, self.t, self.q[:, i]) for i in range(self.n)])
        dq = np.array([np.interp(t, self.t, self.dq[:, i]) for i in range(self.n)])
        ddq = np.array([np.interp(t, self.t, self.ddq[:, i]) for i in range(self.n)])
        return q, dq, ddq

    def metrics(self) -> dict:
        out = {
            "duration": self.duration,
            "max_joint_speed": float(np.max(np.abs(self.dq))) if len(self.t) else 0.0,
            "max_joint_accel": float(np.max(np.abs(self.ddq))) if len(self.t) else 0.0,
            "q_start": self.q[0].tolist(),
            "q_end": self.q[-1].tolist(),
        }
        if self.xy is not None and self.xy_ref is not None:
            out["max_path_error"] = float(np.max(np.linalg.norm(self.xy - self.xy_ref, axis=1)))
        return out

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        header = "t," + ",".join(f"q{i + 1}" for i in range(self.n)) + \
            "," + ",".join(f"dq{i + 1}" for i in range(self.n)) + \
            "," + ",".join(f"ddq{i + 1}" for i in range(self.n))
        data = np.column_stack([self.t, self.q, self.dq, self.ddq])
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        return path


# ---------------------------------------------------------------------------
# 五次多项式（q0 → q1，首末速度/加速度为零）
# ---------------------------------------------------------------------------

def quintic_scalar(s0: float, s1: float, T: float) -> Tuple[Callable, Callable, Callable]:
    h = s1 - s0

    def s(t):
        tau = np.clip(np.asarray(t, dtype=float) / T, 0.0, 1.0)
        return s0 + h * (10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5)

    def ds(t):
        tau = np.clip(np.asarray(t, dtype=float) / T, 0.0, 1.0)
        return h * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / T

    def dds(t):
        tau = np.clip(np.asarray(t, dtype=float) / T, 0.0, 1.0)
        return h * (60 * tau - 180 * tau ** 2 + 120 * tau ** 3) / (T ** 2)

    return s, ds, dds


def joint_quintic(q0: Sequence[float], q1: Sequence[float], T: float, dt: float) -> Trajectory:
    """逐关节五次多项式轨迹（多轴通用）。"""
    q0 = np.asarray(q0, dtype=float).reshape(-1)
    q1 = np.asarray(q1, dtype=float).reshape(-1)
    n = q0.size
    N = int(round(T / dt)) + 1
    t = np.linspace(0.0, T, N)
    q = np.zeros((N, n))
    dq = np.zeros((N, n))
    ddq = np.zeros((N, n))
    fs = []
    for i in range(n):
        s, ds, dds = quintic_scalar(q0[i], q1[i], T)
        q[:, i], dq[:, i], ddq[:, i] = s(t), ds(t), dds(t)
        fs.append((s, ds, dds))

    def sample_fn(tt: float):
        return (np.array([f[0](tt) for f in fs]),
                np.array([f[1](tt) for f in fs]),
                np.array([f[2](tt) for f in fs]))

    return Trajectory(t=t, q=q, dq=dq, ddq=ddq, meta={"type": "joint_quintic", "T": T},
                      sample_fn=sample_fn)


# ---------------------------------------------------------------------------
# 笛卡尔直线（2 轴平面臂：直线路径 + 五次时间标度）
# ---------------------------------------------------------------------------

def cartesian_line(solver: Planar2R, p0: Sequence[float], p1: Sequence[float],
                   T: float, dt: float, q_seed: Sequence[float] | None = None,
                   elbow: str | None = None) -> Trajectory:
    """末端沿直线 p0 → p1 运动；关节角由 IK 求解，速度由 J⁻¹ 映射。"""
    p0 = np.asarray(p0, dtype=float).reshape(2)
    p1 = np.asarray(p1, dtype=float).reshape(2)
    N = int(round(T / dt)) + 1
    t = np.linspace(0.0, T, N)
    s, ds, _ = quintic_scalar(0.0, 1.0, T)
    xy_ref = p0 + (s(t)[:, None] * (p1 - p0)[None, :])
    q = solver.ikine_path(xy_ref, q_start=q_seed)
    if elbow is not None:                       # 强制同一构型分支
        q = np.array([solver.ikine(xy_ref[i], q_ref=q[i], elbow=elbow) for i in range(N)])
    dq = np.zeros((N, 2))
    v = ds(t)[:, None] * (p1 - p0)[None, :]
    for i in range(N):
        J = solver.jacobian(q[i])
        dq[i] = np.linalg.solve(J.T @ J + 1e-10 * np.eye(2), J.T @ v[i])
    ddq = np.gradient(dq, t, axis=0) if N > 2 else np.zeros_like(dq)

    def q_at(tt: float) -> np.ndarray:
        seed = q[int(np.clip(round(tt / dt), 0, N - 1))]
        return solver.ikine(p0 + float(s(tt)) * (p1 - p0), q_ref=seed,
                            clamp_unreachable=True)

    def dq_at(tt: float) -> np.ndarray:
        J = solver.jacobian(q_at(tt))
        v_now = float(ds(tt)) * (p1 - p0)
        return np.linalg.solve(J.T @ J + 1e-10 * np.eye(2), J.T @ v_now)

    def sample_fn(tt: float):
        tc = float(np.clip(tt, 0.0, T))
        h = 1e-5
        t_p, t_m = min(tc + h, T), max(tc - h, 0.0)
        ddq_now = (dq_at(t_p) - dq_at(t_m)) / (t_p - t_m)
        return q_at(tc), dq_at(tc), ddq_now

    return Trajectory(t=t, q=q, dq=dq, ddq=ddq, xy=xy_ref.copy(), xy_ref=xy_ref,
                      meta={"type": "cartesian_line", "T": T, "p0": p0.tolist(),
                            "p1": p1.tolist()}, sample_fn=sample_fn)


def waypoint_path(solver: Planar2R, waypoints: Sequence[Sequence[float]],
                  T_seg: float, dt: float, q_start: Sequence[float] | None = None) -> Trajectory:
    """依次经过多个笛卡尔路径点（每段直线，段间近似停-走）。"""
    segs: List[Trajectory] = []
    q_cur = np.asarray(q_start, dtype=float).reshape(2) if q_start is not None else None
    for i in range(len(waypoints) - 1):
        seg = cartesian_line(solver, waypoints[i], waypoints[i + 1], T=T_seg, dt=dt, q_seed=q_cur)
        q_cur = seg.q[-1]
        segs.append(seg)
    t_all, q_all, dq_all, ddq_all, xy_all = [], [], [], [], []
    t_offset = 0.0
    for seg in segs:
        t_all.append(seg.t + t_offset)
        q_all.append(seg.q)
        dq_all.append(seg.dq)
        ddq_all.append(seg.ddq)
        xy_all.append(seg.xy)
        t_offset = t_all[-1][-1] + dt
    return Trajectory(t=np.concatenate(t_all), q=np.vstack(q_all), dq=np.vstack(dq_all),
                      ddq=np.vstack(ddq_all), xy=np.vstack(xy_all), xy_ref=np.vstack(xy_all),
                      meta={"type": "waypoint_path", "n_seg": len(segs)})


# ---------------------------------------------------------------------------
# 通用 N 自由度笛卡尔直线（MoveL）——SCARA / 6 轴通用
# ---------------------------------------------------------------------------

def cartesian_line_ndof(kin, ik_fn, p_target: Sequence[float], T: float, dt: float,
                        q_start: Sequence[float], n_via: int = 48) -> Trajectory:
    """末端沿**笛卡尔直线**从当前位置运动到 p_target（MoveL），五次多项式时间律。

    为什么需要它：关节空间插值（MoveJ）的末端轨迹会向一侧弓出（本机 SCARA 实测
    10~32 mm），专家与策略走同一条弓形路径时，策略学到的"平均路径"会进一步偏离，
    闭环把机械臂带出训练分布。MoveL 的路径是直线 ⇒ 数据流形简单、可复现。

    参数
    ----
    kin     : ChainKinematics（用 fk(q, link="link4") 取末端位置）
    ik_fn   : 调用签名 ik_fn(p_xyz, q_ref) -> q(4,) 或 None（不可达）
    p_target: 目标末端位置 (3,)
    T, dt   : 段时长与采样周期
    q_start : 起点关节角（起点位置由 FK 计算）
    n_via   : 直线上的 IK via 点数（越多越贴合直线）
    """
    q_start = np.asarray(q_start, dtype=float).reshape(-1)
    n = q_start.size
    p0 = np.asarray(kin.fk(q_start, link="link4")[1], dtype=float).reshape(3)
    p1 = np.asarray(p_target, dtype=float).reshape(3)

    # 1) 直线上逐点 IK（用上一点做参考，保证肘部分支连续）
    s_grid = np.linspace(0.0, 1.0, int(n_via))
    q_via = np.zeros((len(s_grid), n))
    q_prev = q_start.copy()
    for i, s in enumerate(s_grid):
        q = ik_fn(p0 + (p1 - p0) * s, q_prev)
        if q is None:
            raise ValueError(f"MoveL 目标不可达：{p0 + (p1 - p0) * s}")
        q_via[i] = q
        q_prev = q

    # 2) 关节角关于路径参数 s 的平滑曲线（三次样条；无 scipy 时退化为线性 + 滑动平均）
    try:
        from scipy.interpolate import CubicSpline
        spl = CubicSpline(s_grid, q_via, axis=0)

        def q_of_s(s):
            return np.asarray(spl(np.clip(s, 0.0, 1.0)), dtype=float)

        def dq_ds(s):
            return np.asarray(spl(np.clip(s, 0.0, 1.0), 1), dtype=float)

        def ddq_ds2(s):
            return np.asarray(spl(np.clip(s, 0.0, 1.0), 2), dtype=float)
    except Exception:
        q_smooth = q_via.copy()
        for _ in range(3):                      # 简单滑动平均平滑
            q_smooth[1:-1] = (q_smooth[:-2] + 2.0 * q_smooth[1:-1] + q_smooth[2:]) / 4.0

        def q_of_s(s):
            s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
            return np.array([np.interp(s, s_grid, q_smooth[:, j]) for j in range(n)])

        def dq_ds(s):
            s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
            return np.gradient(q_of_s(s), s, axis=0)

        def ddq_ds2(s):
            return np.zeros(n)

    # 3) 五次时间律 s(t)：位置/速度/加速度通过链式法则合成
    s_fun, ds_fun, dds_fun = quintic_scalar(0.0, 1.0, T)
    N = int(round(T / dt)) + 1
    t = np.linspace(0.0, T, N)
    sv, dsv, ddsv = s_fun(t), ds_fun(t), dds_fun(t)
    q = np.array([q_of_s(s) for s in sv])
    dq = np.array([dq_ds(s) * ds for s, ds in zip(sv, dsv)])
    ddq = np.array([ddq_ds2(s) * ds ** 2 + dq_ds(s) * dds
                    for s, ds, dds in zip(sv, dsv, ddsv)])
    xy = np.array([kin.fk(q[i], link="link4")[1] for i in range(N)])
    xy_ref = p0[None, :] + sv[:, None] * (p1 - p0)[None, :]

    def sample_fn(tt: float):
        tt = float(np.clip(tt, 0.0, T))
        s = float(s_fun(tt))
        ds = float(ds_fun(tt))
        dds = float(dds_fun(tt))
        return (q_of_s(s), dq_ds(s) * ds, ddq_ds2(s) * ds ** 2 + dq_ds(s) * dds)

    return Trajectory(t=t, q=q, dq=dq, ddq=ddq, xy=xy, xy_ref=xy_ref,
                      meta={"type": "cartesian_line_ndof", "T": T,
                            "p0": p0.tolist(), "p1": p1.tolist(),
                            "max_path_error": float(np.max(np.linalg.norm(xy - xy_ref, axis=1)))},
                      sample_fn=sample_fn)
