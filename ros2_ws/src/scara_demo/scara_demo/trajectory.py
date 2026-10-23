"""轨迹生成与 JointTrajectory 消息构造。

两种轨迹：
  1) joint_quintic —— 关节空间五次多项式（MoveJ）：起末速度/加速度为零；
  2) move_line —— 笛卡尔直线（MoveL）：直线上取 via 点逐点 IK，再套五次时间律。
MoveL 时每个 via 点的 q 用"上一点做种子"选肘部分支，避免分支跳变。
"""

from __future__ import annotations

import math
from typing import Sequence

from . import kinematics as K


def quintic(tau: float) -> tuple[float, float, float]:
    """返回 (s, ds/dτ, d²s/dτ²) 对归一化时间 τ∈[0,1]。"""
    tau = max(0.0, min(1.0, tau))
    s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5
    ds = 30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4
    dds = 60 * tau - 180 * tau ** 2 + 120 * tau ** 3
    return s, ds, dds


def joint_quintic(q0: Sequence[float], q1: Sequence[float], T: float,
                  dt: float = 0.02) -> list[tuple[float, list[float]]]:
    """关节空间五次多项式采样点 [(t, q)]（t 从 0 开始）。"""
    n = max(int(round(T / dt)), 1)
    out = []
    for k in range(n + 1):
        t = T * k / n
        s, _, _ = quintic(t / T if T > 0 else 1.0)
        q = [float(a) + (float(b) - float(a)) * s for a, b in zip(q0, q1)]
        out.append((t, q))
    return out


def move_line(q0: Sequence[float], xyz: Sequence[float], T: float,
              yaw: float = 0.0, dt: float = 0.02, n_via: int = 48
              ) -> list[tuple[float, list[float]]]:
    """笛卡尔直线：从 FK(q0) 沿直线到 xyz，逐点 IK + 五次时间律。

    两个必须处理的连续性细节（否则 JointTrajectoryController 会因首点跳变而中止）：
      1) 首点直接取当前关节角 q0（不能用 IK 重算，否则 q4 会被 yaw 目标瞬间改写）；
      2) yaw 从当前值沿路径线性过渡到目标值（q4 = yaw - q1 - q2 因此连续）。
    """
    p0 = K.fk(q0)[:3]
    yaw0 = K.fk(q0)[3]
    dyaw = ((float(yaw) - yaw0 + math.pi) % (2.0 * math.pi)) - math.pi
    p1 = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    q_prev = [float(v) for v in q0]
    via = [[float(v) for v in q0]]          # 首点 = 当前状态
    for i in range(1, n_via + 1):
        s = i / n_via
        pt = [p0[j] + (p1[j] - p0[j]) * s for j in range(3)]
        q = K.ik(pt[0], pt[1], pt[2], yaw=yaw0 + dyaw * s, q_seed=q_prev)
        if q is None:
            raise ValueError(f"MoveL 目标不可达：{pt}")
        via.append(q)
        q_prev = q
    n = max(int(round(T / dt)), 1)
    out = []
    for k in range(n + 1):
        t = T * k / n
        s, _, _ = quintic(t / T if T > 0 else 1.0)
        idx = s * n_via
        i0 = min(int(math.floor(idx)), n_via)
        i1 = min(i0 + 1, n_via)
        w = idx - i0
        q = [via[i0][j] * (1 - w) + via[i1][j] * w for j in range(4)]
        out.append((t, q))
    return out


def to_msg(points: Sequence[tuple[float, Sequence[float]]], joint_names: Sequence[str]):
    """把 [(t, q)] 转成 trajectory_msgs/JointTrajectory。"""
    from builtin_interfaces.msg import Duration
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    msg = JointTrajectory()
    msg.joint_names = list(joint_names)
    for t, q in points:
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in q]
        sec = int(math.floor(t))
        nsec = int(round((t - sec) * 1e9))
        if nsec >= 1_000_000_000:
            sec += 1
            nsec -= 1_000_000_000
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)
        msg.points.append(pt)
    return msg
