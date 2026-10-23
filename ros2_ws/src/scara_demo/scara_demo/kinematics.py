"""SCARA-teach-4RPRR 的运动学（与 python/armlab/spec.py 完全同一套公式）。

关节约定（与 URDF 一致）：
    joint1 = q1（绕 +z）
    joint2 = q2（绕 +z）
    joint3 = q3（移动副，轴 -z；q3 增大则末端下降）
    joint4 = q4（绕 +z，工具回转）

正运动学：
    x = a1 cos q1 + a2 cos(q1+q2)
    y = a1 sin q1 + a2 sin(q1+q2)
    z = d1 - q3
    yaw = q1 + q2 + q4

逆运动学（余弦定理 + 方位角减去内角）：
    c2 = (r^2 - a1^2 - a2^2) / (2 a1 a2)
    q2 = ±acos(c2)
    q1 = atan2(y, x) - atan2(a2 sin q2, a1 + a2 cos q2)
    q3 = d1 - z
    q4 = wrap(yaw - q1 - q2)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

A1 = 0.225
A2 = 0.225
D1 = 0.300
STROKE = 0.200

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4"]
JOINT_LIMITS = [(-2.268928, 2.268928), (-2.530727, 2.530727), (0.0, STROKE),
                (-2.0 * math.pi, 2.0 * math.pi)]


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def fk(q: Sequence[float]) -> tuple[float, float, float, float]:
    """正运动学：返回 (x, y, z, yaw)，单位 m / rad。"""
    q1, q2, q3, q4 = (float(v) for v in q)
    x = A1 * math.cos(q1) + A2 * math.cos(q1 + q2)
    y = A1 * math.sin(q1) + A2 * math.sin(q1 + q2)
    z = D1 - q3
    return x, y, z, wrap(q1 + q2 + q4)


def reachable(x: float, y: float, z: float) -> bool:
    r = math.hypot(x, y)
    return (abs(A1 - A2) - 1e-9 <= r <= A1 + A2 + 1e-9
            and 0.0 - 1e-9 <= D1 - z <= STROKE + 1e-9)


def ik(x: float, y: float, z: float, yaw: float = 0.0,
       q_seed: Optional[Sequence[float]] = None) -> Optional[list[float]]:
    """闭式逆解：返回与 q_seed 最接近的肘部分支；不可达返回 None。"""
    r2 = x * x + y * y
    c2 = (r2 - A1 * A1 - A2 * A2) / (2.0 * A1 * A2)
    if abs(c2) > 1.0 + 1e-9:
        return None
    c2 = max(-1.0, min(1.0, c2))
    q3 = D1 - z
    if q3 < -1e-9 or q3 > STROKE + 1e-9:
        return None
    cands = []
    for elbow in (-1.0, 1.0):
        q2 = elbow * math.acos(c2)
        q1 = math.atan2(y, x) - math.atan2(A2 * math.sin(q2), A1 + A2 * math.cos(q2))
        q1 = wrap(q1)
        q4 = wrap(yaw - q1 - q2)
        cands.append([q1, q2, max(0.0, min(STROKE, q3)), q4])
    if q_seed is None:
        return cands[0]
    seed = [float(v) for v in q_seed]

    def dist(q):
        return max(abs(wrap(q[i] - seed[i])) for i in range(4))

    best = min(cands, key=dist)
    if not within_limits(best):
        alt = [c for c in cands if within_limits(c)]
        if not alt:
            return None
        best = min(alt, key=dist)
    return best


def within_limits(q: Sequence[float]) -> bool:
    return all(lo - 1e-6 <= float(v) <= hi + 1e-6
               for v, (lo, hi) in zip(q, JOINT_LIMITS))


def joint_state_dict(names: Sequence[str], positions: Sequence[float]) -> dict[str, float]:
    return {str(n): float(p) for n, p in zip(names, positions)}


def q_from_joint_state(state: dict[str, float]) -> Optional[list[float]]:
    try:
        return [state[n] for n in JOINT_NAMES]
    except KeyError:
        return None
