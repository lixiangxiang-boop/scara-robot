"""运动学：通用串链 FK / 几何雅可比 + 2 轴解析 IK + 通用数值 IK。

任意自由度使用 ``ChainKinematics``（为 6 轴预留）；
2 轴平面臂额外提供解析 IK/雅可比（``Planar2R``），既快又是教学/验证基准。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .spec import ArmSpec, JointSpec, LinkSpec, Vec3


# ---------------------------------------------------------------------------
# 基础旋转工具
# ---------------------------------------------------------------------------

def rot_axis(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues 公式：绕任意轴旋转 angle 的 3x3 旋转矩阵。"""
    a = np.asarray(axis, dtype=float).reshape(3)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    a = a / n
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]])
    # 使用 numpy 三角函数以支持复步长微分（complex-step differentiation）
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _as_q(q: Sequence[float], n: int) -> np.ndarray:
    """关节向量归一化；保持复数类型（供复步长微分使用）。"""
    arr = np.asarray(q)
    dtype = complex if np.iscomplexobj(arr) else float
    return arr.astype(dtype).reshape(n)


def quat_to_matrix(quat: Sequence[float]) -> np.ndarray:
    """四元数 (w, x, y, z) → 3x3 旋转矩阵。"""
    w, x, y, z = (float(v) for v in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    """把角度归一化到 (-pi, pi]。"""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi if isinstance(angle, np.ndarray) \
        else (angle + math.pi) % (2 * math.pi) - math.pi


def cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """3 维叉乘（等价 np.cross，但无 numpy 通用函数的开销）。"""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def _skew_matrix(axis: Sequence[float]) -> np.ndarray:
    a = np.asarray(axis, dtype=float).reshape(3)
    return np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


# ---------------------------------------------------------------------------
# 通用串链运动学
# ---------------------------------------------------------------------------

class ChainKinematics:
    """通用 N 自由度串链的正运动学与几何雅可比（世界系表达）。"""

    def __init__(self, spec: ArmSpec):
        self.spec = spec
        self.n = spec.dof
        # 每个连杆的父连杆索引（链式结构：i 的父为 i-1）
        self.parent = [i - 1 for i in range(len(spec.links))]
        self._fixed_R = [quat_to_matrix(link.quat) for link in spec.links]
        self._fixed_p = [np.asarray(link.pos, dtype=float) for link in spec.links]
        self._joint_axes = [np.asarray(link.joint.axis, dtype=float) if link.is_moving else None
                            for link in spec.links]
        # 预先归一化关节轴并缓存旋转矩阵的常数项：R = I + sinθ·K + (1-cosθ)·K²
        self._I3 = np.eye(3)
        self._joint_K: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
        for axis in self._joint_axes:
            if axis is None:
                self._joint_K.append(None)
            else:
                nrm = float(np.linalg.norm(axis))
                a = axis / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])
                K = _skew_matrix(a)
                self._joint_K.append((K, K @ K))
        # 关节序号（0..n-1），固定连杆为 None
        self.joint_slot: List[Optional[int]] = []
        k = 0
        for link in spec.links:
            if link.is_moving:
                self.joint_slot.append(k)
                k += 1
            else:
                self.joint_slot.append(None)
        # 影响每个连杆的"最远关节"序号：固定连杆继承父连杆；基座类连杆为 -1（雅可比为 0）
        self.max_slot: List[int] = []
        running = -1
        for slot in self.joint_slot:
            if slot is not None:
                running = slot
            self.max_slot.append(running)

    # ---------- 正运动学 ----------
    def _chain_pass(self, q: Sequence[float]):
        """一次遍历同时得到：各连杆 (R,p)、各关节世界轴与轴心点。"""
        q = _as_q(q, self.n)
        dtype = complex if np.iscomplexobj(q) else float
        R_par = np.eye(3, dtype=dtype)
        p_par = np.zeros(3, dtype=dtype)
        transforms: List[Tuple[np.ndarray, np.ndarray]] = []
        axes: List[np.ndarray] = []
        anchors: List[np.ndarray] = []
        for i, link in enumerate(self.spec.links):
            R_fixed = R_par @ self._fixed_R[i]
            p_fixed = p_par + R_par @ self._fixed_p[i]
            slot = self.joint_slot[i]
            if slot is not None:
                axes.append(R_fixed @ self._joint_axes[i])
                anchors.append(p_fixed)
                if self.spec.joints[slot].type == "prismatic":
                    # 移动副：沿关节轴平移 q（不产生旋转）
                    R = R_fixed
                    p_fixed = p_fixed + R_fixed @ (self._joint_axes[i] * q[slot])
                else:
                    K, K2 = self._joint_K[i]  # type: ignore[misc]
                    th = q[slot]
                    R = R_fixed @ (self._I3 + np.sin(th) * K + (1.0 - np.cos(th)) * K2)
            else:
                R = R_fixed
            transforms.append((R, p_fixed))
            R_par, p_par = R, p_fixed
        return transforms, axes, anchors

    def link_transforms(self, q: Sequence[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """返回每个连杆的 (R, p)（世界系）。q 长度 = dof，按关节顺序。"""
        return self._chain_pass(q)[0]

    def fk(self, q: Sequence[float], link: str | int = -1,
           point: Sequence[float] | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """指定连杆（默认末端）的位姿 (R, p)；point 为连杆系下的一点。"""
        idx = self._link_index(link)
        R, p = self.link_transforms(q)[idx]
        if point is not None:
            p = p + R @ np.asarray(point, dtype=float).reshape(3)
        return R, p

    def ee_pose(self, q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
        """末端工具点（无工具时取最后一个连杆的几何末端）的位姿。"""
        R, p = self.link_transforms(q)[-1]
        tip = self.tip_point()
        return R, p + R @ tip

    def ee_position(self, q: Sequence[float]) -> np.ndarray:
        return self.ee_pose(q)[1]

    def com_pose(self, q: Sequence[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """每个连杆质心的 (R, p)（世界系）。"""
        out = []
        tf = self.link_transforms(q)
        for (R, p), link in zip(tf, self.spec.links):
            out.append((R, p + R @ np.asarray(link.com, dtype=float)))
        return out

    def tip_point(self) -> np.ndarray:
        """末端点（在最后一个连杆坐标系下）。"""
        links = self.spec.links
        if not links[-1].is_moving:
            return np.asarray(links[-1].pos, dtype=float)
        geom = links[-1].geom or {}
        if geom.get("type") == "capsule" and geom.get("fromto") is not None:
            return np.asarray(geom["fromto"][3:6], dtype=float)
        return np.array([0.0, 0.0, 0.0])

    def joint_frames(self, q: Sequence[float]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """每个可动关节在世界系下的 (轴方向, 轴上一点)。"""
        _, axes, anchors = self._chain_pass(q)
        return list(zip(axes, anchors))

    # ---------- 几何雅可比 ----------
    def jacobian(self, q: Sequence[float], link: str | int = -1,
                 point: Sequence[float] | None = None) -> np.ndarray:
        """6xN 几何雅可比（世界系）：[线速度; 角速度]，点默认取连杆几何末端。"""
        idx = self._link_index(link)
        transforms, axes, anchors = self._chain_pass(q)
        R, p = transforms[idx]
        if point is None:
            point = self.tip_point() if idx == len(self.spec.links) - 1 else (0.0, 0.0, 0.0)
        target = p + R @ np.asarray(point, dtype=R.dtype).reshape(3)
        max_slot = self.max_slot[idx]
        J = np.zeros((6, self.n), dtype=complex if np.iscomplexobj(q) else float)
        for slot in range(max_slot + 1):
            axis, anchor = axes[slot], anchors[slot]
            if self.spec.joints[slot].type == "prismatic":
                J[0:3, slot] = axis
            else:
                J[0:3, slot] = cross3(axis, target - anchor)
                J[3:6, slot] = axis
        return J

    def all_link_jacobians(
            self, q: Sequence[float],
            points: Optional[Sequence[Optional[Sequence[float]]]] = None) -> np.ndarray:
        """一次遍历计算所有连杆的 6xN 雅可比（默认取各连杆质心），形状 (n_links, 6, N)。"""
        return self.com_states(q, points)[1]

    def com_states(self, q: Sequence[float],
                   points: Optional[Sequence[Optional[Sequence[float]]]] = None
                   ) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """一次遍历返回 (各连杆变换, 各指定点的 6xN 雅可比)；默认点 = 各连杆质心。"""
        transforms, axes, anchors = self._chain_pass(q)
        dtype = complex if np.iscomplexobj(q) else float
        out = np.zeros((len(self.spec.links), 6, self.n), dtype=dtype)
        for idx, link in enumerate(self.spec.links):
            max_slot = self.max_slot[idx]
            if max_slot < 0:
                continue
            R, p = transforms[idx]
            pt = link.com if points is None else (
                points[idx] if points[idx] is not None else link.com)
            target = p + R @ np.asarray(pt, dtype=dtype).reshape(3)
            for slot in range(max_slot + 1):
                axis, anchor = axes[slot], anchors[slot]
                if self.spec.joints[slot].type == "prismatic":
                    out[idx, 0:3, slot] = axis
                else:
                    out[idx, 0:3, slot] = cross3(axis, target - anchor)
                    out[idx, 3:6, slot] = axis
        return transforms, out

    def jacobian_com(self, q: Sequence[float], link: str | int) -> np.ndarray:
        """某连杆质心的几何雅可比（世界系）。"""
        idx = self._link_index(link)
        return self.jacobian(q, link=idx, point=self.spec.links[idx].com)

    def _link_index(self, link: str | int) -> int:
        if isinstance(link, int):
            return link if link >= 0 else len(self.spec.links) + link
        names = [lk.name for lk in self.spec.links]
        return names.index(link)

    # ---------- 常用度量 ----------
    def manipulability(self, q: Sequence[float], link: str | int = -1) -> float:
        J = self.jacobian(q, link=link)
        JJt = J @ J.T
        return float(np.sqrt(max(np.linalg.det(JJt), 0.0)))

    def plot_points(self, q: Sequence[float]) -> np.ndarray:
        """用于绘图的关节位置序列：基座 → 各关节 → 末端。"""
        tf = self.link_transforms(q)
        pts = [np.zeros(3)]
        for link, (R, p) in zip(self.spec.links, tf):
            if link.is_moving:
                pts.append(p)
        pts.append(self.ee_position(q))
        return np.asarray(pts)


def _unzip(frames: List[Tuple[np.ndarray, np.ndarray]]):
    axes = [f[0] for f in frames]
    anchors = [f[1] for f in frames]
    return axes, anchors


def jacobian_finite_difference(kin: ChainKinematics, q: Sequence[float],
                               link: str | int = -1,
                               point: Sequence[float] | None = None,
                               eps: float = 1e-6) -> np.ndarray:
    """有限差分雅可比（独立实现，用于验证解析结果）。"""
    q = np.asarray(q, dtype=float).reshape(kin.n)
    J = np.zeros((6, kin.n))
    idx = kin._link_index(link)
    if point is None:
        point = kin.tip_point() if idx == len(kin.spec.links) - 1 else (0.0, 0.0, 0.0)

    def pose(qq):
        R, p = kin.link_transforms(qq)[idx]
        p = p + R @ np.asarray(point, dtype=float)
        return R, p

    R0, p0 = pose(q)
    for j in range(kin.n):
        dq = np.zeros(kin.n)
        dq[j] = eps
        Rp, pp = pose(q + dq)
        Rm, pm = pose(q - dq)
        J[0:3, j] = (pp - pm) / (2 * eps)
        dR = (Rp - Rm) / (2 * eps)
        W = dR @ R0.T                      # 角速度的反对称矩阵
        J[3:6, j] = [W[2, 1], W[0, 2], W[1, 0]]
    return J


# ---------------------------------------------------------------------------
# 2 轴平面臂解析运动学
# ---------------------------------------------------------------------------

class UnreachableTarget(ValueError):
    """目标点超出工作空间。"""


@dataclass
class Planar2R:
    """2 自由度平面臂解析运动学（从 ArmSpec 提取，仅在构型匹配时可用）。"""

    L1: float
    L2: float
    q_limits: Optional[np.ndarray] = None      # (2,2) [lower, upper]
    tip_offset: float = 0.0                    # 末端工具长度（沿 link2 的 x）
    name: str = "arm2dof"

    # ---------- 构造 ----------
    @staticmethod
    def from_spec(spec: ArmSpec) -> "Planar2R":
        """从 ArmSpec 提取 2R 平面参数；构型不匹配时抛错（自动回退到通用模型）。"""
        if spec.dof != 2:
            raise ValueError(f"需要 2 自由度，当前 dof={spec.dof}")
        links = spec.moving_links
        for link in links:
            if link.joint.axis != (0.0, 0.0, 1.0):  # type: ignore[index]
                raise ValueError("Planar2R 要求关节轴为 +z")
            if abs(link.pos[1]) > 1e-12 or abs(link.pos[2]) > 1e-12:
                raise ValueError("Planar2R 要求连杆首尾相接且位于 x 轴")
            if abs(link.com[1]) > 1e-12 or abs(link.com[2]) > 1e-12:
                raise ValueError("Planar2R 解析动力学要求质心位于连杆轴线上")
        L1 = float(links[1].pos[0])
        L2 = float(_link_len(links[1]))
        tip = 0.0
        tools = [lk for lk in spec.links if not lk.is_moving and lk.name != "base"]
        if tools:  # 工具/负载固定在 link2 末端：其 x 位置相对关节 2 的距离 = L2 + tip
            tip = float(np.asarray(tools[-1].pos, dtype=float)[0]) - L2
        return Planar2R(L1=L1, L2=L2,
                        q_limits=spec.limits_array(),
                        tip_offset=tip, name=spec.name)

    # ---------- 正运动学 ----------
    def fkine(self, q: Sequence[float]) -> np.ndarray:
        """末端（含工具偏移）笛卡尔位置 [x, y]。"""
        q1, q2 = float(q[0]), float(q[1])
        L = self.L2 + self.tip_offset
        x = self.L1 * math.cos(q1) + L * math.cos(q1 + q2)
        y = self.L1 * math.sin(q1) + L * math.sin(q1 + q2)
        return np.array([x, y])

    def joint_positions(self, q: Sequence[float]) -> np.ndarray:
        """(3, 2)：基座、关节 2、末端。"""
        q1, q2 = float(q[0]), float(q[1])
        p2 = np.array([self.L1 * math.cos(q1), self.L1 * math.sin(q1)])
        return np.vstack([[0.0, 0.0], p2, self.fkine(q)])

    # ---------- 雅可比 / 度量 ----------
    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        q1, q2 = float(q[0]), float(q[1])
        L = self.L2 + self.tip_offset
        s1, c1 = math.sin(q1), math.cos(q1)
        s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
        return np.array([[-self.L1 * s1 - L * s12, -L * s12],
                         [self.L1 * c1 + L * c12, L * c12]])

    def det_jacobian(self, q: Sequence[float]) -> float:
        L = self.L2 + self.tip_offset
        return self.L1 * L * math.sin(float(q[1]))

    def manipulability(self, q: Sequence[float]) -> float:
        """Yoshikawa 可操作度 sqrt(det(J J^T)) = |L1·L·sin q2|。"""
        return abs(self.det_jacobian(q))

    def workspace_radii(self) -> Tuple[float, float]:
        """可达半径区间 [r_min, r_max]。"""
        L = self.L2 + self.tip_offset
        return (abs(self.L1 - L), self.L1 + L)

    def is_reachable(self, xy: Sequence[float], tol: float = 1e-12) -> bool:
        r = float(np.linalg.norm(np.asarray(xy, dtype=float)))
        r_min, r_max = self.workspace_radii()
        return (r_min - tol) <= r <= (r_max + tol)

    def static_torque(self, q: Sequence[float], fxy: Sequence[float]) -> np.ndarray:
        """末端静力映射：τ = J^T f（f 为末端**对外输出**的力，世界系）。

        若末端**受到**外力 f_ext（例如负载重力），维持静止所需关节力矩为 −J^T f_ext。
        """
        return self.jacobian(q).T @ np.asarray(fxy, dtype=float).reshape(2)

    # ---------- 逆运动学 ----------
    def ikine_all(self, xy: Sequence[float], clamp_unreachable: bool = False) -> np.ndarray:
        """返回全部解 (k, 2)，肘部上/下各一个。不可达时抛错或投影到边界。"""
        xy = np.asarray(xy, dtype=float).reshape(2)
        r = float(np.linalg.norm(xy))
        r_min, r_max = self.workspace_radii()
        if r > r_max or r < r_min:
            if not clamp_unreachable:
                raise UnreachableTarget(
                    f"目标 {xy} 半径 {r:.4f} m 不在 [{r_min:.4f}, {r_max:.4f}] m 内")
            if r < 1e-12:
                xy = np.array([r_min, 0.0])
            else:
                r_clamped = float(np.clip(r, r_min, r_max))
                xy = xy / r * r_clamped
            r = float(np.linalg.norm(xy))
        L = self.L2 + self.tip_offset
        c2 = (r * r - self.L1 ** 2 - L ** 2) / (2.0 * self.L1 * L)
        c2 = float(np.clip(c2, -1.0, 1.0))
        q2 = math.acos(c2)
        sols = []
        for sign in (1.0, -1.0):
            qq2 = sign * q2
            q1 = math.atan2(xy[1], xy[0]) - math.atan2(L * math.sin(qq2), self.L1 + L * math.cos(qq2))
            sols.append([wrap_to_pi(q1), wrap_to_pi(qq2)])
        out = np.asarray(sols, dtype=float)
        if abs(q2) < 1e-12:                 # 奇异：两解重合
            out = out[:1]
        return out

    def ikine(self, xy: Sequence[float], q_ref: Sequence[float] | None = None,
              elbow: str | None = None, check_limits: bool = True,
              clamp_unreachable: bool = False) -> np.ndarray:
        """给定位姿求关节角。

        elbow: None(默认，按 q_ref 或最小范数选解) | 'up'(q2>0) | 'down'(q2<0)
        """
        sols = self.ikine_all(xy, clamp_unreachable=clamp_unreachable)
        if elbow is not None:
            want = 1.0 if elbow == "up" else -1.0
            sols = np.asarray([s for s in sols if np.sign(s[1]) == want] or [sols[0]])
        if self.q_limits is not None and check_limits:
            ok = [s for s in sols
                  if all(lo - 1e-9 <= v <= hi + 1e-9
                         for v, (lo, hi) in zip(s, self.q_limits))]
            if ok:
                sols = np.asarray(ok)
        if len(sols) == 1:
            return sols[0].copy()
        if q_ref is None:
            return np.asarray(min(sols, key=lambda s: float(np.sum(s ** 2))), dtype=float)
        q_ref = np.asarray(q_ref, dtype=float).reshape(2)
        return np.asarray(min(sols, key=lambda s: float(np.max(np.abs(wrap_to_pi(s - q_ref))))),
                          dtype=float)

    def ikine_path(self, xy_path: Sequence[Sequence[float]],
                   q_start: Sequence[float] | None = None,
                   slack_detect: bool = True) -> np.ndarray:
        """沿路径逐个求 IK，保证解连续（用上一点作为参考）。"""
        pts = np.asarray(xy_path, dtype=float)
        q_prev = np.asarray(q_start, dtype=float).reshape(2) if q_start is not None else None
        out = np.zeros((len(pts), 2))
        for i, xy in enumerate(pts):
            q = self.ikine(xy, q_ref=q_prev, clamp_unreachable=True)
            if q_prev is not None and slack_detect:
                # 若跨越 ±pi 分支导致跳变，尝试另一解
                if np.max(np.abs(wrap_to_pi(q - q_prev))) > 0.5 * math.pi:
                    alt = self.ikine(xy, elbow="down" if q[1] > 0 else "up")
                    if np.max(np.abs(wrap_to_pi(alt - q_prev))) < np.max(np.abs(wrap_to_pi(q - q_prev))):
                        q = alt
            out[i] = q
            q_prev = q
        return out


def _link_len(link: LinkSpec) -> float:
    """从几何信息读取连杆长度（x 方向）。"""
    geom = link.geom or {}
    ft = geom.get("fromto")
    if geom.get("type") == "capsule" and ft and len(ft) == 6:
        return float(ft[3])
    return float(link.pos[0]) if link.pos[0] > 0 else 0.25


# ---------------------------------------------------------------------------
# 通用数值 IK（阻尼最小二乘，任意自由度；6 轴阶段使用）
# ---------------------------------------------------------------------------

class NumericalIK:
    """阻尼最小二乘 IK：位置 +（可选）姿态目标，支持关节限位与步长限制。"""

    def __init__(self, spec: ArmSpec, tol_pos: float = 1e-8, tol_rot: float = 1e-6,
                 damping: float = 1e-3, max_iter: int = 300, step_clip: float = 0.2):
        self.spec = spec
        self.kin = ChainKinematics(spec)
        self.tol_pos, self.tol_rot = tol_pos, tol_rot
        self.damping, self.max_iter, self.step_clip = damping, max_iter, step_clip
        self.limits = spec.limits_array()

    def solve(self, target_p: Sequence[float], target_R: np.ndarray | None = None,
              q0: Sequence[float] | None = None, pos_weight: float = 1.0,
              rot_weight: float = 0.0) -> Tuple[np.ndarray, Dict[str, float]]:
        n = self.spec.dof
        q = np.zeros(n) if q0 is None else np.clip(np.asarray(q0, float).reshape(n),
                                                   self.limits[:, 0], self.limits[:, 1])
        target_p = np.asarray(target_p, dtype=float).reshape(3)
        info = {"success": 0.0, "iters": 0.0, "pos_err": float("inf"), "rot_err": float("inf")}
        for it in range(self.max_iter):
            R, p = self.kin.ee_pose(q)
            e_p = target_p - p
            J = self.kin.jacobian(q)
            e = pos_weight * e_p
            Jw = pos_weight * J[0:3, :]
            if target_R is not None and rot_weight > 0:
                R_err = target_R @ R.T
                e_r = _rot_error_vector(R_err) * rot_weight
                e = np.concatenate([e, e_r])
                Jw = np.vstack([Jw, rot_weight * J[3:6, :]])
            pos_err = float(np.linalg.norm(e_p))
            rot_err = float(np.linalg.norm(e_r)) if target_R is not None and rot_weight > 0 else 0.0
            info.update(pos_err=pos_err, rot_err=rot_err, iters=float(it))
            if pos_err < self.tol_pos and rot_err < self.tol_rot:
                info["success"] = 1.0
                break
            A = Jw @ Jw.T + (self.damping ** 2) * np.eye(Jw.shape[0])
            dq = Jw.T @ np.linalg.solve(A, e)
            nrm = float(np.max(np.abs(dq)))
            if nrm > self.step_clip:
                dq = dq * (self.step_clip / nrm)
            q = q + dq
            # 关节限位：软限位（越界则截断到边界并停止该关节继续外推）
            q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        return q, info


def _rot_error_vector(R_err: np.ndarray) -> np.ndarray:
    """旋转矩阵误差 → 旋转矢量（轴角，世界系）。"""
    cos_theta = float(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    axis = np.array([R_err[2, 1] - R_err[1, 2],
                     R_err[0, 2] - R_err[2, 0],
                     R_err[1, 0] - R_err[0, 1]]) / (2.0 * math.sin(theta))
    return axis * theta
