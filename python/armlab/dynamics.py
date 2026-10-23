"""动力学：通用 N 自由度串链模型 + 2 轴解析闭式模型（统一 Model 接口）。

两个模型都能回答同样的问题（多轴扩展的关键接口）：
    mass_matrix(q) / gravity(q) / coriolis_torque(q,dq) / coriolis_matrix(q,dq)
    inverse_dynamics(q,dq,ddq) / forward_dynamics(q,dq,tau)
    kinetic_energy / potential_energy / energy

实现策略
--------
- ``ChainModel``：任意自由度，M/G 用几何雅可比精确计算；科氏项用 Christoffel 符号
  （∂M/∂q 用中心差分，误差 ~1e-10），并由 MuJoCo 交叉验证；
- ``ClosedForm2R``：2 轴平面臂解析闭式（Spong 标准形式，支持任意重力方向），
  作为教学基准与精度参照；
- ``make_model(spec)``：构型匹配时自动选择解析模型，否则回退通用模型。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from .kinematics import ChainKinematics, Planar2R, rot_axis, wrap_to_pi
from .spec import ArmSpec, LinkSpec


# ---------------------------------------------------------------------------
# 通用串链模型（任意自由度）
# ---------------------------------------------------------------------------

class ChainModel:
    """通用 N 自由度动力学模型（世界系重力，支持 2~6 轴及更多）。"""

    def __init__(self, spec: ArmSpec, friction_smooth: float = 1e-3):
        """friction_smooth：库仑摩擦 tanh 平滑宽度（速度尺度，rad/s 或 m/s）。

        取值影响"粘滞区"的等效阻尼 = Fc/smooth：
        - 1e-3（默认）→ 等效阻尼极大，低惯量轴会出现明显静差（e≈Fc/(I·Kp)）；
        - 0.02 → 接近真实摩擦的"缓变"过渡，静差小一个数量级。
        仿真对比时两侧（被控对象/控制器前馈）必须取同一值。
        """
        self.spec = spec
        self.friction_smooth = float(friction_smooth)
        self.kin = ChainKinematics(spec)
        self.n = spec.dof
        self._idx_all = list(range(len(spec.links)))
        self._masses = np.array([lk.mass for lk in spec.links])
        self._inertia = [lk.inertia_matrix() for lk in spec.links]
        self._gravity = np.asarray(spec.gravity, dtype=float)
        self._armature = np.array([j.armature for j in spec.joints])
        self._damping = np.array([j.damping for j in spec.joints])
        self._frictionloss = np.array([j.frictionloss for j in spec.joints])

    # ---------- 质量矩阵 / 重力 / 科氏 ----------
    def mass_matrix(self, q: Sequence[float]) -> np.ndarray:
        dtype = complex if np.iscomplexobj(q) else float
        M = np.zeros((self.n, self.n), dtype=dtype)
        tf, Js = self.kin.com_states(q)
        for idx, link in enumerate(self.spec.links):
            if link.mass <= 0.0:
                continue
            R, _ = tf[idx]
            Jv, Jw = Js[idx, 0:3, :], Js[idx, 3:6, :]
            Iw = R @ self._inertia[idx] @ R.T
            M += link.mass * (Jv.T @ Jv) + Jw.T @ Iw @ Jw
        M += np.diag(self._armature)
        return 0.5 * (M + M.T)          # 数值对称化

    def gravity(self, q: Sequence[float]) -> np.ndarray:
        """重力补偿力矩 τ_g = ∂V/∂q（直接用于控制器）。"""
        G = np.zeros(self.n)
        _, Js = self.kin.com_states(q)
        for idx, link in enumerate(self.spec.links):
            if link.mass <= 0.0:
                continue
            G -= link.mass * (Js[idx, 0:3, :].T @ self._gravity)
        return G

    def dM_dq(self, q: Sequence[float], eps: float = 1e-8) -> np.ndarray:
        """∂M/∂q，形状 (n, n, n)：T[i, j, k] = ∂M_ij/∂q_k。

        用复步长微分（complex-step）：一次 M 计算/方向，精度接近机器精度。
        """
        q = np.asarray(q, dtype=float).reshape(self.n)
        T = np.zeros((self.n, self.n, self.n))
        for k in range(self.n):
            qc = q.astype(complex)
            qc[k] += 1j * eps
            T[:, :, k] = self.mass_matrix(qc).imag / eps
        return T

    def coriolis_matrix(self, q: Sequence[float], dq: Sequence[float]) -> np.ndarray:
        """科氏/离心矩阵 C(q,dq)（Christoffel 符号）。"""
        dq = np.asarray(dq, dtype=float).reshape(self.n)
        T = self.dM_dq(q)
        A = 0.5 * np.einsum("ijk,k->ij", T, dq)
        B = 0.5 * np.einsum("ikj,k->ij", T, dq)
        C = 0.5 * np.einsum("jki,k->ij", T, dq)
        return A + B - C

    def coriolis_torque(self, q: Sequence[float], dq: Sequence[float]) -> np.ndarray:
        """C(q,dq)·dq（比显式矩阵更快）。"""
        dq = np.asarray(dq, dtype=float).reshape(self.n)
        T = self.dM_dq(q)
        t1 = np.einsum("ijk,j,k->i", T, dq, dq)
        t2 = 0.5 * np.einsum("jki,j,k->i", T, dq, dq)
        return t1 - t2

    # ---------- 摩擦 ----------
    def friction_torque(self, dq: Sequence[float], smooth: Optional[float] = None) -> np.ndarray:
        """粘性 + 库仑摩擦（库仑项用 tanh 平滑，便于数值积分）。"""
        dq = np.asarray(dq, dtype=float).reshape(self.n)
        if smooth is None:
            smooth = self.friction_smooth
        return self._damping * dq + self._frictionloss * np.tanh(dq / smooth)

    # ---------- 动力学方程 ----------
    def bias(self, q: Sequence[float], dq: Sequence[float]) -> np.ndarray:
        """C(q,dq)dq + G(q)（MuJoCo 里的 qfrc_bias）。"""
        return self.coriolis_torque(q, dq) + self.gravity(q)

    def inverse_dynamics(self, q: Sequence[float], dq: Sequence[float], ddq: Sequence[float],
                         friction: bool = False) -> np.ndarray:
        tau = self.mass_matrix(q) @ np.asarray(ddq, dtype=float).reshape(self.n)
        tau = tau + self.bias(q, dq)
        if friction:
            tau = tau + self.friction_torque(dq)
        return tau

    def forward_dynamics(self, q: Sequence[float], dq: Sequence[float],
                         tau: Sequence[float], friction: bool = False) -> np.ndarray:
        M = self.mass_matrix(q)
        rhs = np.asarray(tau, dtype=float).reshape(self.n) - self.bias(q, dq)
        if friction:
            rhs = rhs - self.friction_torque(dq)
        return np.linalg.solve(M, rhs)

    # ---------- 能量 ----------
    def kinetic_energy(self, q: Sequence[float], dq: Sequence[float]) -> float:
        dq = np.asarray(dq, dtype=float).reshape(self.n)
        return float(0.5 * dq @ self.mass_matrix(q) @ dq)

    def potential_energy(self, q: Sequence[float]) -> float:
        V = 0.0
        for idx, link in enumerate(self.spec.links):
            if link.mass <= 0.0:
                continue
            _, p = self.kin.com_pose(q)[idx]
            V -= link.mass * float(self._gravity @ p)
        return V

    def energy(self, q: Sequence[float], dq: Sequence[float]) -> Dict[str, float]:
        T = self.kinetic_energy(q, dq)
        V = self.potential_energy(q)
        return {"kinetic": T, "potential": V, "total": T + V}


# ---------------------------------------------------------------------------
# 2 轴解析闭式模型
# ---------------------------------------------------------------------------

@dataclass
class ClosedForm2R:
    """2 轴平面臂解析动力学（Spong 标准形式；重力方向任意）。"""

    L1: float
    L2: float          # 含工具偏移的等效第二连杆长度
    m1: float
    m2: float          # link2 + 工具折叠后的等效质量
    lc1: float
    lc2: float
    I1: float
    I2: float
    gravity_vec: Tuple[float, float] = (0.0, -9.81)   # 平面内重力加速度矢量
    damping: Tuple[float, float] = (0.0, 0.0)
    frictionloss: Tuple[float, float] = (0.0, 0.0)
    armature: Tuple[float, float] = (0.0, 0.0)
    name: str = "arm2dof"

    def __post_init__(self) -> None:
        self.n = 2
        self._g = np.asarray(self.gravity_vec, dtype=float).reshape(2)
        self._b = np.asarray(self.damping, dtype=float).reshape(2)
        self._fc = np.asarray(self.frictionloss, dtype=float).reshape(2)
        self._arm = np.asarray(self.armature, dtype=float).reshape(2)
        self._beta = self.m2 * self.L1 * self.lc2      # 耦合系数 β

    @staticmethod
    def from_spec(spec: ArmSpec) -> "ClosedForm2R":
        p2r = Planar2R.from_spec(spec)                 # 构型检查在此完成
        links = spec.moving_links
        l1, l2 = links[0], links[1]
        m1, lc1, I1 = l1.mass, float(l1.com[0]), float(l1.inertia[2][2])
        m2, lc2, I2 = l2.mass, float(l2.com[0]), float(l2.inertia[2][2])
        # 末端工具/负载折叠进 link2（点质量 → 平行轴定理）
        tools = [lk for lk in spec.links if not lk.is_moving and lk.name != "base"]
        for tool in tools:
            if tool.mass <= 0.0:
                continue
            px = float(np.asarray(tool.pos, dtype=float)[0]) + float(tool.com[0])
            mp = tool.mass
            m_new = m2 + mp
            lc_new = (m2 * lc2 + mp * px) / m_new
            I_tool = float(tool.inertia_matrix()[2, 2])
            I_new = (I2 + m2 * (lc_new - lc2) ** 2
                     + I_tool + mp * (px - lc_new) ** 2)
            m2, lc2, I2 = m_new, lc_new, I_new
        return ClosedForm2R(
            L1=p2r.L1, L2=p2r.L2 + p2r.tip_offset,
            m1=m1, m2=m2, lc1=lc1, lc2=lc2, I1=I1, I2=I2,
            gravity_vec=(float(spec.gravity[0]), float(spec.gravity[1])),
            damping=(spec.joints[0].damping, spec.joints[1].damping),
            frictionloss=(spec.joints[0].frictionloss, spec.joints[1].frictionloss),
            armature=(spec.joints[0].armature, spec.joints[1].armature),
            name=spec.name,
        )

    # ---------- 核心项 ----------
    def mass_matrix(self, q: Sequence[float]) -> np.ndarray:
        q2 = float(q[1])
        c2 = math.cos(q2)
        M11 = self.I1 + self.I2 + self.m1 * self.lc1 ** 2 + \
            self.m2 * (self.L1 ** 2 + self.lc2 ** 2 + 2 * self.L1 * self.lc2 * c2)
        M12 = self.I2 + self.m2 * (self.lc2 ** 2 + self.L1 * self.lc2 * c2)
        M22 = self.I2 + self.m2 * self.lc2 ** 2
        M = np.array([[M11, M12], [M12, M22]])
        return M + np.diag(self._arm)

    def coriolis_matrix(self, q: Sequence[float], dq: Sequence[float]) -> np.ndarray:
        dq1, dq2 = float(dq[0]), float(dq[1])
        h = self._beta * math.sin(float(q[1]))
        return np.array([[-h * dq2, -h * (dq1 + dq2)],
                         [h * dq1, 0.0]])

    def coriolis_torque(self, q: Sequence[float], dq: Sequence[float]) -> np.ndarray:
        dq1, dq2 = float(dq[0]), float(dq[1])
        h = self._beta * math.sin(float(q[1]))
        return np.array([-h * (2 * dq1 * dq2 + dq2 ** 2), h * dq1 ** 2])

    def gravity(self, q: Sequence[float]) -> np.ndarray:
        q1, q2 = float(q[0]), float(q[1])
        s1, c1 = math.sin(q1), math.cos(q1)
        s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
        e1p = np.array([-s1, c1])       # ∂e1/∂q1
        e12p = np.array([-s12, c12])    # ∂e12/∂q1 = ∂e12/∂q2
        A = self.m1 * self.lc1 + self.m2 * self.L1
        B = self.m2 * self.lc2
        tau1 = -(A * float(self._g @ e1p) + B * float(self._g @ e12p))
        tau2 = -B * float(self._g @ e12p)
        return np.array([tau1, tau2])

    def dM_dq(self, q: Sequence[float]) -> np.ndarray:
        # ∂/∂q1 全为 0；∂M11/∂q2 = -2β sin q2，∂M12/∂q2 = -β sin q2（β = m2·L1·lc2）
        d2 = -2.0 * self._beta * math.sin(float(q[1]))
        T = np.zeros((2, 2, 2))
        T[0, 0, 1] = d2
        T[0, 1, 1] = T[1, 0, 1] = 0.5 * d2
        return T

    # ---------- 摩擦与方程 ----------
    def friction_torque(self, dq: Sequence[float], smooth: float = 1e-3) -> np.ndarray:
        dq = np.asarray(dq, dtype=float).reshape(2)
        return self._b * dq + self._fc * np.tanh(dq / smooth)

    def bias(self, q: Sequence[float], dq: Sequence[float]) -> np.ndarray:
        return self.coriolis_torque(q, dq) + self.gravity(q)

    def inverse_dynamics(self, q: Sequence[float], dq: Sequence[float], ddq: Sequence[float],
                         friction: bool = False) -> np.ndarray:
        tau = self.mass_matrix(q) @ np.asarray(ddq, dtype=float).reshape(2) + self.bias(q, dq)
        if friction:
            tau = tau + self.friction_torque(dq)
        return tau

    def forward_dynamics(self, q: Sequence[float], dq: Sequence[float],
                         tau: Sequence[float], friction: bool = False) -> np.ndarray:
        rhs = np.asarray(tau, dtype=float).reshape(2) - self.bias(q, dq)
        if friction:
            rhs = rhs - self.friction_torque(dq)
        return np.linalg.solve(self.mass_matrix(q), rhs)

    # ---------- 能量 ----------
    def kinetic_energy(self, q: Sequence[float], dq: Sequence[float]) -> float:
        dq = np.asarray(dq, dtype=float).reshape(2)
        return float(0.5 * dq @ self.mass_matrix(q) @ dq)

    def potential_energy(self, q: Sequence[float]) -> float:
        q1, q2 = float(q[0]), float(q[1])
        r1 = np.array([self.lc1 * math.cos(q1), self.lc1 * math.sin(q1)])
        r2 = np.array([self.L1 * math.cos(q1) + self.lc2 * math.cos(q1 + q2),
                       self.L1 * math.sin(q1) + self.lc2 * math.sin(q1 + q2)])
        return -(self.m1 * float(self._g @ r1) + self.m2 * float(self._g @ r2))

    def energy(self, q: Sequence[float], dq: Sequence[float]) -> Dict[str, float]:
        T = self.kinetic_energy(q, dq)
        V = self.potential_energy(q)
        return {"kinetic": T, "potential": V, "total": T + V}


# ---------------------------------------------------------------------------
# 模型工厂与积分器（与模型类型无关）
# ---------------------------------------------------------------------------

def make_model(spec: ArmSpec) -> ChainModel | ClosedForm2R:
    """构型匹配 2 轴平面臂时返回解析模型，否则返回通用模型。"""
    try:
        return ClosedForm2R.from_spec(spec)
    except ValueError:
        return ChainModel(spec)


def simulate(model, q0: Sequence[float], dq0: Sequence[float],
             tau_fn: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
             T: float, dt: float, friction: bool = False,
             method: str = "rk4") -> Dict[str, np.ndarray]:
    """通用积分器：``tau_fn(t, q, dq) -> tau``。

    返回 {'t','q','dq','tau'}，q/dq 形状 (N, n)。method: 'rk4' | 'euler'。
    """
    n = model.n
    N = int(round(T / dt)) + 1
    t = np.linspace(0.0, dt * (N - 1), N)
    q = np.zeros((N, n))
    dq = np.zeros((N, n))
    tau_hist = np.zeros((N, n))
    q[0] = np.asarray(q0, dtype=float).reshape(n)
    dq[0] = np.asarray(dq0, dtype=float).reshape(n)

    def f(qq, dqq, tt):
        tau = np.asarray(tau_fn(tt, qq, dqq), dtype=float).reshape(n)
        ddq = model.forward_dynamics(qq, dqq, tau, friction=friction)
        return dqq, ddq, tau

    for i in range(N - 1):
        ti = t[i]
        qi, dqi = q[i], dq[i]
        if method == "euler":
            dqi_, ddqi, tau_i = f(qi, dqi, ti)
            q[i + 1] = qi + dt * dqi_
            dq[i + 1] = dqi + dt * ddqi
            tau_hist[i] = tau_i
        else:
            k1v, k1a, tau_i = f(qi, dqi, ti)
            k2v, k2a, _ = f(qi + 0.5 * dt * k1v, dqi + 0.5 * dt * k1a, ti + 0.5 * dt)
            k3v, k3a, _ = f(qi + 0.5 * dt * k2v, dqi + 0.5 * dt * k2a, ti + 0.5 * dt)
            k4v, k4a, _ = f(qi + dt * k3v, dqi + dt * k3a, ti + dt)
            q[i + 1] = qi + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
            dq[i + 1] = dqi + dt / 6.0 * (k1a + 2 * k2a + 2 * k3a + k4a)
            tau_hist[i] = tau_i
    tau_hist[-1] = np.asarray(tau_fn(t[-1], q[-1], dq[-1]), dtype=float).reshape(n)
    return {"t": t, "q": q, "dq": dq, "tau": tau_hist}
