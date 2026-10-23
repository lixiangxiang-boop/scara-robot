"""SCARA 机械臂控制器（任务级 API）——"用一个控制器操作机器人动作与抓取"。

对外接口（都会在内部以 1 kHz CTC 稳定推进物理，并周期性回调同步可视化）：

    ctrl = ScaraController(model, data, spec, sync=viewer.sync)
    ctrl.home()                     # 回到初始位姿（夹爪张开）
    ctrl.move_to(x, y, z, yaw=0)    # 末端到笛卡尔点（闭式 IK + 五次多项式 + CTC）
    ctrl.open_gripper() / close_gripper()
    ctrl.pick(x, y)                 # 完整抓取：接近→下探→闭合→抬起
    ctrl.place(x, y)                # 完整放置：搬运→下探→张开→缩回
    ctrl.set_place_hint(px, py)     # 指定放置区（供 place() 规划）

设计要点
--------
- 控制律复用手臂数学模型（`ChainModel`：与 MuJoCo/MATLAB 对拍 1e−15）；
  MuJoCo 只作为物理被控对象；夹爪用位置伺服（`position` 执行器）。
- 所有方法都是**阻塞式**的，但通过 `sync` 回调持续刷新 MuJoCo 窗口，
  因此在交互控制台里也是一条条命令往下执行。
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from . import dynamics as D
from . import scara_scene as SC
from . import trajectory as T
from .kinematics import ChainKinematics
from .spec import ArmSpec


class ScaraController:
    def __init__(self, model, data, spec: ArmSpec, dt_ctrl: float = 1e-3,
                 sync: Optional[Callable[[], None]] = None,
                 kp: Sequence[float] = (60.0, 60.0, 400.0, 20.0),
                 kd: Sequence[float] = (12.0, 12.0, 40.0, 6.0),
                 tau_max: Sequence[float] = (40.0, 40.0, 80.0, 5.0),
                 sync_every: int = 8, dyn=None):
        import mujoco
        self.mujoco = mujoco
        self.model, self.data, self.spec = model, data, spec
        # 动力学模型可注入（学习阶段用闭式快速模型 ScaraFastModel，快 ~70 倍）
        self.dyn = dyn if dyn is not None else D.ChainModel(spec)
        self.kin = ChainKinematics(spec)          # 只负责正/逆运动学
        self.n = 4
        self.dt = dt_ctrl
        self.sub = max(int(round(dt_ctrl / float(model.opt.timestep))), 1)
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.tau_max = np.asarray(tau_max, dtype=float)
        self.sync = sync
        self.sync_every = sync_every
        self.gripper = SC.GRIP_OPEN      # 当前张开量
        self.place_hint = (0.30, 0.12)
        self.record = False              # 置 True 时记录轨迹（供出图/数据分析）
        self.history: dict = {"t": [], "q": [], "tau": [], "grip": []}
        self._q_star_hold = None         # servo_track 的上一次有效关节目标（防死锁）
        self._t = 0.0

    # ---------------- 基础操作 ----------------
    @property
    def q(self) -> np.ndarray:
        sl = SC.arm_joint_addr(self.model)
        return self.data.qpos[sl].copy()

    @property
    def dq(self) -> np.ndarray:
        sl = SC.arm_joint_addr(self.model)
        return self.data.qvel[sl].copy()

    def _sync(self) -> None:
        if self.sync is not None:
            self.sync()

    def _step(self, tau: np.ndarray, grip: Optional[float] = None) -> None:
        self.data.ctrl[: self.n] = tau
        if grip is not None:
            self.gripper = float(grip)
            SC.gripper_ctrl(self.model, self.data, self.gripper)
        if self.record:
            self.history["t"].append(self._t)
            self.history["q"].append(self.q.tolist())
            self.history["tau"].append(tau.tolist())
            self.history["grip"].append(self.gripper)
        for _ in range(self.sub):
            self.mujoco.mj_step(self.model, self.data)
        self._t += self.dt

    def _track(self, traj: T.Trajectory, grip_from: float, grip_to: float) -> None:
        """跟踪一段关节轨迹（含夹爪线性过渡），周期性同步可视化。"""
        N = len(traj.t)
        for k in range(N - 1):
            q = self.q
            q_d, dq_d, ddq_d = traj.sample(float(traj.t[k]))
            ddq_cmd = ddq_d + self.kd * (dq_d - self.dq) + self.kp * (q_d - q)
            tau = self.dyn.inverse_dynamics(q, self.dq, ddq_cmd, friction=True)
            tau = np.clip(tau, -self.tau_max, self.tau_max)
            frac = k / max(N - 1, 1)
            grip = grip_from + (grip_to - grip_from) * frac
            self._step(tau, grip)
            if k % self.sync_every == 0:
                self._sync()
        self._sync()

    def hold(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.dt))):
            q = self.q
            tau = self.dyn.gravity(q) + self.kd * (0.0 - self.dq)
            self._step(np.clip(tau, -self.tau_max, self.tau_max), self.gripper)
            self._sync()

    # ---------------- 任务级 API ----------------
    def open_gripper(self, duration: float = 0.3) -> None:
        traj = T.joint_quintic(self.q, self.q, T=duration, dt=self.dt)
        self._track(traj, self.gripper, SC.GRIP_OPEN)

    def close_gripper(self, duration: float = 0.35) -> None:
        traj = T.joint_quintic(self.q, self.q, T=duration, dt=self.dt)
        self._track(traj, self.gripper, 0.0)

    def move_to(self, x: float, y: float, z: float, yaw: float = 0.0,
                duration: Optional[float] = None) -> bool:
        """末端到笛卡尔点（保持 q4 使工具朝向 yaw；自动选最近的肘部解）。"""
        q_goal = self._ik((x, y, z), yaw)
        if q_goal is None:
            print(f"[controller] 目标不可达：({x:.3f}, {y:.3f}, {z:.3f})")
            return False
        if duration is None:
            dist = float(np.max(np.abs(q_goal - self.q)))
            duration = float(np.clip(0.5 + 1.6 * dist, 0.5, 2.5))
        traj = T.joint_quintic(self.q, q_goal, T=duration, dt=self.dt)
        self._track(traj, self.gripper, self.gripper)
        return True

    def _ik(self, xyz: Sequence[float], yaw: float) -> Optional[np.ndarray]:
        cands = [q for q in (SC.scara_ik(xyz[0], xyz[1], xyz[2], yaw, self.spec, elbow=e)
                             for e in (-1, 1)) if q is not None]
        if not cands:
            return None
        q0 = self.q

        def dist(q):
            return float(np.max(np.abs([np.arctan2(np.sin(q[i] - q0[i]),
                                                   np.cos(q[i] - q0[i])) for i in range(4)])))
        return min(cands, key=dist)

    def pick(self, x: float, y: float, yaw: float = 0.0, hover: float = 0.26,
             grasp: float = 0.18, lift: float = 0.21) -> bool:
        """抓取：张开→高位接近→下探→闭合→抬起。"""
        self.open_gripper()
        if not self.move_to(x, y, hover, yaw):
            return False
        if not self.move_to(x, y, grasp, yaw, duration=0.9):
            return False
        self.close_gripper()
        self.hold(0.1)
        return self.move_to(x, y, lift, yaw, duration=0.6)

    def place(self, x: Optional[float] = None, y: Optional[float] = None,
              yaw: float = 0.0, travel: float = 0.21, release: float = 0.18,
              retract: float = 0.26) -> bool:
        """放置：搬运→下探→张开→全部缩回。"""
        px, py = (self.place_hint if x is None or y is None else (x, y))
        if not self.move_to(px, py, travel, yaw, duration=1.2):
            return False
        if not self.move_to(px, py, release, yaw, duration=0.8):
            return False
        self.open_gripper()
        self.hold(0.1)
        return self.move_to(px, py, retract, yaw, duration=0.7)

    def home(self, duration: float = 1.5) -> bool:
        """回到零位（全缩回、夹爪张开）。"""
        self.open_gripper()
        traj = T.joint_quintic(self.q, np.zeros(4), T=duration, dt=self.dt)
        self._track(traj, self.gripper, SC.GRIP_OPEN)
        return True

    def set_place_hint(self, x: float, y: float) -> None:
        self.place_hint = (float(x), float(y))

    def servo_to(self, x: float, y: float, z: float, yaw: Optional[float] = None,
                 grip: Optional[float] = None, duration: float = 0.02) -> bool:
        """**策略执行接口**：把末端伺服到笛卡尔目标（一个策略步内完成）。

        - 与 `move_to` 的区别：`move_to` 是任务级的大位移规划（0.5~2 s，五次多项式）；
          `servo_to` 只做"小步跟随"（默认 20 ms），供学习到的策略以 50 Hz 输出目标点；
        - yaw=None 时保持当前工具朝向；grip 给定时同时设置夹爪张开量；
        - 目标先**投影到可达工作空间**（径向环带 + z 行程），避免"目标越界 1 mm →
          IK 返回 None → 整拍不动 → 观测不变 → 策略死锁"这一整类问题。
        """
        x, y, z = self._project_ws(float(x), float(y), float(z))
        yaw_now = float(self.q[0] + self.q[1] + self.q[3])      # SCARA：末端朝向 = q1+q2+q4
        q_goal = self._ik((x, y, z), yaw_now if yaw is None else float(yaw))
        if q_goal is None:
            return False
        traj = T.joint_quintic(self.q, q_goal, T=duration, dt=self.dt)
        self._track(traj, self.gripper if grip is None else grip,
                    self.gripper if grip is None else grip)
        return True

    # ---------------- 策略伺服（带前瞻的关节空间跟踪） ----------------
    def _project_ws(self, x: float, y: float, z: float):
        """把笛卡尔目标投影回可达工作空间：径向 [r_min, 0.999·(a1+a2)]、z 在 J3 行程内。"""
        a1 = float(self.spec.links[2].pos[0])
        a2 = float(self.spec.links[3].pos[0])
        d1 = float(self.spec.links[1].pos[2])
        stroke = float(self.spec.joints[2].upper)
        r = float(np.hypot(x, y))
        r_max = (a1 + a2) * 0.999
        if r > r_max:
            s = r_max / r
            x, y = x * s, y * s
        elif r < 1e-3:
            x, y = (1e-3, 0.0) if r < 1e-9 else (x * 1e-3 / r, y * 1e-3 / r)
        z = float(np.clip(z, d1 - stroke + 1e-4, d1 - 1e-4))
        return x, y, z

    def servo_track(self, x: float, y: float, z: float, grip: Optional[float] = None,
                    horizon: float = 0.3, dt_exec: float = 0.02,
                    dq_max: Sequence[float] = (2.5, 2.5, 0.6, 2.5),
                    v_max: float = 0.30) -> bool:
        """**策略执行接口（推荐）**：把末端沿**笛卡尔直线**伺服到目标点（MoveL 式指数收敛）。

        1) 每 1 ms 计算直线上的一小步：Δp = (p_target − p_now)·(dt/horizon)，按 v_max 限速
        （时间常数 = horizon ⇒ 距离目标越近越慢，天然起停，不需要规划器）
        2) 小步 IK 得 q_des（先投影回工作空间；IK 失败则沿用上一次目标，防死锁）
        3) 关节速度前馈 dq_ff = clip((q_des − q)/dt, ±dq_max)
        4) ddq_cmd = kp·dq_ff·dt + kd(dq_ff − dq)，τ = ID(q, dq, ddq_cmd)，逐 1 ms 执行 dt_exec

        为什么是"直线步进 + 速度前馈"：
        - `servo_to`（每拍从零速重启的五次多项式）跟踪误差均值 13 mm / 最大 86 mm；
        - 关节空间指数收敛虽然误差小（均值 2.1 mm），但末端路径会弓出（关节插值），
          大位移起步时关节速度可超 3 rad/s，把策略带出训练分布（实测闭环震荡）；
        - 本实现路径与专家 MoveL 一致（都是直线），速度被限制在专家量级内。
        """
        x, y, z = self._project_ws(float(x), float(y), float(z))
        p_tgt = np.array([x, y, z], dtype=float)
        yaw_now = float(self.q[0] + self.q[1] + self.q[3])
        dqm = np.asarray(dq_max, dtype=float)
        n_sub = max(int(round(dt_exec / self.dt)), 1)
        for _ in range(n_sub):
            dt = self.dt
            p_now = np.asarray(self.kin.fk(self.q, link="link4")[1], dtype=float)
            step = (p_tgt - p_now) * (dt / horizon)
            n_step = float(np.linalg.norm(step))
            if n_step > v_max * dt:               # 限速：大位移段以 v_max 匀速靠近
                step *= (v_max * dt) / n_step
            p_cmd = self._project_ws(*(p_now + step))
            q_des = self._ik(p_cmd, yaw_now)
            if q_des is None:
                q_des = self.q if self._q_star_hold is None else self._q_star_hold
            else:
                self._q_star_hold = q_des
            dq_ff = np.clip((q_des - self.q) / dt, -dqm, dqm)
            ddq_cmd = self.kp * (dq_ff * dt) + self.kd * (dq_ff - self.dq)
            tau = self.dyn.inverse_dynamics(self.q, self.dq, ddq_cmd, friction=True)
            tau = np.clip(tau, -self.tau_max, self.tau_max)
            self._step(tau, self.gripper if grip is None else grip)
        self._sync()
        return True

    # ---------------- 便捷查询 ----------------
    def tcp(self) -> np.ndarray:
        """末端（工具原点）笛卡尔位置。"""
        _, p = self.kin.fk(self.q, link="link4")
        return p

    def object_position(self) -> np.ndarray:
        return self.data.xpos[SC.ids(self.model, "body", "object")].copy()
