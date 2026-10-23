"""SCARA 抓取-放置任务控制器（与具体仿真器解耦，可在 viewer / 批量采样中复用）。

设计
----
- **纯函数式接口**：`PickPlaceTask.step(q, dq)` 输入状态、输出力矩与事件，
  不直接操作 MuJoCo —— 这样同一份控制器既能驱动 viewer 演示，也能批量产数据。
- 控制器用**数学侧**动力学（`ChainModel`：已与 MuJoCo/MATLAB 对拍 1e−15），
  仿真器只当"物理被控对象"。
- 八段状态机：approach → descend → 夹紧 → lift → move → lower → 释放 → retreat。
- 每段的笛卡尔运动用 **MoveL（笛卡尔直线）** 生成：关节空间插值的末端轨迹会向一侧
  弓出（实测 10~32 mm），既不利于跟踪、也让模仿学习的数据流形变复杂。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import dynamics as D
from . import scara_scene as SC
from . import trajectory as T
from .kinematics import ChainKinematics
from .spec import ArmSpec

TABLE_TOP = 0.10
OBJ_SIZE = 0.03
# 两指夹爪：手指相对工具原点向下 20~60 mm，闭合时间距 28 mm（30 mm 物体 2 mm 过盈）
GRASP_Z = 0.180        # 手指跨在物体两侧（指尖离台面 20 mm，留足下探余量）
LIFT_Z = 0.210         # 抓到后"缩回去一些"（物体底离台面 3 cm）
HOVER_Z = 0.260        # 全缩回 / 水平移动高度（指尖 0.185 → 不碰放置好的物体 0.13）
GRIP_OPEN = 0.012      # 手指张开行程（与 scara_scene.GRIP_OPEN 一致）

# J3 是移动副：增益量纲与转动关节不同（力矩 = 力）
KP_DEFAULT = np.array([60.0, 60.0, 400.0, 20.0])
KD_DEFAULT = np.array([12.0, 12.0, 40.0, 6.0])


@dataclass
class Phase:
    name: str
    kind: str                 # "move"（到笛卡尔点）| "hold"（原地等待）
    z: float = 0.0            # 目标高度（move 用）
    grip_from: float = GRIP_OPEN   # 段内夹爪张开量（线性过渡到 grip_to）
    grip_to: float = GRIP_OPEN
    duration: float = 1.0
    event: Optional[str] = None   # 段结束触发的事件：grasp / release


PHASES: List[Phase] = [
    Phase("approach", "move", HOVER_Z, duration=1.0),                       # 高位接近（张开）
    Phase("descend", "move", GRASP_Z, duration=0.9),                        # 伸出抓手到物体两侧
    Phase("grasp", "hold", GRIP_OPEN, 0.0, duration=0.35, event="grasp"),   # 闭合夹紧
    Phase("lift", "move", LIFT_Z, 0.0, 0.0, 0.6),                           # 缩回去一些
    Phase("move", "move", LIFT_Z, 0.0, 0.0, 1.2),                           # 搬运
    Phase("lower", "move", GRASP_Z, 0.0, 0.0, 0.8),                         # 放置时再往下走一些
    Phase("release", "hold", 0.0, GRIP_OPEN, duration=0.35, event="release"),  # 张开释放
    Phase("retract", "move", HOVER_Z, duration=0.7),                        # 全部缩回
]


def phase_waypoints(obj_xy, place_xy):
    """8 个相位的目标点（与 PHASES 顺序一致）。

    注意 hold 段（grasp/release）的目标 = 上一段的终点（原地驻留），
    所以这里用显式表给出，避免 PHASES 里占位用的 z 字段造成歧义。
    """
    return [(obj_xy[0], obj_xy[1], HOVER_Z),      # 0 approach 高位接近
            (obj_xy[0], obj_xy[1], GRASP_Z),      # 1 descend  下探到物体两侧
            (obj_xy[0], obj_xy[1], GRASP_Z),      # 2 grasp    原地夹紧
            (obj_xy[0], obj_xy[1], LIFT_Z),       # 3 lift     原地抬起
            (place_xy[0], place_xy[1], LIFT_Z),   # 4 move     水平搬运
            (place_xy[0], place_xy[1], GRASP_Z),  # 5 lower    下放
            (place_xy[0], place_xy[1], GRASP_Z),  # 6 release  原地张开
            (place_xy[0], place_xy[1], HOVER_Z)]  # 7 retract  全部缩回


class PhaseSequencer:
    """状态驱动的相位状态机（= 脚本专家 / 标签生成器 / DAgger 专家）。

    规则：到达当前相位的目标点（容差内）且在该相位驻留足够时间后，推进到下一相位。
    为什么必须"状态驱动"而不是"时间驱动"：时间驱动的标签在相位边界会产生**自相矛盾**的
    样本（同一个状态既被标为上一相位、又被标为下一相位），策略只能取平均/摇摆，
    闭环会在边界处死锁（实测：到达抓取点后一直判 "descend"，9 s 不动）。
    执行器（run_rollout）用与之同构的规则，因此标签和学习目标天然一致。
    """

    TOL = (0.020, 0.010, 0.010, 0.012, 0.014, 0.010, 0.010, 1.0)   # 到达容差 [m]
    DWELL = (0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.35, 1e9)             # 驻留时间 [s]

    def __init__(self, obj_xy, place_xy):
        self.obj_xy = (float(obj_xy[0]), float(obj_xy[1]))
        self.place_xy = (float(place_xy[0]), float(place_xy[1]))
        self.waypoints = phase_waypoints(self.obj_xy, self.place_xy)
        self.durations = [ph.duration for ph in PHASES]
        self.phase = 0
        self.t_state = 0.0

    def update(self, p_tcp, dt: float) -> int:
        """输入当前末端位置与步长，返回（可能已推进的）相位。"""
        self.t_state += float(dt)
        wp = self.waypoints[self.phase]
        near = float(np.linalg.norm(np.asarray(p_tcp, dtype=float).reshape(3)
                                    - np.asarray(wp, dtype=float))) < self.TOL[self.phase]
        if near and self.t_state >= self.DWELL[self.phase] and self.phase < len(PHASES) - 1:
            self.phase += 1
            self.t_state = 0.0
        return self.phase

    @property
    def progress(self) -> float:
        """当前相位内的相对进度 [0, 1]（驻留相位用其驻留时间归一）。"""
        T = max(self.durations[self.phase], 1e-6)
        return float(min(self.t_state / T, 1.0))


class PickPlaceTask:
    """一次抓取-放置的控制器（可反复 reset 复用）。"""

    def __init__(self, spec: ArmSpec, obj_xy: Sequence[float], place_xy: Sequence[float],
                 psi: float = 0.0, dt_ctrl: float = 1e-3,
                 kp: Sequence[float] = KP_DEFAULT, kd: Sequence[float] = KD_DEFAULT,
                 tau_max: Sequence[float] = (40.0, 40.0, 80.0, 5.0), model=None,
                 friction: bool = True):
        self.spec = spec
        # 动力学模型可注入：批量产数据时用闭式快速模型（ScaraFastModel，快 ~70 倍）
        self.model = model if model is not None else D.ChainModel(spec)
        self.kin = ChainKinematics(spec)          # MoveL 需要末端位置与逐点 IK
        self.friction = bool(friction)
        self.dt = dt_ctrl
        self.kp = np.asarray(kp, dtype=float)
        self.kd = np.asarray(kd, dtype=float)
        self.tau_max = np.asarray(tau_max, dtype=float)
        self.psi = psi
        self.set_targets(obj_xy, place_xy)

    # ---------- 目标与重置 ----------
    def set_targets(self, obj_xy: Sequence[float], place_xy: Sequence[float]) -> None:
        self.obj_xy = (float(obj_xy[0]), float(obj_xy[1]))
        self.place_xy = (float(place_xy[0]), float(place_xy[1]))

    def reset(self, q_start: np.ndarray) -> None:
        """按当前关节角重新规划整条轨迹。"""
        self.traj = self._plan(np.asarray(q_start, dtype=float).reshape(4))
        self.i = 0                  # 轨迹采样下标
        self.phase = 0
        self.events: List[str] = []
        self.done = False

    def _plan(self, q0: np.ndarray) -> T.Trajectory:
        """把八段状态机展开成一条关节轨迹 + 每点所属段号。"""
        segs, q = [], q0.copy()
        self.seg_of_sample: List[int] = []
        for idx, ph in enumerate(PHASES):
            if ph.kind == "hold":
                seg = T.joint_quintic(q, q, T=ph.duration, dt=self.dt)
            else:
                # 目标区归属：approach/descend/lift 在物体上方；move/lower/retract 在放置区
                # （lift 必须写进物体一侧，否则"抬起"会直接斜着飞到放置区，"搬运"段变空转）
                base = (self.obj_xy if ph.name in ("approach", "descend", "lift")
                        else self.place_xy)
                tgt = (base[0], base[1], ph.z)
                try:
                    seg = T.cartesian_line_ndof(self.kin, self._ik_ref, tgt,
                                                T=ph.duration, dt=self.dt, q_start=q)
                except ValueError as exc:
                    raise ValueError(f"目标不可达：{tgt}（{exc}）")
            q = seg.q[-1].copy()
            segs.append(seg)
            self.seg_of_sample += [idx] * len(seg.t)
        # 拼接时间轴必须**累积**各段时长（曾经写错成"只加前一段"，段长不等时参考轨迹会错位）
        t_parts, t0 = [], 0.0
        self._seg_t0 = []
        for seg in segs:
            self._seg_t0.append(t0)
            t_parts.append(seg.t + t0)
            t0 = float(t_parts[-1][-1]) + self.dt
        traj = T.Trajectory(
            t=np.concatenate(t_parts),
            q=np.vstack([s.q for s in segs]), dq=np.vstack([s.dq for s in segs]),
            ddq=np.vstack([s.ddq for s in segs]))
        return traj

    def _ik(self, xyz: Sequence[float], q_cur: np.ndarray) -> Optional[np.ndarray]:
        """闭式 IK + 分支连续性（选与当前角最接近的解）。"""
        cands = [q for q in (SC.scara_ik(xyz[0], xyz[1], xyz[2], self.psi, self.spec, elbow=e)
                             for e in (-1, 1)) if q is not None]
        if not cands:
            return None
        def dist(q):
            d = np.array([np.arctan2(np.sin(q[i] - q_cur[i]), np.cos(q[i] - q_cur[i]))
                          for i in range(4)])
            return float(np.max(np.abs(d)))
        return min(cands, key=dist)

    def _ik_ref(self, xyz: Sequence[float], q_ref: np.ndarray) -> Optional[np.ndarray]:
        """MoveL 逐点 IK：与 q_ref 同分支（最接近解），并把越界目标投回工作空间。"""
        a1 = float(self.spec.links[2].pos[0])
        a2 = float(self.spec.links[3].pos[0])
        d1 = float(self.spec.links[1].pos[2])
        stroke = float(self.spec.joints[2].upper)
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        r = float(np.hypot(x, y))
        r_max = (a1 + a2) * 0.9995
        if r > r_max:
            s = r_max / r
            x, y = x * s, y * s
        z = float(np.clip(z, d1 - stroke + 1e-6, d1 - 1e-6))
        return self._ik((x, y, z), np.asarray(q_ref, dtype=float))

    # ---------- 逐拍调用 ----------
    def step(self, q: np.ndarray, dq: np.ndarray
             ) -> Tuple[np.ndarray, float, Optional[str]]:
        """返回 (关节力矩, 夹爪张开量, 事件)；事件为 'grasp'/'release' 或 None。"""
        idx = min(self.i, len(self.traj.t) - 1)
        q_d, dq_d, ddq_d = self.traj.sample(float(self.traj.t[idx]))
        e = q_d - q
        de = dq_d - dq
        ddq_cmd = ddq_d + self.kd * de + self.kp * e
        tau = self.model.inverse_dynamics(q, dq, ddq_cmd, friction=self.friction)
        tau = np.clip(tau, -self.tau_max, self.tau_max)

        # 夹爪：在所在阶段内从 grip_from 线性过渡到 grip_to
        seg = self.seg_of_sample[idx]
        ph = PHASES[seg]
        t0 = self.traj.t[idx] - self._seg_t0[seg]
        frac = 0.0 if ph.duration <= 0 else min(max(t0 / ph.duration, 0.0), 1.0)
        grip = ph.grip_from + (ph.grip_to - ph.grip_from) * frac

        event = None
        nxt = self.i + 1
        if nxt >= len(self.traj.t):
            self.done = True
        elif self.seg_of_sample[nxt] != self.seg_of_sample[idx]:
            event = PHASES[self.seg_of_sample[nxt] - 1].event   # 下一段起点处的触发条件
        self.i = nxt
        return tau, grip, event

    @property
    def phase_name(self) -> str:
        idx = min(self.i, len(self.seg_of_sample) - 1)
        return PHASES[self.seg_of_sample[idx]].name

    # ---------- 运动原语（供学习到的"相位策略"执行） ----------
    def phase_target(self, idx: int) -> Optional[Tuple[float, float, float]]:
        """第 idx 段的目标点（hold 段 = 原地驻留点，见 phase_waypoints）。"""
        return phase_waypoints(self.obj_xy, self.place_xy)[int(idx)]

    def phase_grip(self, idx: int) -> float:
        """第 idx 段的夹爪指令（该相位终点的夹爪值）。"""
        return float(PHASES[int(idx)].grip_to)

    def phase_segment(self, idx: int, q_now: np.ndarray) -> T.Trajectory:
        """按**当前状态**重新锚定的第 idx 段运动原语（策略闭环执行用）。

        所有相位统一用 MoveL 五次多项式（时长 = 该段标称时长）：hold 段的目标点
        就是上一段的终点 ⇒ 已经到位时表现为原地保持，未完全到位时会先收敛过去，
        因此"策略提前判到 grasp"也能把手臂拉到抓取点再闭合手指。
        策略只需要判断"现在处于哪一段"，运动形状与专家完全一致（跟踪精度回到专家量级）。
        """
        ph = PHASES[int(idx)]
        q_now = np.asarray(q_now, dtype=float).reshape(4)
        tgt = self.phase_target(idx)
        return T.cartesian_line_ndof(self.kin, self._ik_ref, tgt, T=ph.duration,
                                     dt=self.dt, q_start=q_now, n_via=24)
