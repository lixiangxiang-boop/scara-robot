"""机械臂参数定义：通用 N 自由度串链（URDF/MuJoCo 风格约定）。

约定（与 MuJoCo/URDF 完全一致，便于跨工具复用）
------------------------------------------------
- 连杆坐标系原点位于该连杆的**近端关节**处，x 轴指向远端；
- 关节变换 ``T = Trans(pos)·Quat(quat)·Rot(axis, q)``；
- 质量属性：``mass`` + ``com``（连杆系）+ ``inertia``（绕质心，连杆系，3x3）；
- 单位一律 SI：m / kg / rad / N·m；
- 重力是三维矢量，默认 (0, -9.81, 0)：运动平面为竖直平面、重力在面内。
  置零即退化为"水平面工况"（SCARA 式），两种工况共用同一套代码。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (w, x, y, z)
Mat3 = Tuple[Vec3, Vec3, Vec3]

DEG = math.pi / 180.0

# 参数基线（每次确认/修改后更新；会写入 arm_params.json，便于全链路追溯）
BASELINE = {
    "version": "v0.2",
    "confirmed_on": "2026-10-16",
    "note": "2 轴平面臂基线：L1=300 mm、L2=250 mm；竖直面内运动（重力沿 -y）；"
            "质量/惯量为设计估计值，P2（SolidWorks）实测后回填",
    "tool_interface": "Ø30 法兰 + 4×M4 均布孔；工具建模为末端点质量（默认 0.5 kg 选型负载）",
    "extensions": "第 3 轴（底座回转）预留：新增一个带 joint 的连杆即可，控制/仿真代码不变",
}


def box_inertia(m: float, L: float, w: float, t: float) -> Mat3:
    """长方体（x 方向长 L、y 方向宽 w、z 方向厚 t）绕质心的惯量矩阵 [kg·m²]。"""
    ixx = m * (w * w + t * t) / 12.0
    iyy = m * (L * L + t * t) / 12.0
    izz = m * (L * L + w * w) / 12.0
    return ((ixx, 0.0, 0.0), (0.0, iyy, 0.0), (0.0, 0.0, izz))


def _as_tuple3(v: Sequence[float]) -> Vec3:
    return (float(v[0]), float(v[1]), float(v[2]))


@dataclass(frozen=True)
class JointSpec:
    """关节定义（revolute 转轴 / prismatic 移动，后者为多轴扩展预留）。"""

    name: str
    type: str = "revolute"                 # "revolute" | "prismatic"
    axis: Vec3 = (0.0, 0.0, 1.0)           # 关节轴（连杆系，零位）
    lower: float = -math.pi
    upper: float = math.pi
    damping: float = 0.0                   # 粘性摩擦 [N·m·s/rad]
    frictionloss: float = 0.0              # 库仑摩擦 [N·m]
    armature: float = 0.0                  # 电机/减速器折算转子惯量 [kg·m²]
    gear: float = 1.0                      # 传动比（力矩/运动放大，默认直驱）

    def limit_violation(self, q: float) -> float:
        """返回超限程度（<=0 表示在限位内）。"""
        return max(self.lower - q, q - self.upper, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["axis"] = list(self.axis)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "JointSpec":
        d = dict(d)
        d["axis"] = _as_tuple3(d.get("axis", (0, 0, 1)))
        return JointSpec(**d)


@dataclass(frozen=True)
class LinkSpec:
    """连杆定义。``joint=None`` 表示固定连杆（基座/工具/负载）。"""

    name: str
    pos: Vec3 = (0.0, 0.0, 0.0)            # 相对父连杆系的位置（零位）
    quat: Quat = (1.0, 0.0, 0.0, 0.0)      # 相对父连杆系的姿态（零位，wxyz）
    mass: float = 0.0
    com: Vec3 = (0.0, 0.0, 0.0)            # 质心（连杆系）
    inertia: Mat3 = ((0.0, 0, 0), (0, 0, 0), (0, 0, 0))  # 绕质心
    joint: Optional[JointSpec] = None
    geom: Optional[Dict[str, Any]] = None  # 可视化几何（供 MJCF/绘图）

    @property
    def is_moving(self) -> bool:
        return self.joint is not None

    def inertia_matrix(self) -> np.ndarray:
        return np.asarray(self.inertia, dtype=float).reshape(3, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pos": list(self.pos),
            "quat": list(self.quat),
            "mass": self.mass,
            "com": list(self.com),
            "inertia": [list(r) for r in self.inertia],
            "joint": self.joint.to_dict() if self.joint else None,
            "geom": self.geom,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LinkSpec":
        return LinkSpec(
            name=d["name"],
            pos=_as_tuple3(d.get("pos", (0, 0, 0))),
            quat=tuple(float(x) for x in d.get("quat", (1, 0, 0, 0))),  # type: ignore[arg-type]
            mass=float(d.get("mass", 0.0)),
            com=_as_tuple3(d.get("com", (0, 0, 0))),
            inertia=tuple(tuple(float(x) for x in row) for row in d.get("inertia", np.zeros((3, 3))))  # type: ignore[arg-type]
            if "inertia" in d
            else ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            joint=JointSpec.from_dict(d["joint"]) if d.get("joint") else None,
            geom=d.get("geom"),
        )


@dataclass(frozen=True)
class ArmSpec:
    """整臂定义：一串连杆（含关节）+ 可选末端工具/负载。"""

    name: str
    links: Tuple[LinkSpec, ...]
    gravity: Vec3 = (0.0, -9.81, 0.0)
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---------- 派生属性 ----------
    @property
    def dof(self) -> int:
        return sum(1 for link in self.links if link.is_moving)

    @property
    def moving_links(self) -> List[LinkSpec]:
        return [link for link in self.links if link.is_moving]

    @property
    def joint_names(self) -> List[str]:
        return [link.joint.name for link in self.moving_links]  # type: ignore[union-attr]

    @property
    def joints(self) -> List[JointSpec]:
        return [link.joint for link in self.moving_links]  # type: ignore[list-item]

    def joint_index(self, name: str) -> int:
        return self.joint_names.index(name)

    def limits_array(self) -> np.ndarray:
        return np.array([[j.lower, j.upper] for j in self.joints], dtype=float)

    def replace_link(self, name: str, **changes: Any) -> "ArmSpec":
        """返回替换了某个连杆属性的新 spec（不可变数据结构）。"""
        links = [replace(link, **changes) if link.name == name else link for link in self.links]
        return replace(self, links=tuple(links))

    def with_tool(self, mass: float, pos: Optional[Vec3] = None,
                  name: str = "tool", inertia: Optional[Mat3] = None) -> "ArmSpec":
        """返回附加末端负载（点质量）的新 spec。pos 默认位于最后一个连杆远端。"""
        last = self.links[-1]
        if pos is None:
            pos = _link_tip_position(last)
        geom = {"type": "sphere", "size": [0.02], "rgba": [0.9, 0.4, 0.2, 1.0]}
        if inertia is None:      # 默认按匀质小球估算（物理合理、避免零惯量）
            r = 0.02
            I = 0.4 * mass * r * r
            inertia = ((I, 0.0, 0.0), (0.0, I, 0.0), (0.0, 0.0, I))
        tool = LinkSpec(
            name=name, pos=pos, quat=(1.0, 0.0, 0.0, 0.0), mass=float(mass),
            com=(0.0, 0.0, 0.0),
            inertia=inertia,
            joint=None, geom=geom,
        )
        links = [link for link in self.links if link.name != name]
        return replace(self, links=tuple(links + [tool]))

    def widen_limits(self, extra: float = 10.0) -> "ArmSpec":
        """返回把全部关节限位放宽 extra 弧度后的新 spec（无约束对比/仿真用）。"""
        links = []
        for link in self.links:
            if link.is_moving:
                joint = link.joint
                assert joint is not None
                links.append(replace(link, joint=replace(
                    joint, lower=joint.lower - extra, upper=joint.upper + extra)))
            else:
                links.append(link)
        return replace(self, links=tuple(links))

    # ---------- 序列化 ----------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "armlab.arm_spec.v1",
            "name": self.name,
            "dof": self.dof,
            "gravity": list(self.gravity),
            "links": [link.to_dict() for link in self.links],
            "meta": dict(self.meta),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ArmSpec":
        return ArmSpec(
            name=d.get("name", "arm"),
            links=tuple(LinkSpec.from_dict(ld) for ld in d["links"]),
            gravity=_as_tuple3(d.get("gravity", (0, -9.81, 0))),
            meta=dict(d.get("meta", {})),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> "ArmSpec":
        return ArmSpec.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _link_tip_position(link: LinkSpec, default: float = 0.25) -> Vec3:
    """从可视化几何推断连杆远端位置（默认 x 方向）。"""
    geom = link.geom or {}
    ft = geom.get("fromto")
    if geom.get("type") == "capsule" and ft and len(ft) == 6:
        return (float(ft[3]), float(ft[4]), float(ft[5]))
    return (default, 0.0, 0.0)


# ---------------------------------------------------------------------------
# 工厂函数：把具体机器人装配成 ArmSpec
# ---------------------------------------------------------------------------

def planar2r(
    L1: float = 0.300,
    L2: float = 0.250,
    m1: float = 1.50,
    m2: float = 0.90,
    w1: float = 0.050,
    t1: float = 0.030,
    w2: float = 0.040,
    t2: float = 0.024,
    lc1: Optional[float] = None,
    lc2: Optional[float] = None,
    payload: float = 0.0,
    q1_limits_deg: Tuple[float, float] = (-160.0, 160.0),
    q2_limits_deg: Tuple[float, float] = (-150.0, 150.0),
    damping: Tuple[float, float] = (0.02, 0.02),
    frictionloss: Tuple[float, float] = (0.05, 0.05),
    armature: Tuple[float, float] = (0.0, 0.0),
    gravity: Vec3 = (0.0, -9.81, 0.0),
    name: str = "arm2dof",
) -> ArmSpec:
    """构建 2 自由度平面臂（竖直面内，重力在面内）。

    参数：L1/L2 连杆长；m1/m2 质量；w/t 为连杆截面宽/厚（用于生成 3D 惯量）；
    lc 为质心距近端关节距离（默认 L/2 均质杆）；payload 为末端负载质量。
    """
    lc1 = L1 / 2.0 if lc1 is None else float(lc1)
    lc2 = L2 / 2.0 if lc2 is None else float(lc2)

    base = LinkSpec(
        name="base", pos=(0.0, 0.0, 0.0), mass=0.0,
        com=(0.0, 0.0, 0.0),
        inertia=((0.0, 0, 0), (0, 0, 0), (0, 0, 0)), joint=None,
        geom={"type": "cylinder", "size": [0.06, 0.02], "pos": [0.0, 0.0, 0.0],
              "rgba": [0.25, 0.25, 0.30, 1.0]},
    )
    link1 = LinkSpec(
        name="link1", pos=(0.0, 0.0, 0.0), mass=m1, com=(lc1, 0.0, 0.0),
        inertia=box_inertia(m1, L1, w1, t1),
        joint=JointSpec(name="q1", axis=(0.0, 0.0, 1.0),
                        lower=q1_limits_deg[0] * DEG, upper=q1_limits_deg[1] * DEG,
                        damping=damping[0], frictionloss=frictionloss[0], armature=armature[0]),
        geom={"type": "capsule", "fromto": [0.0, 0.0, 0.0, L1, 0.0, 0.0],
              "size": [w1 / 2.0], "rgba": [0.20, 0.45, 0.85, 1.0]},
    )
    link2 = LinkSpec(
        name="link2", pos=(L1, 0.0, 0.0), mass=m2, com=(lc2, 0.0, 0.0),
        inertia=box_inertia(m2, L2, w2, t2),
        joint=JointSpec(name="q2", axis=(0.0, 0.0, 1.0),
                        lower=q2_limits_deg[0] * DEG, upper=q2_limits_deg[1] * DEG,
                        damping=damping[1], frictionloss=frictionloss[1], armature=armature[1]),
        geom={"type": "capsule", "fromto": [0.0, 0.0, 0.0, L2, 0.0, 0.0],
              "size": [w2 / 2.0], "rgba": [0.85, 0.45, 0.15, 1.0]},
    )
    spec = ArmSpec(name=name, links=(base, link1, link2), gravity=gravity,
                   meta={"config": "planar-2R", "plane": "vertical(xOy, gravity -y)",
                         "unit": "SI", "baseline": dict(BASELINE)})
    if payload:
        spec = spec.with_tool(payload)
    return spec


def planar_ndof(
    lengths: Sequence[float],
    masses: Sequence[float],
    widths: Optional[Sequence[float]] = None,
    thicknesses: Optional[Sequence[float]] = None,
    payload: float = 0.0,
    limits_deg: Tuple[float, float] = (-170.0, 170.0),
    gravity: Vec3 = (0.0, -9.81, 0.0),
    name: str = "planar_ndof",
) -> ArmSpec:
    """构建 n 自由度平面臂（全部关节轴平行 z）——3/4 轴等后续阶段直接复用。"""
    n = len(lengths)
    widths = list(widths) if widths is not None else [0.05 * (0.8 ** i) for i in range(n)]
    thicknesses = list(thicknesses) if thicknesses is not None else [0.03 * (0.8 ** i) for i in range(n)]
    links: List[LinkSpec] = [LinkSpec(name="base", mass=0.0, joint=None,
                                      geom={"type": "cylinder", "size": [0.06, 0.02],
                                            "pos": [0.0, 0.0, 0.0]})]
    parent_pos: Vec3 = (0.0, 0.0, 0.0)
    for i, (L, m, w, t) in enumerate(zip(lengths, masses, widths, thicknesses)):
        links.append(LinkSpec(
            name=f"link{i + 1}", pos=parent_pos, mass=float(m), com=(L / 2.0, 0.0, 0.0),
            inertia=box_inertia(float(m), float(L), float(w), float(t)),
            joint=JointSpec(name=f"q{i + 1}", lower=limits_deg[0] * DEG, upper=limits_deg[1] * DEG,
                            damping=0.02, frictionloss=0.05),
            geom={"type": "capsule", "fromto": [0.0, 0.0, 0.0, float(L), 0.0, 0.0],
                  "size": [float(w) / 2.0]},
        ))
        parent_pos = (float(L), 0.0, 0.0)
    spec = ArmSpec(name=name, links=tuple(links), gravity=gravity,
                   meta={"config": f"planar-{n}R", "unit": "SI"})
    if payload:
        spec = spec.with_tool(payload)
    return spec


def scara_teach_4rprr(
    a1: float = 0.225,
    a2: float = 0.225,
    d1: float = 0.300,
    stroke: float = 0.200,
    m1: float = 3.0,
    lc1: float = 0.110,
    m2: float = 1.5,
    lc2: float = 0.105,
    mt: float = 3.0,          # 滑块 + 负载的总质量（m3 + mload）
    I4: float = 2.0e-3,
    q1_limits_deg: Tuple[float, float] = (-130.0, 130.0),
    q2_limits_deg: Tuple[float, float] = (-145.0, 145.0),
    damping: Tuple[float, float, float, float] = (0.50, 0.40, 20.0, 0.02),
    frictionloss: Tuple[float, float, float, float] = (0.30, 0.25, 10.0, 0.01),
    name: str = "scara-teach-4RPRR",
) -> ArmSpec:
    """SCARA 4 轴（R-R-P-R）：复用已有 SCARA 教学机参数（`scara_params.m` / `scara.xml`）。

    结构与约定（与已有模型逐位一致）：
        - 关节轴全部沿 ±z；J1/J2 转动（基座 300 mm 上方 → 水平面内运动）；
        - J3 为**移动副，轴 = −z**（q3 增大 ⇒ 末端下降，z = d1 − q3）★ 最容易写错的一点；
        - J4 末端自转；TCP 与 link4 原点重合（d4 = 0）；
        - 重力沿 −z（SCARA 在水平面工作，重力只作用在 J3 上）。

    惯量口径：I1 = m1·a1²/12、I2 = m2·a2²/12（绕各自质心，绕 z 轴），
    link3/link4 的绕 z 惯量分别取 1e-12 与 I4（与 MJCF 的合法性约束一致）。
    """
    links = [
        LinkSpec(name="base", pos=(0.0, 0.0, 0.0), mass=0.0, joint=None,
                 geom={"type": "cylinder", "size": [0.09, 0.04], "pos": [0.0, 0.0, -0.04],
                       "rgba": [0.25, 0.25, 0.30, 1.0]}),
        LinkSpec(
            name="link1", pos=(0.0, 0.0, d1), mass=m1, com=(lc1, 0.0, 0.0),
            inertia=((1e-6, 0.0, 0.0),
                     (0.0, m1 * a1 ** 2 / 12.0, 0.0),
                     (0.0, 0.0, m1 * a1 ** 2 / 12.0)),
            joint=JointSpec(name="q1", axis=(0.0, 0.0, 1.0),
                            lower=q1_limits_deg[0] * DEG, upper=q1_limits_deg[1] * DEG,
                            damping=damping[0], frictionloss=frictionloss[0]),
            geom={"type": "capsule", "fromto": [0.0, 0.0, 0.0, a1, 0.0, 0.0],
                  "size": [0.02], "rgba": [0.25, 0.60, 1.00, 0.95]},
        ),
        LinkSpec(
            name="link2", pos=(a1, 0.0, 0.0), mass=m2, com=(lc2, 0.0, 0.0),
            inertia=((1e-6, 0.0, 0.0),
                     (0.0, m2 * a2 ** 2 / 12.0, 0.0),
                     (0.0, 0.0, m2 * a2 ** 2 / 12.0)),
            joint=JointSpec(name="q2", axis=(0.0, 0.0, 1.0),
                            lower=q2_limits_deg[0] * DEG, upper=q2_limits_deg[1] * DEG,
                            damping=damping[1], frictionloss=frictionloss[1]),
            geom={"type": "capsule", "fromto": [0.0, 0.0, 0.0, a2, 0.0, 0.0],
                  "size": [0.018], "rgba": [0.30, 0.90, 0.45, 0.95]},
        ),
        LinkSpec(
            name="link3", pos=(a2, 0.0, 0.0), mass=mt, com=(0.0, 0.0, 0.0),
            inertia=((1e-12, 0.0, 0.0), (0.0, 1e-12, 0.0), (0.0, 0.0, 1e-12)),
            joint=JointSpec(name="q3", type="prismatic", axis=(0.0, 0.0, -1.0),
                            lower=0.0, upper=stroke,
                            damping=damping[2], frictionloss=frictionloss[2]),
            # 活塞杆向上伸进套筒（套筒画在 link2 上，见 scara_scene）：工具"伸出/缩回"时
            # 活塞始终藏在套筒里，视觉正确且不会穿透工作台。
            geom={"type": "capsule", "fromto": [0.0, 0.0, 0.0, 0.0, 0.0, 0.06],
                  "size": [0.024], "rgba": [1.00, 0.62, 0.15, 0.95]},
        ),
        LinkSpec(
            name="link4", pos=(0.0, 0.0, 0.0), mass=1e-12, com=(0.0, 0.0, 0.0),
            inertia=((I4, 0.0, 0.0), (0.0, I4, 0.0), (0.0, 0.0, I4)),
            joint=JointSpec(name="q4", axis=(0.0, 0.0, 1.0),
                            lower=-2 * math.pi, upper=2 * math.pi,
                            damping=damping[3], frictionloss=frictionloss[3]),
            geom={"type": "cylinder", "size": [0.03, 0.02], "rgba": [0.98, 0.30, 0.30, 0.95]},
        ),
    ]
    return ArmSpec(
        name=name, links=tuple(links), gravity=(0.0, 0.0, -9.81),
        meta={"config": "SCARA-4RPRR",
              "source": "复用 scara-teach-4RPRR（scara_params.m / scara.xml）",
              "note": "J3 为移动副、轴 −z：z = d1 − q3；水平面内运动，重力沿 −z",
              "unit": "SI"},
    )
