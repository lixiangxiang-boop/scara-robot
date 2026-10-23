"""ArmSpec → MuJoCo MJCF 生成器（任意自由度）与 MuJoCo 交叉验证工具。

生成的模型与 armLab 的数学约定 1:1 对应（连杆系在近端关节、x 轴指向远端），
因此可以直接用于"数学模型 vs MuJoCo"的严格对比：
    M(q)、qfrc_bias(=C·q̇+G)、自由落体轨迹、执行器响应。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .spec import ArmSpec


def _vec(v: Sequence[float]) -> str:
    return " ".join(f"{float(x):.12g}" for x in v)


def _geom_xml(geom: Dict[str, Any], indent: str) -> str:
    attrs = [f'type="{geom.get("type", "box")}"']
    for key, attr in (("size", "size"), ("fromto", "fromto"), ("pos", "pos"), ("rgba", "rgba")):
        if key in geom:
            attrs.append(f'{attr}="{_vec(geom[key])}"')
    return f"{indent}<geom {' '.join(attrs)}/>"


def tip_point(spec: ArmSpec) -> Tuple[float, float, float]:
    """末端点（最后一个连杆坐标系下）。"""
    last = spec.links[-1]
    if not last.is_moving:
        return last.pos  # type: ignore[return-value]
    geom = last.geom or {}
    ft = geom.get("fromto")
    if geom.get("type") == "capsule" and ft and len(ft) == 6:
        return (float(ft[3]), float(ft[4]), float(ft[5]))
    return (0.0, 0.0, 0.0)


def build_xml(spec: ArmSpec, include_friction: bool = True, timestep: float = 1e-3,
              integrator: str = "RK4", use_geoms: bool = True,
              extra_worldbody: str = "", extra_link_geoms: Dict[str, List[str]] | None = None,
              extra_link_bodies: Dict[str, List[str]] | None = None,
              extra_actuators: str = "",
              extra_xml: str = "", extra_head: str = "", extra_assets: str = "") -> str:
    """生成 MJCF 字符串。

    include_friction=False：把 damping/frictionloss/armature 置零，
    用于与理想数学模型做严格交叉验证。
    extra_worldbody / extra_link_geoms / extra_xml：供场景脚本注入工作台、被抓物、
    吸盘几何与 equality 约束（见 armlab.scara_scene）。
    extra_head：注入 <visual>（出现在 <default> 之前）；extra_assets：注入 <asset>
    （出现在 <default> 之后、<worldbody> 之前）——MJCF 对段序有要求。
    extra_link_bodies：给指定连杆注入**子 body**（如夹爪手指，含 slide 关节）；
    extra_actuators：注入额外执行器（如手指的位置伺服）。
    """
    lines: List[str] = []
    lines.append(f'<mujoco model="{spec.name}">')
    lines.append('  <compiler angle="radian" autolimits="true"/>')
    lines.append(f'  <option timestep="{timestep:g}" integrator="{integrator}" '
                 f'gravity="{_vec(spec.gravity)}"/>')
    if extra_head:
        lines.append(extra_head.rstrip("\n"))
    lines.append('  <default>')
    lines.append('    <geom contype="0" conaffinity="0" density="0" group="1"/>')
    lines.append('  </default>')
    if extra_assets:
        lines.append(extra_assets.rstrip("\n"))
    lines.append('  <worldbody>')

    unit = "  "
    for i, link in enumerate(spec.links):
        ind = unit * (2 + i)
        lines.append(f'{ind}<body name="{link.name}" pos="{_vec(link.pos)}" '
                     f'quat="{_vec(link.quat)}">')
        inner = unit * (3 + i)
        if link.is_moving:
            joint = link.joint
            assert joint is not None
            jattrs = [f'name="{joint.name}"',
                      'type="hinge"' if joint.type == "revolute" else 'type="slide"',
                      f'axis="{_vec(joint.axis)}"', 'pos="0 0 0"',
                      f'range="{joint.lower:.12g} {joint.upper:.12g}"', 'limited="true"']
            if include_friction:
                if joint.damping:
                    jattrs.append(f'damping="{joint.damping:.12g}"')
                if joint.frictionloss:
                    jattrs.append(f'frictionloss="{joint.frictionloss:.12g}"')
                if joint.armature:
                    jattrs.append(f'armature="{joint.armature:.12g}"')
            lines.append(f"{inner}<joint {' '.join(jattrs)}/>")
        if link.mass > 0.0:
            I = link.inertia_matrix()
            lines.append(
                f'{inner}<inertial pos="{_vec(link.com)}" mass="{link.mass:.12g}" '
                f'fullinertia="{I[0, 0]:.12g} {I[1, 1]:.12g} {I[2, 2]:.12g} '
                f'{I[0, 1]:.12g} {I[0, 2]:.12g} {I[1, 2]:.12g}"/>')
        if use_geoms and link.geom:
            lines.append(_geom_xml(link.geom, inner))
        for extra in (extra_link_geoms or {}).get(link.name, []):
            lines.append(f"{inner}{extra}")
        for extra in (extra_link_bodies or {}).get(link.name, []):
            lines.append(f"{inner}{extra}")

    site_indent = 3 + len(spec.links)
    lines.append(f'{unit * site_indent}<site name="ee" pos="{_vec(tip_point(spec))}" '
                 'size="0.005" rgba="0 0.8 0 1"/>')
    for d in range(1 + len(spec.links), 1, -1):
        lines.append(f"{unit * d}</body>")
    if extra_worldbody:
        lines.append(extra_worldbody.rstrip("\n"))
    lines.append('  </worldbody>')
    lines.append('  <actuator>')
    for joint in spec.joints:
        lines.append(f'    <motor name="tau_{joint.name}" joint="{joint.name}" gear="1"/>')
    if extra_actuators:
        lines.append(extra_actuators.rstrip("\n"))
    lines.append('  </actuator>')
    if extra_xml:
        lines.append(extra_xml.rstrip("\n"))
    lines.append('</mujoco>')
    return "\n".join(lines)


def build_model(spec: ArmSpec, **kwargs):
    """生成 MuJoCo 模型（需要安装 mujoco）。"""
    import mujoco
    return mujoco.MjModel.from_xml_string(build_xml(spec, **kwargs))


# ---------------------------------------------------------------------------
# 交叉验证工具
# ---------------------------------------------------------------------------

def mass_matrix(model, data, q: Sequence[float]) -> np.ndarray:
    """MuJoCo 关节空间质量矩阵（含 armature）。"""
    import mujoco
    data.qpos[:] = np.asarray(q, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    M = np.zeros((model.nv, model.nv))
    try:                        # MuJoCo >= 3.2: mj_fullM(m, d, dst)
        mujoco.mj_fullM(model, data, M)
    except TypeError:           # 旧版本: mj_fullM(m, dst, sparse M)
        sparse = getattr(data, "M", None)
        if sparse is None:
            sparse = data.qM
        mujoco.mj_fullM(model, M, sparse)
    return M


def bias_forces(model, data, q: Sequence[float], dq: Sequence[float]) -> np.ndarray:
    """MuJoCo 偏置力矩 qfrc_bias = C(q,q̇)q̇ + G(q)。"""
    import mujoco
    data.qpos[:] = np.asarray(q, dtype=float)
    data.qvel[:] = np.asarray(dq, dtype=float)
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)
    return data.qfrc_bias.copy()


def gravity_torque(model, data, q: Sequence[float]) -> np.ndarray:
    return bias_forces(model, data, q, np.zeros(model.nv))


def forward_dynamics(model, data, q: Sequence[float], dq: Sequence[float],
                     tau: Sequence[float]) -> np.ndarray:
    """MuJoCo 无约束前向动力学：q̈ = M⁻¹(τ - C·q̇ - G)。"""
    import mujoco
    data.qpos[:] = np.asarray(q, dtype=float)
    data.qvel[:] = np.asarray(dq, dtype=float)
    data.qfrc_applied[:] = np.asarray(tau, dtype=float)
    mujoco.mj_forward(model, data)
    return data.qacc.copy()


def ee_position(model, data, q: Sequence[float]) -> np.ndarray:
    """MuJoCo 中末端 site 的世界坐标。"""
    import mujoco
    data.qpos[:] = np.asarray(q, dtype=float)
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    return data.site_xpos[site_id].copy()


def make_data(model):
    import mujoco
    return mujoco.MjData(model)
