"""SCARA 抓取-放置场景（MuJoCo，偏真实感渲染 + 两指夹爪 + 相机）。

场景要素
--------
- 机器人：`spec.scara_teach_4rprr()`（4 轴 R-R-P-R），外加
  **J1 底座**（底板 + 立柱）与 **两指平行夹爪**（两只手指各一根 slide 关节）；
- 环境：地面 + 四腿工作台 + 放置区 + 自由刚体（30 mm 立方体，50 g）；
- 渲染：`<light>` 主光+补光 + 柔化阴影 + 金属/塑料/木质材质 + 渐变天空盒；
- 相机：`cam_front`（工作台正前方）、`cam_top`（俯视）、`cam_wrist`（腕部随动），
  后续做视觉策略时用 `mujoco.Renderer` 直接取图。

抓取模型
--------
**真实摩擦抓取**：手指是带高摩擦垫的实体，靠位置伺服闭合压住物体；
另提供"吸附辅助"（weld，默认关闭）作为兜底，见 `grasp_assist`。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .spec import ArmSpec
from . import mjcf as _mjcf

# ---------------------------------------------------------------------------
# 几何常量（与 scara_task 的动作高度配套）
# ---------------------------------------------------------------------------
TABLE_TOP = 0.10
# 台面往 +x 挪开：让 J1 基座（最大半径 0.11）完全站在桌外，避免立柱穿过台面
TABLE_CENTER = (0.40, 0.0)
TABLE_HALF = (0.26, 0.26)          # 台面半尺寸（0.52 × 0.52 m）
TABLE_THICK = 0.02
LEG_HALF = 0.018
OBJ_SIZE = 0.03
GRIP_OPEN = 0.012                  # 手指行程（每侧）
# 闭合时间距略小于物体宽度（2 mm 过盈）——否则正压力为 0，摩擦抓取会打滑
GRIP_GAP_CLOSED = 0.028
FINGER_SPAN_Z = (0.02, 0.060)      # 手指相对工具原点向下的范围（缩短：抓取时指尖离台面 20 mm）


def _finger_xml(name: str, side: int) -> List[str]:
    """单只手指（side = -1 左 / +1 右）。side*joint axis 朝外张开。"""
    x0 = side * (GRIP_GAP_CLOSED / 2 + 0.004)       # 闭合位置（内表面贴住物体）
    axis = f"{side} 0 0"                            # 向外为正位移
    return [
        f'<body name="finger_{name}" pos="{x0:.4f} 0 -0.02">',
        f'  <joint name="grip_{name}" type="slide" axis="{axis}" '
        f'range="0 {GRIP_OPEN}" damping="2"/>',
        '  <geom name="finger_pad_' + name + '" type="box" '
        'size="0.004 0.015 0.020" pos="0 0 -0.040" '
        'mass="0.03" contype="1" conaffinity="1" friction="1.6 0.05 0.001" '
        'solref="0.008 1" rgba="0.25 0.28 0.32 1"/>',
        '</body>',
    ]


def build_scara_pick_place_xml(
    spec: ArmSpec,
    object_pos: Sequence[float] = (0.30, -0.05, TABLE_TOP + OBJ_SIZE / 2),
    object_size: Sequence[float] = (OBJ_SIZE, OBJ_SIZE, OBJ_SIZE),
    object_mass: float = 0.05,
    place_center: Sequence[float] = (0.30, 0.12),
    place_size: Sequence[float] = (0.05, 0.05),
    table: bool = True,
    gripper: bool = True,
    timestep: float = 1e-4,
    integrator: str = "RK4",
    scene_scale: float = 1.0,
) -> str:
    """生成（偏真实感的）SCARA 抓取-放置场景 MJCF。"""
    ox, oy, oz = object_pos
    sx, sy, sz = object_size
    px, py = place_center
    psx, psy = place_size
    a2 = float(spec.links[3].pos[0])          # 小臂长（套筒画在小臂末端）

    # ---------------- worldbody 附加内容 ----------------
    world: List[str] = []
    world.append('    <geom name="floor" type="plane" size="4 4 0.05" pos="0 0 0" '
                 'material="floor_mat" contype="0" conaffinity="0"/>')
    world.append('    <light name="key" pos="0.55 0.35 1.15" dir="-0.35 -0.22 -1" '
                 'directional="false" diffuse="0.95 0.95 0.95" specular="0.35 0.35 0.35" '
                 'castshadow="true"/>')
    world.append('    <light name="fill" pos="-0.35 -0.45 0.75" dir="0.35 0.45 -0.8" '
                 'directional="false" diffuse="0.45 0.5 0.6" specular="0.1 0.1 0.12" '
                 'castshadow="false"/>')

    if table:
        cx, cy = TABLE_CENTER
        hx, hy = TABLE_HALF
        world.append(f'    <geom name="table_top" type="box" size="{hx} {hy} {TABLE_THICK / 2}" '
                     f'pos="{cx} {cy} {TABLE_TOP - TABLE_THICK / 2}" '
                     'material="table_mat" contype="1" conaffinity="1" '
                     'friction="0.7 0.05 0.001"/>')
        leg_h = TABLE_TOP - TABLE_THICK
        for sx_, sy_ in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            lx = cx + sx_ * (hx - 0.03)
            ly = cy + sy_ * (hy - 0.03)
            world.append(f'    <geom name="leg_{sx_}{sy_}" type="box" '
                         f'size="{LEG_HALF} {LEG_HALF} {leg_h / 2}" '
                         f'pos="{lx} {ly} {leg_h / 2}" material="metal_mat" '
                         'contype="1" conaffinity="1"/>')
    world += [
        f'    <geom name="place_zone" type="box" size="{psx} {psy} 0.001" '
        f'pos="{px} {py} {TABLE_TOP + 0.001}" contype="0" conaffinity="0" '
        'material="zone_mat"/>',
        f'    <body name="object" pos="{ox} {oy} {oz}">',
        '      <freejoint name="object_free"/>',
        f'      <geom name="object_geom" type="box" size="{sx / 2} {sy / 2} {sz / 2}" '
        f'mass="{object_mass}" contype="1" conaffinity="1" friction="1.0 0.05 0.001" '
        'solref="0.01 1" material="object_mat"/>',
        '    </body>',
    ]

    # ---------------- 机器人附加几何/子 body ----------------
    link_geoms: Dict[str, List[str]] = {
        # J1 底座：底板 → 立柱 → 轴承法兰 → 细轴。
        # 大臂胶囊半径 0.02、位于 z=0.30 ⇒ 最低点 z=0.28；因此**轴顶停在 0.279**，
        # 既像真实轴承座，又不会像之前的 Ø150 顶盖那样插进大臂里（视觉重叠）。
        "base": [
            '<geom name="base_plate" type="box" size="0.11 0.11 0.01" pos="0 0 0.01" '
            'material="metal_mat" contype="1" conaffinity="1"/>',
            '<geom name="base_column" type="cylinder" size="0.055 0.11" pos="0 0 0.13" '
            'material="metal_mat" contype="1" conaffinity="1"/>',
            '<geom name="base_flange" type="cylinder" size="0.075 0.012" pos="0 0 0.212" '
            'material="dark_mat" contype="1" conaffinity="1"/>',
            '<geom name="base_shaft" type="cylinder" size="0.020 0.0275" pos="0 0 0.2515" '
            'material="dark_mat" contype="1" conaffinity="1"/>',
        ],
        # J3 固定套筒：刻意做短（0.12 m），下端 z=0.18 → 不会与台面物体干涉
        "link2": [
            f'<geom name="z_housing" type="capsule" fromto="{a2} 0 0 {a2} 0 -0.12" '
            'size="0.030" contype="0" conaffinity="0" material="silver_mat"/>',
        ],
    }
    link_bodies: Dict[str, List[str]] = {}
    actuators: List[str] = []
    if gripper:
        # 掌部（固定在 link4 上）+ 腕部相机
        link_geoms["link4"] = [
            '<geom name="palm" type="box" size="0.035 0.028 0.010" pos="0 0 -0.010" '
            'contype="0" conaffinity="0" material="dark_mat"/>',
            # ── J4 外侧支架：立柱 + 横梁，把第一人称相机送到"外侧 165 mm、上方 70 mm"──
            # 这个位置能同时看到爪尖、被夹物与周围台面，且不会被机械臂自身遮挡。
            '<geom name="bracket_riser" type="box" size="0.008 0.010 0.035" '
            'pos="0.045 0 0.045" contype="0" conaffinity="0" material="dark_mat"/>',
            '<geom name="bracket_arm" type="box" size="0.065 0.010 0.008" '
            'pos="0.115 0 0.088" contype="0" conaffinity="0" material="dark_mat"/>',
            '<geom name="cam_body" type="box" size="0.014 0.016 0.014" '
            'pos="0.165 0 0.078" contype="0" conaffinity="0" material="metal_mat"/>',
            # 第一人称相机：看向"爪尖下方 50 mm"（≈被抓物中心高度）
            '<camera name="cam_eye" pos="0.165 0 0.070" xyaxes="0 1 0 -0.588 0 0.808" '
            'fovy="55"/>',
            # 保留一个贴身腕部相机（更近的抓取近景）
            '<camera name="cam_wrist" pos="0.05 0 0.03" mode="fixed" fovy="58"/>',
        ]
        link_bodies["link4"] = _finger_xml("l", -1) + _finger_xml("r", +1)
        actuators += [
            '    <position name="grip_l_motor" joint="grip_l" kp="900" kv="6" '
            f'ctrlrange="0 {GRIP_OPEN}"/>',
            '    <position name="grip_r_motor" joint="grip_r" kp="900" kv="6" '
            f'ctrlrange="0 {GRIP_OPEN}"/>',
        ]

    # ---------------- 资产 / 视觉 ----------------
    assets = """  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="0.42 0.55 0.78" rgb2="0.95 0.96 1.0" width="256" height="256"/>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.86 0.87 0.89" rgb2="0.78 0.80 0.84" width="300" height="300"/>
    <texture name="table_tex" type="2d" builtin="checker" rgb1="0.80 0.72 0.60" rgb2="0.74 0.66 0.54" width="300" height="300"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="10 10" reflectance="0.05"/>
    <material name="table_mat" texture="table_tex" texrepeat="3 3" reflectance="0.08"/>
    <material name="metal_mat" rgba="0.78 0.80 0.84 1" specular="0.8" shininess="0.6" reflectance="0.15"/>
    <material name="silver_mat" rgba="0.82 0.85 0.90 1" specular="0.9" shininess="0.7"/>
    <material name="dark_mat" rgba="0.22 0.24 0.28 1" specular="0.5" shininess="0.5"/>
    <material name="zone_mat" rgba="0.15 0.95 0.35 0.75" reflectance="0.05"/>
    <material name="object_mat" rgba="1.0 0.42 0.18 1" specular="0.4" shininess="0.4"/>
  </asset>"""
    head = """  <visual>
    <headlight ambient="0.35 0.35 0.40" diffuse="0.30 0.30 0.35" specular="0.15 0.15 0.15"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map stiffness="0.02" znear="0.02"/>
    <global azimuth="140" elevation="-18" offwidth="1920" offheight="1080"/>
  </visual>"""

    extra_xml = (
        '  <equality>\n'
        '    <!-- 吸附辅助（默认关闭）：需要时由 grasp_assist() 打开 -->\n'
        '    <weld name="grasp_assist" body1="object" body2="link4" active="false"\n'
        '          anchor="0 0 0" relpose="0 0 0 1 0 0 0"/>\n'
        '  </equality>'
    )

    # 固定视角相机（相机放在世界坐标系下，看向工作台）
    cameras = (
        # 相机默认朝自身 -z 看；xyaxes 给出相机的 x/y 轴（world 坐标）
        '    <camera name="cam_front" pos="0.26 -0.95 0.55" '
        'xyaxes="1 0 0 0 0.428 0.904" fovy="42"/>\n'
        # 俯视相机：整个工作台正上方，正对台面中心 (0.26, 0)，覆盖 0.72×0.52 m 桌面
        f'    <camera name="cam_top" pos="{TABLE_CENTER[0]} {TABLE_CENTER[1]} 1.25" '
        'xyaxes="1 0 0 0 1 0" fovy="38"/>'
    )
    world.insert(0, cameras)

    return _mjcf.build_xml(
        spec,
        include_friction=True,
        timestep=timestep,
        integrator=integrator,
        extra_head=head,
        extra_assets=assets,
        extra_worldbody="\n".join(world),
        extra_link_geoms=link_geoms,
        extra_link_bodies=link_bodies,
        extra_actuators="\n".join(actuators),
        extra_xml=extra_xml,
    )


def build_model(spec: ArmSpec, **kwargs):
    """生成场景 MuJoCo 模型。"""
    import mujoco
    return mujoco.MjModel.from_xml_string(build_scara_pick_place_xml(spec, **kwargs))


# ---------------------------------------------------------------------------
# 便捷访问器
# ---------------------------------------------------------------------------

def ids(model, kind: str, name: str) -> int:
    import mujoco
    obj = {"body": mujoco.mjtObj.mjOBJ_BODY, "site": mujoco.mjtObj.mjOBJ_SITE,
           "camera": mujoco.mjtObj.mjOBJ_CAMERA, "equality": mujoco.mjtObj.mjOBJ_EQUALITY,
           "joint": mujoco.mjtObj.mjOBJ_JOINT}[kind]
    return mujoco.mj_name2id(model, obj, name)


def object_qpos_addr(model, name: str = "object") -> int:
    import mujoco
    return int(model.jnt_qposadr[ids(model, "joint", "object_free")])


def arm_joint_addr(model) -> slice:
    """手臂 4 个关节在 qpos/qvel 中的下标（手指与物体排在其后）。"""
    import mujoco
    adr = [int(model.jnt_qposadr[ids(model, "joint", j)]) for j in ("q1", "q2", "q3", "q4")]
    return slice(min(adr), max(adr) + 1)


def gripper_ctrl(model, data, opening: float) -> None:
    """设置两指张开量（0=闭合，GRIP_OPEN=全开）。"""
    for name in ("grip_l_motor", "grip_r_motor"):
        aid = _actuator_id(model, name)
        if aid >= 0:
            data.ctrl[aid] = opening


def _actuator_id(model, name: str) -> int:
    import mujoco
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def grab(model, data, eq_id: int, object_body: str = "object",
         tool_body: str = "link4") -> bool:
    """吸附辅助：把当前"物体相对工具"的位姿写入 weld 并激活。"""
    import mujoco
    obj = ids(model, "body", object_body)
    tool = ids(model, "body", tool_body)
    R_t = data.xmat[tool].reshape(3, 3)
    rel_pos = R_t.T @ (data.xpos[obj] - data.xpos[tool])
    R_rel = R_t.T @ data.xmat[obj].reshape(3, 3)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R_rel.reshape(9))
    model.eq_data[eq_id, 0:3] = 0.0
    model.eq_data[eq_id, 3:6] = rel_pos
    model.eq_data[eq_id, 6:10] = q
    data.eq_active[eq_id] = 1
    return True


def release(model, data, eq_id: int) -> None:
    data.eq_active[eq_id] = 0


# ---------------------------------------------------------------------------
# 闭式逆解（与既有 SCARA 模型一致）
# ---------------------------------------------------------------------------

def scara_ik(x: float, y: float, z: float, psi: float, spec: ArmSpec,
             elbow: int = -1) -> Optional[np.ndarray]:
    """SCARA 闭式逆解：[q1, q2, q3, q4]；不可达返回 None（q3 增大 ⇒ 末端下降）。"""
    a1 = float(spec.links[2].pos[0])
    a2 = float(spec.links[3].pos[0])
    d1 = float(spec.links[1].pos[2])
    stroke = float(spec.joints[2].upper)
    r2 = x * x + y * y
    c2 = (r2 - a1 * a1 - a2 * a2) / (2.0 * a1 * a2)
    if abs(c2) > 1.0:
        return None
    q2 = elbow * float(np.arccos(np.clip(c2, -1.0, 1.0)))
    q1 = float(np.arctan2(y, x) - np.arctan2(a2 * np.sin(q2), a1 + a2 * np.cos(q2)))
    q3 = d1 - z
    if q3 < -1e-9 or q3 > stroke + 1e-9:
        return None
    q4 = float((psi - q1 - q2 + np.pi) % (2 * np.pi) - np.pi)
    return np.array([q1, q2, float(np.clip(q3, 0.0, stroke)), q4])
