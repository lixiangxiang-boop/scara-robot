"""从 MuJoCo 场景生成 ROS2 资产（保证"两边同一套参数"）。

输入（唯一数据源，均来自 python/armlab）：
    spec.py        -> scara_teach_4rprr()：臂长/立柱/行程/质量/惯量/限位
    scara_scene.py -> build_scara_pick_place_xml()：台面/四腿/放置区/物块/相机

输出：
    src/scara_description/urdf/params.generated.xacro   （URDF 的数字来源）
    src/scara_bringup/worlds/scara_world.generated.sdf  （Gazebo 世界，含相机）
    outputs/data/ros2_scene_params.json                 （对照报告）

运行：python ros2_ws/tools/gen_ros_assets.py
"""

from __future__ import annotations

import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from armlab import scara_scene as SC  # noqa: E402
from armlab import spec as S          # noqa: E402

OUT_XACRO = ROOT / "ros2_ws" / "src" / "scara_description" / "urdf" / "params.generated.xacro"
OUT_WORLD = ROOT / "ros2_ws" / "src" / "scara_bringup" / "worlds" / "scara_world.generated.sdf"
OUT_JSON = ROOT / "outputs" / "data" / "ros2_scene_params.json"

IMG_W, IMG_H = 320, 240   # 腕部相机：320x240 @10 Hz（WSL 软件渲染下减轻负担）


def vec(text: str) -> list[float]:
    return [float(v) for v in text.replace(",", " ").split()]


def normalize(a: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else a


def rot_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵 -> (roll, pitch, yaw)，SDF 的 R = Rz(yaw) Ry(pitch) Rx(roll)。"""
    pitch = -math.asin(max(-1.0, min(1.0, float(R[2, 0]))))
    if abs(math.cos(pitch)) > 1e-6:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:                                    # 万向节锁
        roll = 0.0
        yaw = math.atan2(-float(R[0, 1]), float(R[1, 1]))
    return roll, pitch, yaw


def parse_cameras(mjcf: str) -> dict[str, dict]:
    """从 MJCF 提取相机：名字、父 body、pos、xyaxes、fovy。"""
    root = ET.fromstring(mjcf)
    world = root.find("worldbody")
    out: dict[str, dict] = {}

    def add(cam: ET.Element, parent: str) -> None:
        name = cam.get("name")
        if not name:
            return
        xy = cam.get("xyaxes")
        out[name] = {
            "parent": parent,
            "pos": vec(cam.get("pos", "0 0 0")),
            "xyaxes": vec(xy) if xy else [1, 0, 0, 0, 1, 0],
            "fovy_deg": float(cam.get("fovy", "45")),
        }

    for cam in world.findall("camera"):
        add(cam, "world")
    for body in world.iter("body"):
        for cam in body.findall("camera"):
            add(cam, body.get("name", "?"))
    return out


def camera_pose(cam: dict) -> dict:
    """把 MuJoCo 相机（x/y 轴 + fovy）换算成 Gazebo 相机位姿与水平视场角。

    MuJoCo：相机朝 -z，xyaxes 给出相机 x/y 轴（在父坐标系里）。
    Gazebo：相机朝 +x。取 f = -z_mj、up = y_mj，则
        x_sensor = f, y_sensor = up × f（左），z_sensor = up
    """
    x = normalize(np.array(cam["xyaxes"][0:3], dtype=float))
    y = normalize(np.array(cam["xyaxes"][3:6], dtype=float))
    z = normalize(np.cross(x, y))
    f = -z
    up = y
    left = normalize(np.cross(up, f))
    R = np.column_stack([f, left, up])
    roll, pitch, yaw = rot_to_rpy(R)
    fovy = math.radians(cam["fovy_deg"])
    fovx = 2.0 * math.atan(math.tan(fovy / 2.0) * IMG_W / IMG_H)
    return {
        "pos": cam["pos"],
        "rpy": [roll, pitch, yaw],
        "fovx": fovx,
        "fovy": fovy,
        "parent": cam["parent"],
    }


def fmt(v: float, nd: int = 5) -> str:
    return f"{v:.{nd}f}".rstrip("0").rstrip(".")


def write_xacro(spec, cams: dict[str, dict]) -> None:
    a1 = float(spec.links[2].pos[0])
    a2 = float(spec.links[3].pos[0])
    d1 = float(spec.links[1].pos[2])
    stroke = float(spec.joints[2].upper)
    l1, l2 = spec.links[1], spec.links[2]
    l3, l4 = spec.links[3], spec.links[4]
    lines = [
        "<?xml version=\"1.0\"?>",
        "<!-- 由 ros2_ws/tools/gen_ros_assets.py 自动生成，请勿手改。",
        "     数据源：python/armlab/spec.py 与 scara_scene.py（MuJoCo 场景）。-->",
        "<robot xmlns:xacro=\"http://www.ros.org/wiki/xacro\">",
        f"  <xacro:property name=\"a1\" value=\"{fmt(a1)}\"/>",
        f"  <xacro:property name=\"a2\" value=\"{fmt(a2)}\"/>",
        f"  <xacro:property name=\"d1\" value=\"{fmt(d1)}\"/>",
        f"  <xacro:property name=\"stroke\" value=\"{fmt(stroke)}\"/>",
        f"  <xacro:property name=\"m1\" value=\"{fmt(l1.mass)}\"/>",
        f"  <xacro:property name=\"com1\" value=\"{fmt(l1.com[0])}\"/>",
        f"  <xacro:property name=\"I1x\" value=\"{fmt(l1.inertia[0][0], 9)}\"/>",
        f"  <xacro:property name=\"I1yz\" value=\"{fmt(l1.inertia[1][1], 9)}\"/>",
        f"  <xacro:property name=\"m2\" value=\"{fmt(l2.mass)}\"/>",
        f"  <xacro:property name=\"com2\" value=\"{fmt(l2.com[0])}\"/>",
        f"  <xacro:property name=\"I2x\" value=\"{fmt(l2.inertia[0][0], 9)}\"/>",
        f"  <xacro:property name=\"I2yz\" value=\"{fmt(l2.inertia[1][1], 9)}\"/>",
        f"  <xacro:property name=\"m3\" value=\"{fmt(l3.mass)}\"/>",
        f"  <xacro:property name=\"m4\" value=\"{fmt(max(l4.mass, 0.3))}\"/>",
        f"  <xacro:property name=\"grip_open\" value=\"{fmt(SC.GRIP_OPEN)}\"/>",
        f"  <xacro:property name=\"grip_gap_closed\" value=\"{fmt(SC.GRIP_GAP_CLOSED)}\"/>",
        f"  <xacro:property name=\"finger_span_top\" value=\"{fmt(SC.FINGER_SPAN_Z[0])}\"/>",
        f"  <xacro:property name=\"finger_span_bottom\" value=\"{fmt(SC.FINGER_SPAN_Z[1])}\"/>",
        f"  <xacro:property name=\"img_w\" value=\"{IMG_W}\"/>",
        f"  <xacro:property name=\"img_h\" value=\"{IMG_H}\"/>",
    ]
    for name, cam in cams.items():
        p = camera_pose(cam)
        key = name.replace("cam_", "")
        px, py, pz = p["pos"]
        rr, pp, yy = p["rpy"]
        lines += [
            f"  <xacro:property name=\"cam_{key}_x\" value=\"{fmt(px)}\"/>",
            f"  <xacro:property name=\"cam_{key}_y\" value=\"{fmt(py)}\"/>",
            f"  <xacro:property name=\"cam_{key}_z\" value=\"{fmt(pz)}\"/>",
            f"  <xacro:property name=\"cam_{key}_roll\" value=\"{fmt(rr, 6)}\"/>",
            f"  <xacro:property name=\"cam_{key}_pitch\" value=\"{fmt(pp, 6)}\"/>",
            f"  <xacro:property name=\"cam_{key}_yaw\" value=\"{fmt(yy, 6)}\"/>",
            f"  <xacro:property name=\"cam_{key}_fovx\" value=\"{fmt(p['fovx'], 6)}\"/>",
        ]
    lines.append("</robot>")
    OUT_XACRO.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] {OUT_XACRO.relative_to(ROOT)}")


def camera_sensor_xml(name: str, cam: dict, pose_xyz=(0, 0, 0), pose_rpy=(0, 0, 0),
                      indent: int = 4) -> str:
    p = camera_pose(cam)
    pad = " " * indent
    return "\n".join([
        f'{pad}<sensor name="{name}" type="camera">',
        f'{pad}  <pose>{" ".join(fmt(v) for v in pose_xyz)} '
        f'{" ".join(fmt(v) for v in pose_rpy)}</pose>',
        f'{pad}  <camera>',
        f'{pad}    <horizontal_fov>{fmt(p["fovx"], 6)}</horizontal_fov>',
        f'{pad}    <image><width>{640 if name == "cam_top" else IMG_W}</width>'
        f'<height>{480 if name == "cam_top" else IMG_H}</height></image>',
        f'{pad}    <clip><near>0.02</near><far>8.0</far></clip>',
        f'{pad}  </camera>',
        f'{pad}  <always_on>1</always_on>',
        f'{pad}  <update_rate>10</update_rate>',
        f'{pad}  <visualize>false</visualize>',
        f'{pad}  <plugin filename="libgazebo_ros_camera.so" name="{name}_driver">',
        f'{pad}    <ros>',
        f'{pad}      <image_topic_name>image_raw</image_topic_name>',
        f'{pad}      <camera_info_topic_name>camera_info</camera_info_topic_name>',
        f'{pad}    </ros>',
        f'{pad}    <frame_name>{name}</frame_name>',
        f'{pad}  </plugin>',
        f'{pad}</sensor>',
    ])


def write_world(cams: dict[str, dict]) -> None:
    cx, cy = SC.TABLE_CENTER
    hx, hy = SC.TABLE_HALF
    thick = SC.TABLE_THICK
    top = SC.TABLE_TOP
    leg = SC.LEG_HALF
    leg_h = top - thick
    obj = SC.OBJ_SIZE / 2
    px, py = 0.30, 0.12
    psx, psy = 0.025, 0.025

    legs = []
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        lx = cx + sx * (hx - 0.03)
        ly = cy + sy * (hy - 0.03)
        legs.append(f"""
      <link name="leg{i}">
        <pose>{fmt(lx)} {fmt(ly)} {fmt(leg_h / 2)} 0 0 0</pose>
        <collision name="leg{i}_c"><geometry><box><size>{fmt(2*leg)} {fmt(2*leg)} {fmt(leg_h)}</size></box></geometry></collision>
        <visual name="leg{i}_v"><geometry><box><size>{fmt(2*leg)} {fmt(2*leg)} {fmt(leg_h)}</size></box></geometry>
          <material><ambient>0.55 0.42 0.30 1</ambient><diffuse>0.62 0.48 0.34 1</diffuse></material></visual>
      </link>""")

    # SDF 要求 sensor 位于 link 内：每个相机一个 link，link 的位姿就是相机位姿
    cam_models = []
    for name in ("cam_top", "cam_front"):
        if name not in cams:
            continue
        p = camera_pose(cams[name])
        sensor = camera_sensor_xml(name, cams[name], indent=8)
        cam_models.append(f"""
      <link name="{name}_link">
        <pose>{fmt(p['pos'][0])} {fmt(p['pos'][1])} {fmt(p['pos'][2])} {fmt(p['rpy'][0], 6)} {fmt(p['rpy'][1], 6)} {fmt(p['rpy'][2], 6)}</pose>
{sensor}
      </link>""")

    sdf = f"""<?xml version="1.0" ?>
<!-- 由 ros2_ws/tools/gen_ros_assets.py 从 MuJoCo 场景生成，请勿手改。
     台面中心 ({fmt(cx)}, {fmt(cy)})，{fmt(2*hx)} x {fmt(2*hy)} m，台面高 {fmt(top)} m；
     物块 {fmt(2*obj)} mm；相机位置/视场角与 scara_scene.py 完全一致。 -->
<sdf version="1.6">
  <world name="scara_world">
    <scene>
      <ambient>0.30 0.32 0.36 1</ambient>
      <background>0.78 0.85 0.95 1</background>
      <shadows>true</shadows>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 3 0 0 0</pose>
      <diffuse>0.55 0.55 0.52 1</diffuse>
      <specular>0.05 0.05 0.05 1</specular>
      <direction>-0.4 0.3 -1</direction>
    </light>

    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
      <real_time_factor>1.0</real_time_factor>
      <ode>
        <solver><type>quick</type><iters>50</iters><sor>1.2</sor></solver>
        <constraints>
          <cfm>0.0</cfm><erp>0.2</erp>
          <contact_max_correcting_vel>0.1</contact_max_correcting_vel>
          <contact_surface_layer>0.0005</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    <gravity>0 0 -9.81</gravity>

    <include><uri>model://ground_plane</uri></include>

    <!-- 工作台（与 MuJoCo 同参数） -->
    <model name="table">
      <static>true</static>
      <link name="top">
        <pose>{fmt(cx)} {fmt(cy)} {fmt(top - thick / 2)} 0 0 0</pose>
        <collision name="top_c">
          <geometry><box><size>{fmt(2*hx)} {fmt(2*hy)} {fmt(thick)}</size></box></geometry>
          <surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface>
        </collision>
        <visual name="top_v">
          <geometry><box><size>{fmt(2*hx)} {fmt(2*hy)} {fmt(thick)}</size></box></geometry>
          <material><ambient>0.78 0.66 0.48 1</ambient><diffuse>0.86 0.74 0.56 1</diffuse></material>
        </visual>
      </link>{"".join(legs)}
    </model>

    <!-- 放置区（与 MuJoCo place_zone 同尺寸） -->
    <model name="place_area">
      <static>true</static>
      <link name="link">
        <pose>{fmt(px)} {fmt(py)} {fmt(top + 0.0016)} 0 0 0</pose>
        <visual name="v">
          <geometry><box><size>{fmt(2*psx)} {fmt(2*psy)} 0.001</size></box></geometry>
          <material><ambient>0.15 0.75 0.25 1</ambient><diffuse>0.20 0.85 0.30 1</diffuse></material>
        </visual>
      </link>
    </model>

    <!-- 物块（30 mm / 50 g） -->
    <model name="target_box">
      <pose>0.30 -0.05 {fmt(top + obj)} 0 0 0</pose>
      <link name="link">
        <inertial>
          <mass>0.05</mass>
          <inertia><ixx>7.5e-6</ixx><iyy>7.5e-6</iyy><izz>7.5e-6</izz>
            <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="c">
          <geometry><box><size>{fmt(2*obj)} {fmt(2*obj)} {fmt(2*obj)}</size></box></geometry>
          <surface><friction><ode><mu>1.2</mu><mu2>1.2</mu2></ode></friction>
            <bounce><restitution_coefficient>0.0</restitution_coefficient></bounce></surface>
        </collision>
        <visual name="v">
          <geometry><box><size>{fmt(2*obj)} {fmt(2*obj)} {fmt(2*obj)}</size></box></geometry>
          <material><ambient>0.95 0.80 0.10 1</ambient><diffuse>1.00 0.86 0.15 1</diffuse></material>
        </visual>
      </link>
    </model>

    <!-- 世界相机（cam_top / cam_front，位姿与 MuJoCo 一致） -->
    <model name="world_cameras">
      <static>true</static>{"".join(cam_models)}
    </model>

    <!-- 实体状态服务：/get_entity_state（用于验收物块最终位姿） -->
    <plugin filename="libgazebo_ros_state.so" name="gazebo_ros_state"/>
  </world>
</sdf>
"""
    OUT_WORLD.write_text(sdf, encoding="utf-8")
    print(f"[ok] {OUT_WORLD.relative_to(ROOT)}")


def main() -> None:
    spec = S.scara_teach_4rprr()
    mjcf = SC.build_scara_pick_place_xml(spec)
    cams = parse_cameras(mjcf)
    write_xacro(spec, cams)
    write_world(cams)
    report = {
        "spec": {
            "a1": float(spec.links[2].pos[0]), "a2": float(spec.links[3].pos[0]),
            "d1": float(spec.links[1].pos[2]), "stroke": float(spec.joints[2].upper),
            "m1": spec.links[1].mass, "m2": spec.links[2].mass,
        },
        "table": {"center": SC.TABLE_CENTER, "half": SC.TABLE_HALF,
                  "top": SC.TABLE_TOP, "thick": SC.TABLE_THICK, "leg_half": SC.LEG_HALF},
        "object": {"size": SC.OBJ_SIZE},
        "cameras": {n: camera_pose(c) | {"fovy_deg": cams[n]["fovy_deg"]}
                    for n, c in cams.items()},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] {OUT_JSON.relative_to(ROOT)}")
    print("\n相机参数（由 MuJoCo xyaxes 换算为 Gazebo 位姿）:")
    for n, c in cams.items():
        p = camera_pose(c)
        print(f"  {n:10s} parent={c['parent']:6s} pos={[round(v,3) for v in p['pos']]} "
              f"rpy={[round(v,3) for v in p['rpy']]} fovx={math.degrees(p['fovx']):.1f}°")


if __name__ == "__main__":
    main()
