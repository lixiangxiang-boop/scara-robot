"""Gazebo Classic + ros2_control 联调启动。

启动顺序（每一步都不可省）：
  1) gazebo（gzserver/gzclient）+ 世界文件
  2) robot_state_publisher（发 robot_description + TF）
  3) spawn_entity（把机器人从 /robot_description 生成到 Gazebo）
  4) controller_manager spawner：joint_state_broadcaster → scara_controller → gripper_controller
     （必须等机器人生成完成，否则 ros2_control 的硬件接口还没注册）

运行：ros2 launch scara_bringup gazebo.launch.py [gui:=false]
"""

import os
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

# 每个连杆的渲染颜色（URDF->SDF 会丢材质，这里在转换后的 SDF 上注入）
LINK_COLORS = {
    "base_link": "0.32 0.33 0.37 1",
    "link1": "0.20 0.55 0.95 1",
    "link2": "0.25 0.80 0.40 1",
    "link3": "1.00 0.62 0.15 1",
    "link4": "0.85 0.25 0.25 1",
    "gripper_left": "0.55 0.57 0.60 1",
    "gripper_right": "0.55 0.57 0.60 1",
    "cam_eye_link": "0.25 0.25 0.28 1",
    "cam_wrist_link": "0.25 0.25 0.28 1",
}


def make_colored_sdf(urdf_path: str) -> str:
    """URDF -> SDF，并给每个连杆 visual 注入材质（解决 Gazebo 白模）。"""
    sdf_text = subprocess.check_output(["gz", "sdf", "-p", urdf_path],
                                       text=True, stderr=subprocess.DEVNULL)
    root = ET.fromstring(sdf_text)
    for link in root.iter("link"):
        rgba = LINK_COLORS.get(link.get("name", ""), "0.45 0.45 0.48 1")
        for visual in link.findall("visual"):
            mat = ET.SubElement(visual, "material")
            ET.SubElement(mat, "ambient").text = rgba
            ET.SubElement(mat, "diffuse").text = rgba
            ET.SubElement(mat, "specular").text = "0.08 0.08 0.08 1"
    out = os.path.join(tempfile.gettempdir(), "scara_colored.sdf")
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=False)
    return out


def generate_launch_description() -> LaunchDescription:
    gui = LaunchConfiguration("gui")
    bringup_pkg = FindPackageShare("scara_bringup")
    desc_pkg = FindPackageShare("scara_description")
    gazebo_pkg = FindPackageShare("gazebo_ros")

    # 世界文件由 ros2_ws/tools/gen_ros_assets.py 从 MuJoCo 场景生成（台面/四腿/物块/相机）
    world = PathJoinSubstitution([bringup_pkg, "worlds", "scara_world.generated.sdf"])

    # 关键一步：gazebo_ros2_control 会把 robot_description 作为 "--param name:=value"
    # 传给 controller_manager；rcl 的参数解析不接受多行值（会报
    # "Couldn't parse parameter override rule"）。这里先 xacro 展开，去掉注释并压成
    # 单行，再发布到 /robot_description —— 这是本环境（Humble + gazebo_ros2_control
    # 0.4.10）能稳定起控制器的写法，详见 ROS2 手册 7.3 节。
    xacro_path = os.path.join(get_package_share_directory("scara_description"),
                              "urdf", "scara.urdf.xacro")
    urdf_xml = subprocess.check_output(["xacro", xacro_path], text=True)
    urdf_xml = re.sub(r"<!--.*?-->", "", urdf_xml, flags=re.DOTALL)
    urdf_one_line = " ".join(urdf_xml.split())
    robot_description = ParameterValue(urdf_one_line, value_type=str)
    # 带颜色的 SDF 供 spawn 使用（RSP 仍用 URDF，TF 不变）
    urdf_file = os.path.join(tempfile.gettempdir(), "scara_robot.urdf")
    with open(urdf_file, "w", encoding="utf-8") as f:
        f.write(urdf_xml)
    colored_sdf = make_colored_sdf(urdf_file)

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([gazebo_pkg, "/launch/gazebo.launch.py"]),
        launch_arguments={"world": world, "gui": gui, "verbose": "false"}.items(),
    )
    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher",
               parameters=[{"robot_description": robot_description, "use_sim_time": True}],
               output="screen")
    spawn = Node(package="gazebo_ros", executable="spawn_entity.py",
                 arguments=["-file", colored_sdf, "-entity", "scara",
                            "-x", "0", "-y", "0", "-z", "0", "-timeout", "90"],
                 output="screen")

    def spawner(name: str) -> Node:
        return Node(package="controller_manager", executable="spawner",
                    arguments=[name, "--controller-manager", "/controller_manager"],
                    output="screen")

    after_spawn = RegisterEventHandler(OnProcessExit(
        target_action=spawn,
        on_exit=[TimerAction(period=2.0,
                             actions=[spawner("joint_state_broadcaster"),
                                      spawner("scara_controller")])],
    ))

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true",
                              description="Gazebo 客户端（可视化）开关"),
        # WSLg 默认把 OpenGL 绑到核显；这个变量让 Mesa 的 D3D12 驱动选 RTX 4060
        SetEnvironmentVariable("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA"),
        gz,
        rsp,
        spawn,
        after_spawn,
    ])
