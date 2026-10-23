"""MoveIt2 move_group（接 scara_controller 执行）。

运行（先 gazebo.launch.py）：
    ros2 launch scara_bringup move_group.launch.py
    ros2 run scara_demo moveit_plan --q 0.6 -1.2 0.04 0.6
"""

from __future__ import annotations

import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_launch_description() -> LaunchDescription:
    desc = get_package_share_directory("scara_description")
    cfg = get_package_share_directory("scara_bringup")
    xacro_path = os.path.join(desc, "urdf", "scara.urdf.xacro")
    robot_description = ParameterValue(Command(["xacro ", xacro_path]), value_type=str)
    with open(os.path.join(cfg, "config", "scara.srdf"), "r", encoding="utf-8") as f:
        srdf = f.read()

    params = {
        "robot_description": robot_description,
        "robot_description_semantic": srdf,
        "robot_description_kinematics": load_yaml(os.path.join(cfg, "config", "kinematics.yaml")),
        "robot_description_planning": load_yaml(os.path.join(cfg, "config", "joint_limits.yaml")),
        "use_sim_time": True,
    }
    params.update(load_yaml(os.path.join(cfg, "config", "moveit_controllers.yaml")))
    # OMPL 参数必须嵌在 pipeline 名（ompl）下，平铺会被忽略并退回 CHOMP
    params["ompl"] = load_yaml(os.path.join(cfg, "config", "ompl_planning.yaml"))
    params["planning_pipelines"] = ["ompl"]

    return LaunchDescription([
        Node(package="moveit_ros_move_group", executable="move_group",
             output="screen", parameters=[params]),
    ])
