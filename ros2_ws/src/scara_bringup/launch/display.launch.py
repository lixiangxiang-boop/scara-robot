"""RViz 显示（不启动物理仿真）：robot_state_publisher + joint_state_publisher_gui + rviz2。

运行：ros2 launch scara_bringup display.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_gui = LaunchConfiguration("gui")
    desc_pkg = FindPackageShare("scara_description")
    xacro_file = PathJoinSubstitution([desc_pkg, "urdf", "scara.urdf.xacro"])
    rviz_cfg = PathJoinSubstitution([desc_pkg, "rviz", "scara.rviz"])

    robot_description = ParameterValue(Command(["xacro ", xacro_file]), value_type=str)

    return LaunchDescription([
        SetEnvironmentVariable("MESA_D3D12_DEFAULT_ADAPTER_NAME", "NVIDIA"),
        DeclareLaunchArgument("gui", default_value="true",
                              description="是否打开 joint_state_publisher_gui 滑块"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}],
             output="screen"),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui",
             condition=IfCondition(use_gui), output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", rviz_cfg],
             output="screen"),
    ])
