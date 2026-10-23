# ROS2 + Gazebo 工作空间（SCARA-teach-4RPRR）

## 目录

| 包 | 作用 |
| --- | --- |
| `scara_description` | URDF/xacro（几何、惯量、ros2_control 接口）、RViz 配置 |
| `scara_bringup` | Gazebo 世界、`controllers.yaml`、`display` / `gazebo` 两个 launch |
| `scara_demo` | 运动学/轨迹（MoveL、五次多项式）、抓取演示、TF 自检 |

## 一键构建（Windows 侧）

```powershell
powershell -ExecutionPolicy Bypass -File ros2_ws\sync_build.ps1
```

等价于在 WSL 里：`bash /mnt/c/.../ros2_ws/build.sh`（源文件复制到 `~/scara_ros2_ws/src` 后在 ext4 上编译，避免 /mnt/c 慢）。

## 运行

```bash
source /opt/ros/humble/setup.bash
source ~/scara_ros2_ws/install/setup.bash

ros2 launch scara_bringup gazebo.launch.py          # Gazebo + 控制器
ros2 run scara_demo pick_place                      # 抓取-放置演示
ros2 run scara_demo move_to --x 0.30 --y -0.05 --z 0.20   # 单点 MoveL
ros2 run scara_demo check_tf                        # 解析 FK vs TF 对照
ros2 launch scara_bringup display.launch.py         # 只开 RViz（滑块驱动）
```

## 主要接口

- 关节：`joint1..joint4`（q1,q2,q3,q4；`joint3` 为移动副，轴 -z，行程 0..0.2 m）
- 夹爪：`gripper_left_joint` / `gripper_right_joint`（0..0.012 m，两侧同步）
- 控制器：`/scara_controller/follow_joint_trajectory`（FollowJointTrajectory action）
- 夹爪指令：`/gripper_controller/commands`（Float64MultiArray）
- 状态：`/joint_states`（joint_state_broadcaster，1 kHz）
- TF：`world → base_link → link1 → link2 → link3 → link4`
