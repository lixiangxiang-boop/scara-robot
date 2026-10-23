"""联调自检：把"解析正运动学"与"ROS TF（world→link4）"逐位姿对照。

原理：URDF 的 link4 原点就是项目定义的 TCP。
  解析 FK：x = a1 cos q1 + a2 cos(q1+q2)，y = a1 sin q1 + ...，z = d1 - q3
  TF：world → link4（由 robot_state_publisher 根据 /joint_states 计算）
若两者一致（<1 mm），说明"参数、URDF、ros2_control、TF"整条链没有偏差。

运行（gazebo.launch.py 启动后）：ros2 run scara_demo check_tf
"""

from __future__ import annotations

import math

import rclpy
import tf2_ros
from rclpy.time import Time

from . import kinematics as K
from .controller_client import ScaraClient

TEST_CONFIGS = [
    [0.6, -1.2, 0.04, 0.6],
    [-0.8, -1.6, 0.10, -0.2],
    [1.2, -0.9, 0.02, 1.0],
]


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = ScaraClient("scara_check_tf")
    buffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(buffer, node)

    if not node.wait_for_robot():
        node.destroy_node(); rclpy.shutdown(); return

    print("=" * 74)
    print(f"{'目标关节角':<34}{'|FK-TF| 位置误差':>18}{'yaw 误差':>14}")
    print("=" * 74)
    worst = 0.0
    for q_ref in TEST_CONFIGS:
        if not node.move_joints(q_ref, T=2.2):
            print("  运动失败，跳过")
            continue
        for _ in range(30):                     # 等 TF 更新
            rclpy.spin_once(node, timeout_sec=0.02)
        q = node.q()
        if q is None:
            continue
        fx, fy, fz, fyaw = K.fk(q)
        try:
            tr = buffer.lookup_transform("world", "link4", Time())
        except Exception as exc:                # noqa: BLE001
            print(f"  TF 查询失败：{exc}")
            continue
        t = tr.transform.translation
        r = tr.transform.rotation
        tx_yaw = math.atan2(2.0 * (r.w * r.z + r.x * r.y),
                            1.0 - 2.0 * (r.y * r.y + r.z * r.z))
        derr = math.sqrt((fx - t.x) ** 2 + (fy - t.y) ** 2 + (fz - t.z) ** 2)
        yerr = abs(math.atan2(math.sin(fyaw - tx_yaw), math.cos(fyaw - tx_yaw)))
        worst = max(worst, derr)
        print(f"{str([round(v, 3) for v in q]):<34}{derr * 1000:>14.3f} mm{yerr:>14.2e} rad")
    print("=" * 74)
    print(f"最大位置偏差 = {worst * 1000:.3f} mm  ->  "
          f"{'通过（<1 mm）' if worst < 1e-3 else '超差，检查 URDF 参数'}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
