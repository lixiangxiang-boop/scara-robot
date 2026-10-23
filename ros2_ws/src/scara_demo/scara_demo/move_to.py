"""把末端沿直线移动到指定位姿（MoveL）。

运行：ros2 run scara_demo move_to --x 0.30 --y -0.05 --z 0.20 [--yaw 0.0] [--time 1.0]
"""

from __future__ import annotations

import argparse
import sys

import rclpy

from .controller_client import ScaraClient


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--time", type=float, default=1.0)
    args, ros_args = parser.parse_known_args(argv)

    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = ScaraClient("scara_move_to")
    if not node.wait_for_robot():
        node.destroy_node(); rclpy.shutdown(); return
    ok = node.move_line(args.x, args.y, args.z, yaw=args.yaw, T=args.time)
    tcp = node.tcp()
    if tcp is not None:
        node.get_logger().info(
            f"到位：TCP = ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f})，yaw={tcp[3]:.3f}")
    node.get_logger().info("成功" if ok else "失败")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
