"""监视物块位姿：每 50 ms 查询 /get_entity_state，位置变化 > 5 mm 就打印一行。

用途：定位抓取序列里"哪一步把物块碰跑了"。
运行：ros2 run scara_demo watch_object [--name target_box] [--seconds 30]
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from gazebo_msgs.srv import GetEntityState
from rclpy.node import Node


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="target_box")
    ap.add_argument("--seconds", type=float, default=30.0)
    args, ros_args = ap.parse_known_args(argv)

    rclpy.init(args=[sys.argv[0]] + list(ros_args))
    node = Node("scara_watch_object")
    client = node.create_client(GetEntityState, "/get_entity_state")
    if not client.wait_for_service(timeout_sec=5.0):
        node.get_logger().error("没有 /get_entity_state 服务")
        node.destroy_node(); rclpy.shutdown(); return

    last = None
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < args.seconds:
        req = GetEntityState.Request()
        req.name = args.name
        req.reference_frame = "world"
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=1.0)
        res = fut.result()
        if res is not None and res.success:
            p = res.state.pose.position
            now = (p.x, p.y, p.z)
            if last is None:
                node.get_logger().info(
                    f"初始位姿 = ({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
            else:
                d = ((now[0] - last[0]) ** 2 + (now[1] - last[1]) ** 2
                     + (now[2] - last[2]) ** 2) ** 0.5
                if d > 0.005:
                    node.get_logger().warn(
                        f"t={time.time() - t0:5.2f}s 物块移动 {d * 1000:5.1f} mm → "
                        f"({p.x:.4f}, {p.y:.4f}, {p.z:.4f})")
            last = now
        time.sleep(0.05)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
