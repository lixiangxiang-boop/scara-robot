"""订阅一帧相机图像并存成 PPM（无需额外图像库），用于联调留证/手册插图。

运行：ros2 run scara_demo capture --topic /cam_top/image_raw --out /tmp/cam_top.ppm
"""

from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class Capture(Node):
    def __init__(self, topic: str, out: str) -> None:
        super().__init__("scara_capture")
        self.out = out
        self.done = False
        self.create_subscription(Image, topic, self._on_image, 1)
        self.get_logger().info(f"等待图像：{topic}")

    def _on_image(self, msg: Image) -> None:
        if self.done:
            return
        h, w = msg.height, msg.width
        data = bytes(msg.data)
        enc = msg.encoding.lower()
        if enc == "rgb8":
            buf = data
        elif enc == "bgr8":
            buf = bytearray(len(data))
            buf[0::3] = data[2::3]
            buf[1::3] = data[1::3]
            buf[2::3] = data[0::3]
            buf = bytes(buf)
        else:
            self.get_logger().warn(f"暂不支持编码 {msg.encoding}")
            self.done = True
            return
        with open(self.out, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
            f.write(buf)
        self.get_logger().info(f"已保存 {w}x{h} {msg.encoding} → {self.out}")
        self.done = True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/cam_top/image_raw")
    ap.add_argument("--out", default="/tmp/cam_top.ppm")
    ap.add_argument("--timeout", type=float, default=10.0)
    args, ros_args = ap.parse_known_args(argv)
    rclpy.init(args=[sys.argv[0]] + list(ros_args))
    node = Capture(args.topic, args.out)
    import time
    t0 = time.time()
    while rclpy.ok() and not node.done and time.time() - t0 < args.timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
