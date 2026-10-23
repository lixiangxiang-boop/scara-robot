"""视觉闭环抓取：cam_top 图像 → 颜色分割 → 反投影到台面 → 抓取。

原理（眼在手外，eye-to-hand）：
  1) 像素 (u,v) 与相机内参 fx,fy,cx,cy 得到光学坐标系射线
        r_cam = ((u-cx)/fx, (v-cy)/fy, 1)
  2) 相机位姿已知（与世界文件同源的 rpy），光学轴 x右/y下/z前：
        R_opt = R_sensor · R_conv,  R_conv 列 = [(0,-1,0), (0,0,-1), (1,0,0)]
  3) 与台面平面 z = z_table 求交：
        t = (z_table - cam_z) / (R_opt r_cam)_z,   p = cam_pos + t·R_opt r_cam

验证：把检测结果与 Gazebo 真值 /get_entity_state 对比（打印毫米级误差）。

运行（先 gazebo.launch.py）：
    ros2 run scara_demo vision_pick            # 只检测并打印
    ros2 run scara_demo vision_pick --pick     # 检测后直接抓取
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import rclpy
from gazebo_msgs.srv import GetEntityState
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from .controller_client import ScaraClient
from .pick_place_node import run_sequence

# cam_top 的位姿与视场角（与 MuJoCo 场景同源，见 docs/manual/ROS2 手册 2.3 节）
CAM_TOP = {"pos": (0.40, 0.0, 1.25), "rpy": (0.0, 1.5707963, 1.5707963)}


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class VisionPick(ScaraClient):
    def __init__(self, name: str = "scara_vision_pick") -> None:
        super().__init__(name)
        self.rgb = None
        self.info = None
        self.create_subscription(Image, "/cam_top/image_raw", self._on_image, 10)
        self.create_subscription(CameraInfo, "/cam_top/camera_info", self._on_info, 10)

    def _on_info(self, msg: CameraInfo) -> None:
        self.info = msg

    def _on_image(self, msg: Image) -> None:
        h, w = msg.height, msg.width
        data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        if msg.encoding.lower() == "rgb8":
            img = data.reshape(h, w, 3)
        elif msg.encoding.lower() == "bgr8":
            img = data.reshape(h, w, 3)[:, :, ::-1]
        else:
            return
        self.rgb = img

    def detect(self, timeout: float = 10.0):
        """返回 (u, v, area, n_pixels) 或 None。黄色物块：R/G 高、B 低。"""
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.rgb is None or self.info is None:
                continue
            img = self.rgb.astype(np.int16)
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            mask = (r > 170) & (g > 140) & (b < 70) & (np.abs(r - g) < 50)
            n = int(mask.sum())
            if n < 80:
                continue
            vs, us = np.nonzero(mask)
            return float(us.mean()), float(vs.mean()), float(n), n
        return None

    def pixel_to_table(self, u: float, v: float, z_table: float):
        fx, fy = self.info.k[0], self.info.k[4]
        cx, cy = self.info.k[2], self.info.k[5]
        r_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        R_sensor = rpy_matrix(*CAM_TOP["rpy"])
        R_conv = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        R_opt = R_sensor @ R_conv
        d = R_opt @ r_cam
        cam = np.array(CAM_TOP["pos"])
        t = (z_table - cam[2]) / d[2]
        return cam + t * d, (fx, fy, cx, cy)

    def ground_truth(self, name: str = "target_box"):
        from gazebo_msgs.srv import GetEntityState
        cli = self._svc(GetEntityState, "/get_entity_state")
        if cli is None:
            return None
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = "world"
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        res = fut.result()
        if res is None or not res.success:
            return None
        p = res.state.pose.position
        return p.x, p.y, p.z


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", action="store_true", help="检测后直接执行抓取-放置")
    ap.add_argument("--place-x", type=float, default=0.30)
    ap.add_argument("--place-y", type=float, default=0.12)
    ap.add_argument("--z-table", type=float, default=0.115, help="投影平面高度（物块中心）")
    ap.add_argument("--grasp-mode", default="vacuum")
    args, ros_args = ap.parse_known_args(argv)

    rclpy.init(args=[sys.argv[0]] + list(ros_args))
    node = VisionPick()
    if not node.wait_for_robot():
        node.destroy_node(); rclpy.shutdown(); return

    det = node.detect()
    if det is None:
        node.get_logger().error("没有检测到黄色物块（检查 cam_top 图像）")
        node.destroy_node(); rclpy.shutdown(); return
    u, v, area, n = det
    p, intr = node.pixel_to_table(u, v, args.z_table)
    node.get_logger().info(
        f"像素中心 = ({u:.1f}, {v:.1f})，面积 {area:.0f} px，内参 fx={intr[0]:.1f} "
        f"fy={intr[1]:.1f} cx={intr[2]:.1f} cy={intr[3]:.1f}")
    node.get_logger().info(f"反投影到 z={args.z_table:.3f} 平面：({p[0]:.4f}, {p[1]:.4f}) m")
    gt = node.ground_truth()
    if gt is not None:
        err = math.hypot(p[0] - gt[0], p[1] - gt[1])
        node.get_logger().info(
            f"Gazebo 真值 = ({gt[0]:.4f}, {gt[1]:.4f})，视觉定位误差 = {err * 1000:.1f} mm")
    if args.pick:
        run_sequence(node, float(p[0]), float(p[1]), args.place_x, args.place_y,
                     grasp_mode=args.grasp_mode)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
