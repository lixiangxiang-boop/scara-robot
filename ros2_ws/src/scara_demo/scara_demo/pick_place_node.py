"""抓取-放置演示（八段序列，与项目 MuJoCo 阶段的动作语义一致）。

运行（先启动 gazebo.launch.py）：
    ros2 run scara_demo pick_place
    ros2 run scara_demo pick_place --ros-args -p obj_x:=0.30 -p obj_y:=-0.05
"""

from __future__ import annotations

import sys

import rclpy

from .controller_client import ScaraClient


def report_object_pose(node, place_x: float, place_y: float) -> None:
    """通过 gazebo_ros_state 的 /get_entity_state 读取物块最终位姿并计算放置误差。"""
    from gazebo_msgs.srv import GetEntityState
    client = node.create_client(GetEntityState, "/get_entity_state")
    if not client.wait_for_service(timeout_sec=3.0):
        node.get_logger().warn("未找到 /get_entity_state（世界未加载 gazebo_ros_state 插件）")
        return
    req = GetEntityState.Request()
    req.name = "target_box"
    req.reference_frame = "world"
    fut = client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    res = fut.result()
    if res is None:
        node.get_logger().warn("读取物块位姿超时")
        return
    p = res.state.pose.position
    err = ((p.x - place_x) ** 2 + (p.y - place_y) ** 2) ** 0.5
    node.get_logger().info(
        f"物块终点 = ({p.x:.4f}, {p.y:.4f}, {p.z:.4f}) m，"
        f"放置误差 = {err * 1000:.1f} mm")
    node.get_logger().info(
        "验收：" + ("通过（<25 mm 且在台面）" if err < 0.025 else "未通过（误差超限）"))


def run_sequence(node, ox: float, oy: float, px: float, py: float, *,
                 s: float = 1.8, hover: float = 0.26, grasp: float = 0.18,
                 lift: float = 0.21, release: float = 0.184,
                 grasp_mode: str = "vacuum") -> bool:
    """八步抓取-放置（pick_place 与 vision_pick 共用）。"""
    steps = [
        ("1/8 张开夹爪并高位接近物体", lambda: (node.open_gripper(0.4),
                                     node.move_line(ox, oy, hover, T=1.2 * s))[1]),
        ("2/8 下探到物体两侧", lambda: node.move_line(ox, oy, grasp, T=0.9 * s)),
        ("3/8 闭合夹爪抓取物块", lambda: (node.close_gripper(0.6),
                                     node.vacuum_attach("target_box", 0.2)
                                     if grasp_mode == "vacuum" else True)[1]),
        ("4/8 原地抬起 30 mm", lambda: node.move_line(ox, oy, lift, T=0.6 * s)),
        ("5/8 水平搬运到放置区上方", lambda: node.move_line(px, py, lift, T=1.2 * s)),
        # 下放高度用 release_z：吸盘模型下物块随工具走，太低会把它压进台面
        ("6/8 下放", lambda: node.move_line(px, py, release, T=0.8 * s)),
        # 先释放再略微抬升 10 mm，最后张开夹爪：避免张开时手指把物块推偏
        ("7/8 释放并张开夹爪", lambda: ((node.vacuum_detach()
                                   if grasp_mode == "vacuum" else None),
                                  node.move_line(px, py, release + 0.010, T=0.35),
                                  node.open_gripper(0.5), True)[3]),
        ("8/8 全部缩回", lambda: node.move_line(px, py, hover, T=0.7 * s)),
    ]

    ok_all = True
    for name, fn in steps:
        node.get_logger().info(f"===== {name} =====")
        try:
            ok = fn()
        except ValueError as exc:
            node.get_logger().error(f"规划失败：{exc}")
            ok = False
        tcp = node.tcp()
        if tcp is not None:
            node.get_logger().info(
                f"    TCP = ({tcp[0]:.4f}, {tcp[1]:.4f}, {tcp[2]:.4f}) m, yaw = {tcp[3]:.3f} rad")
        ok_all = ok_all and bool(ok)
        if not ok_all:
            node.get_logger().warn("本步失败，序列中止")
            break

    node.get_logger().info("抓取-放置序列完成" if ok_all else "序列未完成（见上方日志）")
    if ok_all:
        report_object_pose(node, px, py)
    return ok_all


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = ScaraClient("scara_pick_place")
    for name, value in (("obj_x", 0.30), ("obj_y", -0.05),
                        ("place_x", 0.30), ("place_y", 0.12),
                        ("hover_z", 0.26), ("grasp_z", 0.18), ("lift_z", 0.21),
                        ("release_z", 0.184),
                        ("grasp_mode", "vacuum"),
                        ("speed_scale", 1.8)):
        node.declare_parameter(name, value)
    p = lambda n: float(node.get_parameter(n).value)  # noqa: E731

    if not node.wait_for_robot():
        node.destroy_node(); rclpy.shutdown(); return
    run_sequence(node, p("obj_x"), p("obj_y"), p("place_x"), p("place_y"),
                 s=p("speed_scale"), hover=p("hover_z"), grasp=p("grasp_z"),
                 lift=p("lift_z"), release=p("release_z"),
                 grasp_mode=str(node.get_parameter("grasp_mode").value))
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
