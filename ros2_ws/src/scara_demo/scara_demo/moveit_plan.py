"""用 MoveIt2 规划并执行关节空间目标（验证 MoveIt ↔ scara_controller 联通）。

运行（先 gazebo.launch.py + move_group.launch.py）：
    ros2 run scara_demo moveit_plan --q 0.6 -1.2 0.04 0.6
"""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState

JOINTS = ["joint1", "joint2", "joint3", "joint4"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", nargs=4, type=float, required=True)
    ap.add_argument("--group", default="arm")
    ap.add_argument("--timeout", type=float, default=120.0)
    args, ros_args = ap.parse_known_args(argv)

    rclpy.init(args=[sys.argv[0]] + list(ros_args))
    node = Node("scara_moveit_plan")
    client = ActionClient(node, MoveGroup, "/move_action")
    if not client.wait_for_server(timeout_sec=10.0):
        node.get_logger().error("没有 /move_action（先启动 move_group.launch.py）")
        node.destroy_node(); rclpy.shutdown(); return

    goal = MoveGroup.Goal()
    req = goal.request
    req.group_name = args.group
    req.num_planning_attempts = 5
    req.allowed_planning_time = 5.0
    req.max_velocity_scaling_factor = 0.3
    req.max_acceleration_scaling_factor = 0.3
    req.planner_id = "RRTConnect"
    req.pipeline_id = "ompl"
    # 用当前机器人状态作为起点（否则空的 start_state 会被判为 INVALID_ROBOT_STATE=-17）
    req.start_state.is_diff = True
    cons = Constraints()
    for name, q in zip(JOINTS, args.q):
        jc = JointConstraint()
        jc.joint_name = name
        jc.position = float(q)
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0
        cons.joint_constraints.append(jc)
    req.goal_constraints.append(cons)
    goal.planning_options.plan_only = False

    node.get_logger().info(f"MoveIt 目标：{JOINTS} = {args.q}")
    send = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send, timeout_sec=40.0)
    handle = send.result()
    if handle is None or not handle.accepted:
        node.get_logger().error("MoveGroup 目标被拒绝")
        node.destroy_node(); rclpy.shutdown(); return
    res_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_future, timeout_sec=args.timeout)
    res = res_future.result()
    if res is None:
        node.get_logger().error("等待 MoveIt 结果超时")
        node.destroy_node(); rclpy.shutdown(); return
    code = res.result.error_code.val
    npts = len(res.result.planned_trajectory.joint_trajectory.points)
    node.get_logger().info(
        f"error_code={code}（1=SUCCESS），规划点数={npts}，"
        f"执行={'完成' if code == 1 else '未完成'}")

    state = {}

    def cb(msg: JointState) -> None:
        for n, p in zip(msg.name, msg.position):
            state[n] = float(p)

    sub = node.create_subscription(JointState, "/joint_states", cb, 10)
    t0 = time.time()
    while time.time() - t0 < 2.0:
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(sub)
    if all(n in state for n in JOINTS):
        err = max(abs(state[n] - q) for n, q in zip(JOINTS, args.q))
        node.get_logger().info(
            f"到位 = {[round(state[n], 4) for n in JOINTS]}，最大误差 {err:.4f}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
