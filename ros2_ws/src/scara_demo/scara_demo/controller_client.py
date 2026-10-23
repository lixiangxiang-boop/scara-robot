"""与 ros2_control 控制器对接的客户端封装（演示节点共用）。

接口关系：
    本节点 --FollowJointTrajectory action--> /scara_controller/follow_joint_trajectory
    本节点 --Float64MultiArray----------> /gripper_controller/commands
    本节点 <--sensor_msgs/JointState----- /joint_states（joint_state_broadcaster 发布）
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray
import tf2_ros

from . import kinematics as K
from . import trajectory as TRAJ

GRIPPER_OPEN = 0.012     # 与 URDF 的 finger_stroke 一致
GRIPPER_JOINTS = ("gripper_left_joint", "gripper_right_joint")


class ScaraClient(Node):
    """封装：等待控制器就绪、发送 JointTrajectory、控制夹爪、读关节状态。"""

    def __init__(self, name: str = "scara_client") -> None:
        super().__init__(name)
        self.action = ActionClient(self, FollowJointTrajectory,
                                   "/scara_controller/follow_joint_trajectory")
        self.grip_pub = self.create_publisher(Float64, "/gripper/target", 10)
        self.joint_state: dict[str, float] = {}
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self._svc_clients: dict[str, object] = {}   # 注意：不能叫 _clients（rclpy Node 内部同名列表）
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._vacuum_timer = None
        self._vacuum_offset = None      # 4x4：工具 -> 物体 的固定相对位姿
        self._vacuum_object = "target_box"
        self.get_logger().info("等待 /joint_states 与 JointTrajectory action ...")

    def _on_joint_state(self, msg: JointState) -> None:
        for n, p in zip(msg.name, msg.position):
            self.joint_state[n] = float(p)

    # ---------- 就绪检查 ----------
    def wait_for_robot(self, timeout: float = 60.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.q() is not None and self.action.server_is_ready():
                self.get_logger().info(
                    f"控制器就绪，当前关节角 = {[round(v, 4) for v in self.q()]}")
                return True
        self.get_logger().error("等待超时：请确认 gazebo.launch.py 已启动且控制器 active")
        return False

    def q(self) -> Optional[list[float]]:
        return K.q_from_joint_state(self.joint_state)

    # ---------- 运动 ----------
    def send(self, points: Sequence[tuple[float, Sequence[float]]],
             timeout: float = 30.0) -> bool:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = TRAJ.to_msg(points, K.JOINT_NAMES)
        goal.goal_time_tolerance.sec = 1
        send_future = self.action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=timeout)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().error("轨迹目标被拒绝（检查关节名/限位/控制器状态）")
            return False
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future,
                                         timeout_sec=timeout + points[-1][0])
        res = result_future.result()
        if res is None:
            self.get_logger().error("等待轨迹结果超时")
            return False
        code = res.result.error_code
        if code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().warn(f"轨迹结束但 error_code={code}（{res.result.error_string}）")
        # 让 joint_states 再刷新几拍
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.02)
        return code == FollowJointTrajectory.Result.SUCCESSFUL

    def move_line(self, x: float, y: float, z: float, yaw: float = 0.0,
                  T: float = 1.0) -> bool:
        q0 = self.q()
        if q0 is None:
            return False
        if not K.reachable(x, y, z):
            self.get_logger().error(f"目标不可达：({x:.3f}, {y:.3f}, {z:.3f})")
            return False
        pts = TRAJ.move_line(q0, (x, y, z), T=T, yaw=yaw)
        self.get_logger().info(
            f"MoveL → ({x:.3f}, {y:.3f}, {z:.3f})，{len(pts)} 个点，{T:.2f} s")
        return self.send(pts)

    def move_joints(self, q_goal: Sequence[float], T: float = 1.0) -> bool:
        q0 = self.q()
        if q0 is None:
            return False
        pts = TRAJ.joint_quintic(q0, q_goal, T=T)
        return self.send(pts)

    # ---------- 夹爪（力控：gazebo_ros_force_system 的 /apply_joint_effort） ----------
    def _svc(self, srv_type, name: str):
        if name not in self._svc_clients:
            self._svc_clients[name] = self.create_client(srv_type, name)
        cli = self._svc_clients[name]
        if not cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(f"服务不可用：{name}")
            return None
        return cli

    def _apply_effort(self, joint: str, effort: float, duration: float) -> bool:
        from gazebo_msgs.srv import ApplyJointEffort
        cli = self._svc(ApplyJointEffort, "/apply_joint_effort")
        if cli is None:
            return False
        req = ApplyJointEffort.Request()
        req.joint_name = joint
        req.effort = float(effort)
        req.start_time.sec = 0
        req.start_time.nanosec = 0
        req.duration.sec = int(duration)
        req.duration.nanosec = int((duration - int(duration)) * 1e9)
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        res = fut.result()
        return bool(res is not None and res.success)

    def _clear_forces(self) -> None:
        from gazebo_msgs.srv import JointRequest
        cli = self._svc(JointRequest, "/clear_joint_efforts")
        if cli is None:
            return
        for j in GRIPPER_JOINTS:
            req = JointRequest.Request()
            req.joint_name = j
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)

    def set_gripper(self, closing: bool, settle: float = 0.6,
                    force: float = 1.5) -> None:
        """力控夹爪：发目标位置给 Gazebo 插件（0.012 张开 / 0.0 闭合）。

        插件内部用 PD 力控：f = clamp(kp*(target-q) - kd*dq, ±fmax)。
        手指被物块挡住时停在接触位置，PD 力就是夹持力（真实摩擦抓取）。
        """
        self.grip_pub.publish(Float64(data=0.0 if closing else GRIPPER_OPEN))
        self.get_logger().info(
            f"夹爪：{'闭合（力控）' if closing else '张开'}，目标 "
            f"{0.0 if closing else GRIPPER_OPEN:.3f} m")
        t0 = time.time()
        while time.time() - t0 < settle:
            rclpy.spin_once(self, timeout_sec=0.05)

    def open_gripper(self, settle: float = 0.6) -> None:
        self.set_gripper(False, settle)

    def close_gripper(self, settle: float = 0.8) -> None:
        """缓慢闭合（分 4 级目标）：避免指垫高速撞击把物块推跑。"""
        for target in (0.009, 0.006, 0.003, 0.0):
            self.grip_pub.publish(Float64(data=target))
            t0 = time.time()
            while time.time() - t0 < 0.15:
                rclpy.spin_once(self, timeout_sec=0.03)
        self.get_logger().info("夹爪：闭合（力控，分 4 级缓闭）")
        t0 = time.time()
        while time.time() - t0 < settle:
            rclpy.spin_once(self, timeout_sec=0.05)

    # ---------- 吸盘式附着（vacuum 模式） ----------
    # 为什么需要它：Gazebo Classic 的 gazebo_ros2_control 把 position/velocity 命令实现成
    # 运动学置位，指关节又是 30 g 的微型关节，力控在限位附近会让 ODE 失稳（实测 NaN）。
    # 因此演示默认用"吸盘"模型：抓取瞬间记录工具->物体的固定变换，之后每 50 ms 用
    # /set_entity_state 让物体跟随工具（与项目早期 MuJoCo 的 weld 吸盘同思路）。
    # 力控版本保留在 set_gripper(force=...)，可在手册附录里作为进阶实验。
    def _lookup_mat(self, target: str, source: str):
        import numpy as np
        tr = self._tf_buffer.lookup_transform(target, source, Time())
        t = tr.transform.translation
        q = tr.transform.rotation
        w, x, y, z = q.w, q.x, q.y, q.z
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = [t.x, t.y, t.z]
        return M

    def vacuum_attach(self, object_name: str = "target_box", settle: float = 0.2) -> bool:
        import numpy as np
        from gazebo_msgs.srv import GetEntityState
        self._vacuum_object = object_name
        cli = self._svc(GetEntityState, "/get_entity_state")
        if cli is None:
            return False
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.02)
        try:
            T_world_tool = self._lookup_mat("world", "link4")
        except Exception as exc:                       # noqa: BLE001
            self.get_logger().warn(f"TF 查询失败（吸盘附着）：{exc}")
            return False
        req = GetEntityState.Request()
        req.name = object_name
        req.reference_frame = "world"
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        res = fut.result()
        if res is None or not res.success:
            self.get_logger().warn("读不到物块位姿，吸盘附着失败")
            return False
        p = res.state.pose.position
        q = res.state.pose.orientation
        w, x, y, z = q.w, q.x, q.y, q.z
        T_world_obj = np.eye(4)
        T_world_obj[:3, :3] = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
        T_world_obj[:3, 3] = [p.x, p.y, p.z]
        self._vacuum_offset = np.linalg.inv(T_world_tool) @ T_world_obj
        if self._vacuum_timer is None:
            self._vacuum_timer = self.create_timer(0.05, self._vacuum_step)
        self.get_logger().info(f"吸盘：已附着 {object_name}")
        t0 = time.time()
        while time.time() - t0 < settle:
            rclpy.spin_once(self, timeout_sec=0.05)
        return True

    def _vacuum_step(self) -> None:
        import numpy as np
        from gazebo_msgs.srv import SetEntityState
        if self._vacuum_offset is None:
            return
        cli = self._svc_clients.get("/set_entity_state")
        if cli is None:
            cli = self.create_client(SetEntityState, "/set_entity_state")
            self._svc_clients["/set_entity_state"] = cli
        if not cli.service_is_ready():
            return
        try:
            T_world_tool = self._lookup_mat("world", "link4")
        except Exception:                              # noqa: BLE001
            return
        T = T_world_tool @ self._vacuum_offset
        import math
        m = T[:3, :3]
        # 旋转矩阵 -> 四元数（xyz w 顺序写入）
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        else:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
        req = SetEntityState.Request()
        req.state.name = self._vacuum_object
        req.state.pose.position.x = float(T[0, 3])
        req.state.pose.position.y = float(T[1, 3])
        req.state.pose.position.z = float(T[2, 3])
        req.state.pose.orientation.x = float(x)
        req.state.pose.orientation.y = float(y)
        req.state.pose.orientation.z = float(z)
        req.state.pose.orientation.w = float(w)
        req.state.reference_frame = "world"
        cli.call_async(req)

    def vacuum_detach(self) -> None:
        if self._vacuum_timer is not None:
            self.destroy_timer(self._vacuum_timer)
            self._vacuum_timer = None
        self._vacuum_offset = None
        self.get_logger().info("吸盘：已释放")

    # ---------- 查询 ----------
    def tcp(self) -> Optional[tuple[float, float, float, float]]:
        q = self.q()
        return None if q is None else K.fk(q)
