#!/usr/bin/env bash
# 联调自检脚本（在 WSL 里运行）：重启 Gazebo 无界面 -> 检查控制器/关节状态
set -eo pipefail

source /opt/ros/humble/setup.bash
source "$HOME/scara_ros2_ws/install/setup.bash"

cleanup() {
  # 用 [x] 写法避免 pkill 匹配到自己的命令行（否则会"自杀"并漏杀真正的 gzserver）
  for pat in "[r]os2 launch scara_bringup" "[g]zserver" "[s]pawner_" \
             "[r]obot_state_publisher" "[g]azebo_ros2_control" "[s]cara_pick_place"; do
    pkill -f "$pat" 2>/dev/null || true
  done
  sleep 2
}

cleanup
cd "$HOME/scara_ros2_ws"
setsid nohup ros2 launch scara_bringup gazebo.launch.py gui:=false \
  > /tmp/scara_launch.log 2>&1 < /dev/null &

for i in $(seq 1 30); do
  sleep 1
  if grep -q "Successfully loaded controller" /tmp/scara_launch.log 2>/dev/null; then
    break
  fi
done

echo "=== 控制器状态 ==="
timeout 12 ros2 control list_controllers 2>&1 || true
echo
echo "=== /joint_states（一次）==="
timeout 8 ros2 topic echo /joint_states --once 2>&1 | head -16 || true
echo
echo "=== 日志中的错误 ==="
grep -iE "error|fail|exception" /tmp/scara_launch.log | head -12 || true

echo
echo "=== 相机话题 ==="
timeout 8 ros2 topic list 2>/dev/null | grep -E "cam_.*image_raw" || true
timeout 10 ros2 run scara_demo capture --topic /cam_top/image_raw --out /tmp/cam_top.ppm \
  --ros-args -p use_sim_time:=true 2>&1 | tail -2 || true

echo
echo "=== 解析 FK vs TF ==="
timeout 90 ros2 run scara_demo check_tf 2>&1 | tail -12 || true

echo
echo "=== 抓取-放置演示 ==="
timeout 150 ros2 run scara_demo pick_place 2>&1 \
  | grep -E "=====|TCP|完成|失败|轨迹|error_code|WARN" | tail -26 || true

echo
echo "=== 相机取帧 ==="
timeout 15 ros2 run scara_demo capture --topic /cam_top/image_raw --out /tmp/cam_top.ppm 2>&1 | tail -1 || true
timeout 15 ros2 run scara_demo capture --topic /cam_eye/image_raw --out /tmp/cam_eye.ppm 2>&1 | tail -1 || true
cp -f /tmp/cam_top.ppm /tmp/cam_eye.ppm \
  "/mnt/c/Users/42962/Documents/ChatGPT/2轴机器人的全栈开发/outputs/data/" 2>/dev/null || true
ls -l /tmp/cam_*.ppm 2>/dev/null || true
