#!/usr/bin/env bash
# 在 WSL2(Ubuntu 22.04 + ROS2 Humble) 里同步并构建 SCARA 工作空间。
# 用法（在 WSL 里）：bash /mnt/c/.../ros2_ws/build.sh [--clean]
set -eo pipefail     # 注意：不能加 -u，ROS 的 setup.bash 会引用未定义变量

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="${SCARA_WS:-$HOME/scara_ros2_ws}"

echo "== 源目录: $SRC_DIR"
echo "== 工作空间: $WS"

source /opt/ros/humble/setup.bash

mkdir -p "$WS/src"
if [[ "${1:-}" == "--clean" ]]; then
  echo "== 清理 src（仅 scara_* 包）"
  rm -rf "$WS"/src/scara_description "$WS"/src/scara_bringup "$WS"/src/scara_demo "$WS"/src/scara_gripper_plugin
fi

for pkg in scara_description scara_bringup scara_demo scara_gripper_plugin; do
  rm -rf "$WS/src/$pkg"
  cp -a "$SRC_DIR/src/$pkg" "$WS/src/$pkg"
done

cd "$WS"
colcon build --symlink-install --event-handlers console_cohesion+ \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

# colcon + --symlink-install 下 ament_python 的入口点被装到 install/<pkg>/bin/，
# 而 `ros2 run` 只认 install/<pkg>/lib/<pkg>/。补一层软链接，保证 ros2 run 可用。
if [[ -d "install/scara_demo/bin" ]]; then
  mkdir -p install/scara_demo/lib/scara_demo
  for exe in install/scara_demo/bin/*; do
    ln -sfn "../../bin/$(basename "$exe")" "install/scara_demo/lib/scara_demo/$(basename "$exe")"
  done
fi

echo
echo "== 构建完成。新开终端执行："
echo "   source /opt/ros/humble/setup.bash && source $WS/install/setup.bash"
