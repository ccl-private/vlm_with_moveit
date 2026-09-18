#!/usr/bin/env bash
set -euo pipefail

MOVEIT_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_LOG_DIR="$MOVEIT_EXPERIMENT_ROOT/logs/ros"
export ROS_HOME="$MOVEIT_EXPERIMENT_ROOT/logs/ros_home"
mkdir -p "$ROS_LOG_DIR" "$ROS_HOME"

source "$MOVEIT_EXPERIMENT_ROOT/scripts/activate.sh"
set +u
source "$MOVEIT_EXPERIMENT_ROOT/install/vision_moveit_bridge/share/vision_moveit_bridge/local_setup.bash"
set -u

if [[ -z "${DISPLAY:-}" ]]; then
  echo "未检测到图形显示（DISPLAY 为空）。请从 VNC 桌面的终端运行此脚本。" >&2
  exit 1
fi

exec ros2 launch vision_moveit_bridge headless_panda.launch.py start_rviz:=true
