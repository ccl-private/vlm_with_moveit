#!/usr/bin/env bash
set -euo pipefail

MOVEIT_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_LOG_DIR="$MOVEIT_EXPERIMENT_ROOT/logs/ros"
export ROS_HOME="$MOVEIT_EXPERIMENT_ROOT/logs/ros_home"
mkdir -p "$ROS_LOG_DIR"
mkdir -p "$ROS_HOME"
source "$MOVEIT_EXPERIMENT_ROOT/scripts/activate.sh"
set +u
source "$MOVEIT_EXPERIMENT_ROOT/install/vision_moveit_bridge/share/vision_moveit_bridge/local_setup.bash"
set -u
exec ros2 launch vision_moveit_bridge headless_panda.launch.py
