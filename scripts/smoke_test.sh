#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./scripts/activate.sh

echo "ROS_DISTRO=${ROS_DISTRO}"
ros2 pkg prefix moveit_core
ros2 pkg prefix moveit_resources_panda_moveit_config
ros2 pkg executables moveit_ros_move_group | head -1
echo "MoveIt package discovery passed."
