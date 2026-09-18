#!/usr/bin/env bash
set -euo pipefail

# 一键启动：MoveIt 仅规划，MuJoCo 在 VNC 中执行并显示双相机。
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

ros_domain_id="${MOVEIT_ROS_DOMAIN_ID:-72}"
log_dir="${root_dir}/logs/stage2_vnc"
mkdir -p "$log_dir"

echo "[阶段 2] 启动本地 MoveIt 规划服务（ROS_DOMAIN_ID=${ros_domain_id}）…"
# 以独立进程组启动。ros2 launch 会派生多个节点，清理时必须终止整个进程组，
# 否则 move_group 等子节点会成为孤儿并污染后续回合的 ROS 域。
setsid env ROS_DOMAIN_ID="$ros_domain_id" ./scripts/start_moveit_headless.sh >"${log_dir}/moveit_latest.log" 2>&1 &
moveit_pid=$!

cleanup() {
  kill -INT -- "-$moveit_pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$moveit_pid" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
  kill -TERM -- "-$moveit_pid" 2>/dev/null || true
  wait "$moveit_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 等待 move_group、桥接和假控制器完成发现；随后首个任务消息不会丢失。
sleep 10
echo "[阶段 2] 正在打开 VNC 任务窗口…"
ROS_DOMAIN_ID="$ros_domain_id" ./scripts/run_truth_baseline.sh --vnc --instruction "抓取红色杯子并放到托盘"
