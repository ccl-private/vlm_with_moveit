#!/usr/bin/env bash
set -euo pipefail

# 一键启动：CAD 3D 模型匹配、MoveIt 规划与 MuJoCo/VNC 抓取执行。
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
ros_domain_id="${MOVEIT_ROS_DOMAIN_ID:-76}"
log_dir="${root_dir}/logs/stage3_cad_vnc"
mkdir -p "$log_dir"

echo "[阶段 3] 启动本地 MoveIt 规划服务（ROS_DOMAIN_ID=${ros_domain_id}）…"
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

sleep 10
echo "[阶段 3] 正在进行 RGB-D → CAD 模型匹配，并打开 VNC 任务窗口…"
ROS_DOMAIN_ID="$ros_domain_id" ./scripts/run_cad_matching.sh --vnc --instruction "抓取红色杯子并放到托盘"
