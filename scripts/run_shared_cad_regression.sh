#!/usr/bin/env bash
# 两种 CAD 使用完全相同的通用 matcher、主动观察与执行参数，保存两路视频。
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
if [[ -z "${DISPLAY:-}" ]]; then
  echo "请从 VNC 桌面的终端执行，以保存主视角和腕部视频。" >&2
  exit 1
fi
run_dir="${1:-logs/shared_cad_regression/$(date +%Y%m%d_%H%M%S)}"
domain_base="${MOVEIT_SHARED_DOMAIN_BASE:-222}"
mkdir -p "$run_dir"
sha256sum src/vision_moveit_demo/vision_moveit_demo/*.py > "$run_dir/source_before.sha256"
planner_pid=""
cleanup() {
  if [[ -n "$planner_pid" ]] && kill -0 "$planner_pid" 2>/dev/null; then
    kill -INT "$planner_pid"
    wait "$planner_pid" || true
  fi
  planner_pid=""
}
trap cleanup EXIT INT TERM
instructions=("抓取黄色马克杯把手并放到托盘" "抓取橙色兔子耳朵并放到托盘")
for index in 0 1; do
  episode_dir="$run_dir/run_$index"
  mkdir -p "$episode_dir"
  ROS_DOMAIN_ID="$((domain_base + index))" ./scripts/start_moveit_headless.sh > "$episode_dir/moveit.log" 2>&1 &
  planner_pid=$!
  sleep 8
  ROS_DOMAIN_ID="$((domain_base + index))" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    ./scripts/run_cad_matching.sh --vnc \
      --instruction "${instructions[$index]}" \
      --force-matcher-type mesh --active-observation \
      --active-observation-mode on_path --max-wrist-observations 6 \
      --episode-dir "$episode_dir/episode" --video-dir "$episode_dir/video" \
      --active-observation-debug-dir "$episode_dir/observations" > "$episode_dir/runtime.log" 2>&1
  cleanup
done
sha256sum src/vision_moveit_demo/vision_moveit_demo/*.py > "$run_dir/source_after.sha256"
cmp "$run_dir/source_before.sha256" "$run_dir/source_after.sha256"
source scripts/activate.sh
python scripts/verify_shared_cad_episodes.py \
  "$run_dir/run_0/episode/episode.json" "$run_dir/run_1/episode/episode.json" \
  --report "$run_dir/summary.json"
echo "两种 CAD 的通用流程回归通过，日志与视频：$run_dir"
