#!/usr/bin/env bash
set -euo pipefail

# End-to-end evidence for the generic CAD surface matcher.  Run this from a VNC
# desktop when the fifth run of each group should save its two MP4 files.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"
base_dir="${MOVEIT_GENERIC_REGRESSION_DIR:-$root_dir/logs/generic_mesh_regression}"
# 这些 seed 在回归开始前已用同一桌面边界/碰撞检查验证为可编辑；不能把
# “编辑器正确拒绝与其它物体相交”误计成 CAD matcher 的随机失败。用户仍可
# 通过环境变量传入恰好五个逗号分隔的 seed 复现或替换该组中等扰动。
random_seed_csv="${MOVEIT_GENERIC_RANDOM_SEEDS:-20261010,20261012,20261015,20261016,20261017}"
IFS=',' read -r -a random_seeds <<<"$random_seed_csv"
if [[ "${#random_seeds[@]}" != "5" ]] || ! [[ "${random_seeds[*]}" =~ ^[0-9\ ]+$ ]]; then
  echo "MOVEIT_GENERIC_RANDOM_SEEDS 必须是恰好五个逗号分隔的整数 seed" >&2
  exit 2
fi
# 通用 mesh 的固定相机 yaw 只作粗工作区；面向把手 ROI 的稳定回归默认以
# 可达性筛选过的 NBV sweep 获取互补腕部表面。需要诊断局部策略时可显式设为
# on_path，例如 MOVEIT_GENERIC_OBSERVATION_MODE=on_path ./scripts/...
observation_mode="${MOVEIT_GENERIC_OBSERVATION_MODE:-sweep}"
max_wrist_observations="${MOVEIT_GENERIC_MAX_WRIST_OBSERVATIONS:-5}"
# 默认保持最终验收所需的两组视频；无 DISPLAY 的 CI/服务器可显式关闭录像，
# 但仍执行完全相同的视觉、规划、接触和托盘验证。
record_videos="${MOVEIT_GENERIC_RECORD_VIDEOS:-1}"
if [[ "$record_videos" != "0" && "$record_videos" != "1" ]]; then
  echo "MOVEIT_GENERIC_RECORD_VIDEOS 只能是 0 或 1" >&2
  exit 2
fi
if ! [[ "$max_wrist_observations" =~ ^[1-9][0-9]*$ ]]; then
  echo "MOVEIT_GENERIC_MAX_WRIST_OBSERVATIONS 必须是正整数" >&2
  exit 2
fi
mkdir -p "$base_dir"

run_one() {
  local group="$1" index="$2" random_seed="$3" record="$4"
  local run_dir="$base_dir/$group/run_$(printf '%02d' "$index")"
  local group_offset=0
  [[ "$group" == "random" ]] && group_offset=20
  local domain=$((110 + index + group_offset))
  local moveit_log="$run_dir/moveit.log" task_log="$run_dir/task.log"
  mkdir -p "$run_dir"
  # 独立进程组让清理时可以准确终止本轮 launch 及其 ROS 子进程。
  ROS_DOMAIN_ID="$domain" setsid ./scripts/start_moveit_headless.sh >"$moveit_log" 2>&1 &
  local moveit_pid=$!
  cleanup() {
    # ros2 launch 偶尔不会在 SIGINT 后退出，若无限 wait 会令整个 10 轮
    # 回归卡在已完成的一轮。PID 是本函数刚启动的 launch，限时 TERM 后再
    # KILL 只清理该子树，不会影响用户已有的其它 ROS domain。
    kill -TERM -- "-$moveit_pid" 2>/dev/null || true
    for _attempt in {1..20}; do
      kill -0 "$moveit_pid" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "$moveit_pid" 2>/dev/null; then
      kill -KILL -- "-$moveit_pid" 2>/dev/null || true
    fi
    wait "$moveit_pid" 2>/dev/null || true
  }
  sleep 10
  local args=(--instruction '抓取黄色马克杯把手并放到托盘' --force-matcher-type mesh
    --active-observation --active-observation-mode "$observation_mode" --max-wrist-observations "$max_wrist_observations"
    --episode-dir "$run_dir/episode")
  if [[ "$group" == "random" ]]; then
    args+=(--randomize-target --random-seed "$random_seed")
  fi
  if [[ "$record" == "1" ]]; then
    args+=(--vnc --video-dir "$run_dir/video")
  fi
  set +e
  ROS_DOMAIN_ID="$domain" ./scripts/run_cad_matching.sh "${args[@]}" >"$task_log" 2>&1
  local status=$?
  set -e
  cleanup
  return "$status"
}

failures=0
for group in default random; do
  for index in 1 2 3 4 5; do
    record=0
    [[ "$index" == "5" && "$record_videos" == "1" ]] && record=1
    random_seed="${random_seeds[$((index - 1))]}"
    echo "[generic-regression] group=$group run=$index record=$record"
    run_one "$group" "$index" "$random_seed" "$record" || failures=$((failures + 1))
  done
done

BASE_DIR="$base_dir" RANDOM_SEEDS="$random_seed_csv" python - <<'PY'
import json
import os
from pathlib import Path

base = Path(os.environ["BASE_DIR"])
records = []
required_events = {"grasp_attached", "grasp_released_physical", "physical_settle_complete", "tray_verification"}
for group in ("default", "random"):
    for run in range(1, 6):
        path = base / group / f"run_{run:02d}" / "episode" / "episode.json"
        record = {"group": group, "run": run, "episode": str(path)}
        if path.exists():
            payload = json.loads(path.read_text())
            names = {event["name"] for event in payload.get("events", [])}
            method = payload.get("pose_estimate", {}).get("matching_method")
            record.update(
                evaluation_only_success=payload.get("evaluation_only_success") is True,
                generic_matcher=method == "generic_mesh_fpfh_ransac_multiscale_icp_visible_surface_refinement",
                required_events=required_events.issubset(names),
                physical_release_without_reset=(
                    payload.get("physical_release", {}).get("qpos_reset_at_release") is False
                    and payload.get("physical_release", {}).get("qvel_reset_at_release") is False
                ),
                scenario=payload.get("scenario"),
            )
            record["passed"] = all(record[key] for key in (
                "evaluation_only_success", "generic_matcher", "required_events", "physical_release_without_reset"
            ))
        else:
            record.update(passed=False, reason="episode.json missing")
        records.append(record)
summary = {
    "random_seeds": [int(item) for item in os.environ["RANDOM_SEEDS"].split(",")],
    "records": records,
    "passed": all(item["passed"] for item in records),
}
(base / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
raise SystemExit(0 if summary["passed"] else 1)
PY

[[ "$failures" == "0" ]]
