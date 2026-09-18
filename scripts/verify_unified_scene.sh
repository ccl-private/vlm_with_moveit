#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${script_dir}/.." && pwd)"
source "${script_dir}/activate.sh"
export MOVEIT_EXPERIMENT_ROOT="${root_dir}"
export PYTHONPATH="${root_dir}/src/vision_moveit_demo${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
python -m vision_moveit_demo.unified_verify
