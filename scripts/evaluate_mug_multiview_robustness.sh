#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${root_dir}/scripts/activate.sh"
export MOVEIT_EXPERIMENT_ROOT="${root_dir}"
export PYTHONPATH="${root_dir}/src/vision_moveit_demo${PYTHONPATH:+:${PYTHONPATH}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "${root_dir}"
exec "${root_dir}/.native_env/bin/python" -m vision_moveit_demo.mug_multiview_robustness_eval "$@"
