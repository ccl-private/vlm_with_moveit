#!/usr/bin/env bash
set -euo pipefail

MOVEIT_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOVEIT_DEMO_PYTHON="$MOVEIT_EXPERIMENT_ROOT/.native_env/bin/python"
source "$MOVEIT_EXPERIMENT_ROOT/scripts/activate.sh"
export MOVEIT_EXPERIMENT_ROOT
export PYTHONPATH="$MOVEIT_EXPERIMENT_ROOT/src/vision_moveit_demo${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

"$MOVEIT_DEMO_PYTHON" -m vision_moveit_demo.demo "$@"
