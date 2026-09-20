#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${root_dir}/scripts/activate.sh"
export MOVEIT_EXPERIMENT_ROOT="${root_dir}"
export PYTHONPATH="${root_dir}/src/vision_moveit_demo${PYTHONPATH:+:${PYTHONPATH}}"

vnc_mode=0
for argument in "$@"; do
  if [[ "$argument" == "--vnc" ]]; then
    vnc_mode=1
    break
  fi
done

if [[ "$vnc_mode" == "1" ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "--vnc 需要从 VNC 桌面的终端运行（DISPLAY 为空）。" >&2
    exit 1
  fi
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
  if [[ "${MOVEIT_USE_VIRTUALGL:-1}" == "1" ]]; then
    if ! command -v vglrun >/dev/null; then
      echo "未找到 VirtualGL（vglrun），无法按默认配置使用 A800 图形渲染。" >&2
      exit 1
    fi
    exec vglrun -d "${MOVEIT_VGL_DEVICE:-egl0}" -- "${root_dir}/.native_env/bin/python" -m vision_moveit_demo.cad_pick_place "$@"
  fi
else
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

exec "${root_dir}/.native_env/bin/python" -m vision_moveit_demo.cad_pick_place "$@"
