#!/usr/bin/env bash
set -euo pipefail

MOVEIT_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$MOVEIT_EXPERIMENT_ROOT/scripts/activate.sh"
export MOVEIT_EXPERIMENT_ROOT
export PYTHONPATH="$MOVEIT_EXPERIMENT_ROOT/src/vision_moveit_demo${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" != "--headless" && -z "${DISPLAY:-}" ]]; then
  echo "未检测到图形显示（DISPLAY 为空）。请从 VNC 桌面的终端运行，或传入 --headless。" >&2
  exit 1
fi

if [[ "${1:-}" == "--headless" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
else
  export MUJOCO_GL="${MUJOCO_GL:-glfw}"
fi

echo "[阶段 1] DISPLAY=${DISPLAY:-<empty>}，MUJOCO_GL=${MUJOCO_GL}"

if [[ "${1:-}" != "--headless" && "${MOVEIT_USE_VIRTUALGL:-1}" == "1" ]]; then
  if ! command -v vglrun >/dev/null; then
    echo "未找到 VirtualGL（vglrun）；无法按默认配置使用 A800 图形渲染。" >&2
    exit 1
  fi
  moveit_vgl_device="${MOVEIT_VGL_DEVICE:-egl0}"
  echo "[阶段 1] 启用 VirtualGL：A800 EGL 设备 ${moveit_vgl_device}。"
  exec vglrun -d "$moveit_vgl_device" -- "$MOVEIT_EXPERIMENT_ROOT/.native_env/bin/python" -m vision_moveit_demo.unified_preview "$@"
fi

exec "$MOVEIT_EXPERIMENT_ROOT/.native_env/bin/python" -m vision_moveit_demo.unified_preview "$@"
