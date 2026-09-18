#!/usr/bin/env bash
# Source this file; do not execute it as a child process.

MOVEIT_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOVEIT_MICROMAMBA="$MOVEIT_EXPERIMENT_ROOT/.tools/micromamba"
MOVEIT_ENV_PREFIX="$MOVEIT_EXPERIMENT_ROOT/.native_env"

if [[ ! -x "$MOVEIT_MICROMAMBA" || ! -d "$MOVEIT_ENV_PREFIX" ]]; then
  echo "MoveIt environment is missing. Run ./scripts/install_env.sh first." >&2
  return 1 2>/dev/null || exit 1
fi

export MAMBA_ROOT_PREFIX="$MOVEIT_EXPERIMENT_ROOT/.mamba_ros"
export CONDA_PKGS_DIRS="$MAMBA_ROOT_PREFIX/pkgs"
eval "$("$MOVEIT_MICROMAMBA" --root-prefix "$MAMBA_ROOT_PREFIX" shell hook --shell bash)"

# RoboStack 的激活钩子会读取可选的 CONDA_BUILD；调用方若启用 set -u 会导致其报错。
MOVEIT_RESTORE_NOUNSET=0
if [[ $- == *u* ]]; then
  MOVEIT_RESTORE_NOUNSET=1
  set +u
fi
micromamba activate "$MOVEIT_ENV_PREFIX"
if [[ "$MOVEIT_RESTORE_NOUNSET" == "1" ]]; then
  set -u
fi

# 仅在调用者未明确指定时使用本实验的独立 ROS 域，避免与服务器上的其他 ROS 测试串话。
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-87}"

unset MOVEIT_MICROMAMBA MOVEIT_ENV_PREFIX MOVEIT_RESTORE_NOUNSET
