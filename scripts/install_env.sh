#!/usr/bin/env bash
set -euo pipefail

MOVEIT_EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOVEIT_MICROMAMBA="$MOVEIT_EXPERIMENT_ROOT/.tools/micromamba"
MOVEIT_ENV_PREFIX="$MOVEIT_EXPERIMENT_ROOT/.native_env"
MOVEIT_PROXY="${MOVEIT_PROXY-http://192.168.2.189:7890}"
MOVEIT_CONDA_FORGE_CHANNEL="${MOVEIT_CONDA_FORGE_CHANNEL-https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge}"
export MAMBA_ROOT_PREFIX="$MOVEIT_EXPERIMENT_ROOT/.mamba_ros"
export CONDA_PKGS_DIRS="$MAMBA_ROOT_PREFIX/pkgs"
# 网络不稳定时让 micromamba 对单个连接多次重试；配置只对本脚本有效。
export MAMBA_REMOTE_CONNECT_TIMEOUT_SECS="${MAMBA_REMOTE_CONNECT_TIMEOUT_SECS:-30}"
export MAMBA_REMOTE_MAX_RETRIES="${MAMBA_REMOTE_MAX_RETRIES:-8}"
# 代理对大量并发大文件不稳定，默认串行下载；可在命令行覆盖。
export MAMBA_DOWNLOAD_THREADS="${MAMBA_DOWNLOAD_THREADS:-1}"
# 对部分镜像不请求 zstd 格式索引，降低中断后索引缓存损坏的风险。
export MAMBA_REPODATA_USE_ZST="${MAMBA_REPODATA_USE_ZST:-false}"

if [[ "$MOVEIT_PROXY" != "direct" ]]; then
  export http_proxy="$MOVEIT_PROXY"
  export https_proxy="$MOVEIT_PROXY"
  export HTTP_PROXY="$MOVEIT_PROXY"
  export HTTPS_PROXY="$MOVEIT_PROXY"
fi
# conda-forge 由已验证的国内镜像直连；RoboStack Jazzy 没有完整镜像，继续走代理。
export no_proxy="${no_proxy:+$no_proxy,}mirrors.ustc.edu.cn"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}mirrors.ustc.edu.cn"

if [[ ! -x "$MOVEIT_MICROMAMBA" ]]; then
  mkdir -p "$MOVEIT_EXPERIMENT_ROOT/.tools"
  archive="$(mktemp /tmp/micromamba.XXXXXX.tar.bz2)"
  trap 'rm -f "$archive"' EXIT
  curl --fail --location --silent --show-error \
    https://micro.mamba.pm/api/micromamba/linux-64/latest \
    --output "$archive"
  tar -xjf "$archive" -C "$MOVEIT_EXPERIMENT_ROOT/.tools" \
    --strip-components=1 bin/micromamba
  chmod +x "$MOVEIT_MICROMAMBA"
fi

mkdir -p "$MAMBA_ROOT_PREFIX"
"$MOVEIT_MICROMAMBA" config set --file "$MAMBA_ROOT_PREFIX/.condarc" repodata_use_zst false

MOVEIT_MAMBA_ACTION=create
if [[ -f "$MOVEIT_ENV_PREFIX/conda-meta/history" ]]; then
  MOVEIT_MAMBA_ACTION=install
fi
"$MOVEIT_MICROMAMBA" --root-prefix "$MAMBA_ROOT_PREFIX" "$MOVEIT_MAMBA_ACTION" --yes --prefix "$MOVEIT_ENV_PREFIX" \
  --override-channels --strict-channel-priority \
  --channel "$MOVEIT_CONDA_FORGE_CHANNEL" --channel robostack-jazzy \
  ros-jazzy-ros-base ros-jazzy-moveit \
  ros-jazzy-moveit-resources-panda-moveit-config ros-dev-tools \
  ros-jazzy-joint-state-broadcaster ros-jazzy-joint-trajectory-controller

echo "Installed native MoveIt environment: $MOVEIT_ENV_PREFIX"
