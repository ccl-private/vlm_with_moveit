#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./scripts/activate.sh

if [[ ! -d src ]]; then
  echo "Missing ROS 2 source directory: $PWD/src" >&2
  exit 1
fi

colcon build --symlink-install "$@"
