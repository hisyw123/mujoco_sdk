#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
build_dir="$repo_dir/build-sdk"
robot_args=(--robot greeny-12inch)
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  if [[ -f "$1" ]]; then
    model_path="$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")"
    robot_args=("$model_path")
  else
    robot_args=(--robot "$1")
  fi
  shift
fi

"$repo_dir/scripts/build_sdk.sh"

echo "[GREENY] Starting: ${robot_args[*]}"
exec "$build_dir/bin/greeny_homing" \
  "${robot_args[@]}" \
  "$@"
