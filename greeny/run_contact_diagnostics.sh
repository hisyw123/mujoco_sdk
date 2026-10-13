#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
robot_args=(--robot greeny-12inch)
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  if [[ -f "$1" ]]; then
    robot_args=("$1")
  else
    robot_args=(--robot "$1")
  fi
  shift
fi

exec "$script_dir/run.sh" \
  "${robot_args[@]}" \
  --mode rl \
  --diagnostics \
  "$@"
