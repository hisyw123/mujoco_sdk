#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$project_dir/.." && pwd)"
"$repo_dir/scripts/build_sdk.sh"
exec "$repo_dir/build-sdk/bin/roughterrain3_sim" --robot rough-terrain-3-bipedal "$@"
