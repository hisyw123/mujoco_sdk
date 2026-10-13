#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
"$repo_dir/scripts/setup_onnxruntime.sh"
cmake -S "$repo_dir" -B "$repo_dir/build-sdk" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMUJOCO_BUILD_GREENY=ON \
  -DMUJOCO_BUILD_ROUGHTERRAIN3=ON \
  -DMUJOCO_BUILD_TESTS=OFF \
  -DMUJOCO_TEST_PYTHON_UTIL=OFF \
  -DMUJOCO_BUILD_EXAMPLES=OFF \
  -DMUJOCO_BUILD_SIMULATE=ON \
  -DMUJOCO_GREENY_ENABLE_ONNX=ON \
  -DBUILD_TESTING=ON \
  -DGREENY_ONNXRUNTIME_ROOT="$repo_dir/third_party/onnxruntime" \
  -DONNXRUNTIME_ROOT="$repo_dir/third_party/onnxruntime" "$@"
cmake --build "$repo_dir/build-sdk" --parallel "${BUILD_JOBS:-4}" \
  --target greeny_homing roughterrain3_sim controller_contract_test
