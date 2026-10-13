#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version=1.22.0
sdk_dir="$repo_dir/third_party/onnxruntime"
if [[ -f "$sdk_dir/.sdk-version" ]] && [[ "$(cat "$sdk_dir/.sdk-version")" == "$version" ]] && [[ -f "$sdk_dir/lib/libonnxruntime.so" ]]; then
  exit 0
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'The automatic SDK setup supports Linux x86_64 only.' >&2
  exit 1
fi
archive="$repo_dir/.cache/onnxruntime-linux-x64-$version.tgz"
mkdir -p "$repo_dir/.cache" "$sdk_dir"
checksum=8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3
if ! echo "$checksum  $archive" | sha256sum --check --status; then
  curl --fail --location --retry 3 --output "$archive" \
    "https://github.com/microsoft/onnxruntime/releases/download/v$version/onnxruntime-linux-x64-$version.tgz"
fi
echo "$checksum  $archive" | sha256sum --check
tar -xzf "$archive" --strip-components=1 -C "$sdk_dir"
echo "$version" > "$sdk_dir/.sdk-version"
