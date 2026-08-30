#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
build_dir="${root}/build"

rm -rf "${build_dir}"
cmake -S "${root}" -B "${build_dir}" \
    -DGGML_NATIVE=ON
cmake --build "${build_dir}" --config Release -j32
