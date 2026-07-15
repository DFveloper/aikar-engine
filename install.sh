#!/usr/bin/env bash

set -euo pipefail

root=$(cd "$(dirname "$0")" && pwd)
dest="${HOME}/.local/bin"

"${root}/CUDA-build.sh"
mkdir -p "${dest}"

for file in "${root}/build/bin/"*; do
    [ -e "${file}" ] || continue
    ln -sfn "${file}" "${dest}/$(basename "${file}")"
done

echo "build complete; ${dest} now points to ${root}/build/bin"
