#!/usr/bin/env bash

set -euo pipefail

host_system=$(uname -s)

cmake_options=(
    -S .
    -B build
    -DGGML_NATIVE=ON
    -DGGML_VULKAN=ON
)

cmake_help=$(cmake --help)
if [[ "$cmake_help" == *"--fresh"* ]]; then
    cmake_options+=(--fresh)
else
    rm -rf build/
fi

case "$host_system" in
    MINGW*|MSYS*|CYGWIN*)
        if ! command -v ninja >/dev/null 2>&1; then
            w64devkit_bin="$HOME/Downloads/Programs/w64devkit/bin"
            if [[ -x "$w64devkit_bin/ninja.exe" ]]; then
                export PATH="$w64devkit_bin:$PATH"
            fi
        fi

        if ! command -v ninja >/dev/null 2>&1; then
            echo "Vulkan-build.sh: Ninja is required on Windows" >&2
            exit 1
        fi

        cmake_options+=(
            -G Ninja
            -DLLAMA_OPENSSL=OFF
        )
        ;;
    *)
        if command -v ninja >/dev/null 2>&1; then
            cmake_options+=(-G Ninja)
        fi
        ;;
esac

cmake "${cmake_options[@]}"
cmake --build build --config Release -j16
