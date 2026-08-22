#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

cd "$PROJECT_ROOT"

# ============================================================
# Platform detection
# ============================================================

UNAME_S="$(uname -s 2>/dev/null || echo unknown)"

if [[ -n "${WSL_INTEROP:-}" ]] || grep -qi microsoft /proc/version 2>/dev/null; then
    PLATFORM="wsl"
elif [[ "$UNAME_S" == Linux* ]]; then
    if [[ -n "${TERMUX_VERSION:-}" ]] ||
       [[ "${PREFIX:-}" == *"com.termux"* ]]; then
        PLATFORM="termux"
    else
        PLATFORM="linux"
    fi
elif [[ "${OS:-}" == "Windows_NT" ]] ||
     [[ "$UNAME_S" == "Windows_NT" ]] ||
     [[ "$UNAME_S" == MINGW* ]] ||
     [[ "$UNAME_S" == MSYS* ]] ||
     [[ "$UNAME_S" == CYGWIN* ]]; then
    PLATFORM="windows"
else
    PLATFORM="unknown"
fi

echo "[aikar-engine] Platform: $PLATFORM"

# ============================================================
# Helpers
# ============================================================

find_command() {
    local name="$1"

    if command -v "$name" >/dev/null 2>&1; then
        command -v "$name"
        return 0
    fi

    return 1
}

is_windows_exe() {
    [[ "$1" == *.exe ]]
}

# ============================================================
# Find CMake
# ============================================================

CMAKE=""

case "$PLATFORM" in
    wsl)
        # Prefer native Linux CMake.
        if [[ -x /usr/bin/cmake ]]; then
            CMAKE="/usr/bin/cmake"
        elif [[ -x /usr/local/bin/cmake ]]; then
            CMAKE="/usr/local/bin/cmake"
        else
            # Fall back to Windows CMake visible through WSL PATH.
            CMAKE="$(find_command cmake 2>/dev/null || true)"

            if [[ -z "$CMAKE" ]]; then
                CMAKE="$(find_command cmake.exe 2>/dev/null || true)"
            fi
        fi
        ;;

    termux)
        CMAKE="$(find_command cmake 2>/dev/null || true)"

        if [[ -z "$CMAKE" && -n "${PREFIX:-}" ]]; then
            if [[ -x "$PREFIX/bin/cmake" ]]; then
                CMAKE="$PREFIX/bin/cmake"
            fi
        fi
        ;;

    windows)
        CMAKE="$(find_command cmake 2>/dev/null || true)"

        if [[ -z "$CMAKE" ]]; then
            CMAKE="$(find_command cmake.exe 2>/dev/null || true)"
        fi
        ;;

    linux|unknown)
        CMAKE="$(find_command cmake 2>/dev/null || true)"
        ;;
esac

if [[ -z "$CMAKE" ]]; then
    echo "[ERROR] CMake not found." >&2

    case "$PLATFORM" in
        termux)
            echo "Install with: pkg install cmake ninja" >&2
            ;;
        wsl|linux)
            echo "Install native Linux CMake, e.g. apt install cmake ninja-build" >&2
            ;;
        windows)
            echo "Install CMake or add cmake.exe to PATH." >&2
            ;;
    esac

    exit 1
fi

echo "[aikar-engine] CMake: $CMAKE"

# ============================================================
# Determine whether CMake is native or Windows executable
# ============================================================

CMAKE_IS_WINDOWS=0

if [[ "$PLATFORM" == "windows" ]] || is_windows_exe "$CMAKE"; then
    CMAKE_IS_WINDOWS=1
fi

# Windows executables running inside WSL need Windows paths.
if [[ "$PLATFORM" == "wsl" && "$CMAKE_IS_WINDOWS" -eq 1 ]]; then
    if ! command -v wslpath >/dev/null 2>&1; then
        echo "[ERROR] Windows CMake under WSL requires wslpath." >&2
        exit 1
    fi

    CMAKE_SOURCE_DIR="$(wslpath -w "$PROJECT_ROOT")"
    CMAKE_BUILD_DIR="$(wslpath -w "$BUILD_DIR")"

    echo "[aikar-engine] CMake mode: Windows executable via WSL"
else
    CMAKE_SOURCE_DIR="$PROJECT_ROOT"
    CMAKE_BUILD_DIR="$BUILD_DIR"

    echo "[aikar-engine] CMake mode: native"
fi

# ============================================================
# Find Ninja
# ============================================================

NINJA=""

# Important:
# Windows CMake should use Windows Ninja.
# Native CMake should use native Ninja.
if [[ "$CMAKE_IS_WINDOWS" -eq 1 ]]; then
    if command -v ninja.exe >/dev/null 2>&1; then
        NINJA="$(command -v ninja.exe)"
    elif command -v ninja >/dev/null 2>&1; then
        candidate="$(command -v ninja)"

        if [[ "$candidate" == *.exe ]]; then
            NINJA="$candidate"
        fi
    fi
else
    if command -v ninja >/dev/null 2>&1; then
        candidate="$(command -v ninja)"

        if [[ "$candidate" != *.exe ]]; then
            NINJA="$candidate"
        fi
    fi
fi

if [[ -n "$NINJA" ]]; then
    echo "[aikar-engine] Ninja: $NINJA"
else
    echo "[aikar-engine] Ninja: not found"
fi

# ============================================================
# Parallel jobs
# ============================================================

if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
elif command -v getconf >/dev/null 2>&1; then
    JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
else
    JOBS=4
fi

if ! [[ "$JOBS" =~ ^[0-9]+$ ]] || [[ "$JOBS" -lt 1 ]]; then
    JOBS=4
fi

echo "[aikar-engine] Build jobs: $JOBS"

# ============================================================
# CMake options
# ============================================================

CMAKE_OPTIONS=(
    -S "$CMAKE_SOURCE_DIR"
    -B "$CMAKE_BUILD_DIR"
    -DGGML_NATIVE=ON
    -DGGML_VULKAN=ON
    -DCMAKE_BUILD_TYPE=Release
)

if [[ "$CMAKE_IS_WINDOWS" -eq 1 ]]; then
    C_COMPILER="$(find_command gcc.exe 2>/dev/null || find_command gcc 2>/dev/null || true)"
    CXX_COMPILER="$(find_command g++.exe 2>/dev/null || find_command g++ 2>/dev/null || true)"

    if [[ -z "$C_COMPILER" || -z "$CXX_COMPILER" ]]; then
        echo "[ERROR] GCC and G++ are required for the Windows build." >&2
        exit 1
    fi

    echo "[aikar-engine] C compiler: $C_COMPILER"
    echo "[aikar-engine] C++ compiler: $CXX_COMPILER"

    CMAKE_OPTIONS+=(
        -DCMAKE_C_COMPILER="$C_COMPILER"
        -DCMAKE_CXX_COMPILER="$CXX_COMPILER"
    )
fi

if [[ -n "$NINJA" ]]; then
    CMAKE_OPTIONS+=(-G Ninja)
fi

if [[ "$CMAKE_IS_WINDOWS" -eq 1 ]]; then
    CMAKE_OPTIONS+=(
        -DLLAMA_OPENSSL=OFF
    )
fi

# ============================================================
# Fresh configure
# ============================================================

if "$CMAKE" --help 2>/dev/null | grep -q -- '--fresh'; then
    CMAKE_OPTIONS+=(--fresh)
else
    rm -rf -- "$BUILD_DIR"
fi

# ============================================================
# Configure
# ============================================================

echo
echo "============================================================"
echo " Configuring aikar-engine Vulkan build"
echo "============================================================"
echo

"$CMAKE" "${CMAKE_OPTIONS[@]}"

# ============================================================
# Build
# ============================================================

echo
echo "============================================================"
echo " Building"
echo "============================================================"
echo

"$CMAKE" \
    --build "$CMAKE_BUILD_DIR" \
    --config Release \
    --parallel "$JOBS"

echo
echo "============================================================"
echo " Build complete"
echo "============================================================"
