#!/bin/bash
# Cross-compile mnn_bench for RK3588S (Linux aarch64) with MNN RKNN plugin.
#
# Options:
#   --sysroot PATH       sysroot to use (must match target board glibc)
#   --cc PATH            C compiler (default aarch64-linux-gnu-gcc)
#   --cxx PATH           C++ compiler (default aarch64-linux-gnu-g++)
#   --march ARCH         target architecture (default armv8-a)
#   --build-dir DIR      build directory (default build-aarch64)
#   --extra-cxx-flags S  additional C++ flags
#
# Defaults add -static-libstdc++ -static-libgcc to reduce libstdc++ dependency.
set -euo pipefail
cd "$(dirname "$0")/.."

MNN_SRC="${MNN_SRC:-$PWD/3rd_party/MNN}"
RKNN_API_DIR="${RKNN_API_DIR:-$PWD/3rd_party/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include}"
BUILD_DIR="${BUILD_DIR:-build-aarch64}"
TARGET_ARCH="${TARGET_ARCH:-armv8-a}"
C_COMPILER="${C_COMPILER:-aarch64-linux-gnu-gcc}"
CXX_COMPILER="${CXX_COMPILER:-aarch64-linux-gnu-g++}"
SYSROOT=""
EXTRA_CXX_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
    --sysroot) SYSROOT="$2"; shift 2;;
    --cc) C_COMPILER="$2"; shift 2;;
    --cxx) CXX_COMPILER="$2"; shift 2;;
    --march) TARGET_ARCH="$2"; shift 2;;
    --build-dir) BUILD_DIR="$2"; shift 2;;
    --extra-cxx-flags) EXTRA_CXX_FLAGS="$2"; shift 2;;
    -h|--help)
        echo "Usage: $0 [--sysroot PATH] [--cc PATH] [--cxx PATH] [--march ARCH] [--build-dir DIR]"
        exit 0
        ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done

if ! command -v "$CXX_COMPILER" >/dev/null 2>&1; then
    echo "ERROR: $CXX_COMPILER not found. Install the AArch64 cross toolchain, e.g.:"
    echo "  sudo apt install g++-aarch64-linux-gnu"
    exit 1
fi

if [[ ! -d "$MNN_SRC" ]]; then
    echo "ERROR: MNN source not found: $MNN_SRC"
    exit 1
fi
if [[ ! -f "$RKNN_API_DIR/rknn_api.h" ]]; then
    echo "ERROR: rknn_api.h not found under: $RKNN_API_DIR"
    exit 1
fi

SYSROOT_CMAKE=()
if [[ -n "$SYSROOT" ]]; then
    SYSROOT_CMAKE=(-DCMAKE_SYSROOT="$SYSROOT")
fi

CXX_FLAGS="-march=$TARGET_ARCH -static-libstdc++ -static-libgcc"
if [[ -n "$EXTRA_CXX_FLAGS" ]]; then
    CXX_FLAGS="$CXX_FLAGS $EXTRA_CXX_FLAGS"
fi

cmake -S . -B "$BUILD_DIR" \
    -DFETCHCONTENT_SOURCE_DIR_MNN="$MNN_SRC" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER="$C_COMPILER" \
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
    -DCMAKE_C_FLAGS="-march=$TARGET_ARCH" \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc" \
    "${SYSROOT_CMAKE[@]}" \
    -DMNN_WITH_PLUGIN=ON \
    -DMNN_RKNN=ON \
    -DMNN_RKNN_CONVERT_MODE=OFF \
    -DMNN_BUILD_CONVERTER=OFF \
    -DMNN_BUILD_DEMO=OFF \
    -DMNN_BUILD_TOOLS=ON \
    -DMNN_METAL=OFF \
    -DMNN_CUDA=OFF \
    -DMNN_OPENCL=OFF \
    -DMNN_VULKAN=OFF \
    -DRKNN_API_INCLUDE_DIR="$RKNN_API_DIR"

cmake --build "$BUILD_DIR" -j16
echo "Binary: $BUILD_DIR/mnn_bench"
