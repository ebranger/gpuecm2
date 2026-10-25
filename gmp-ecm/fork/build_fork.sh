#!/bin/sh
# Builds Greg Childers' GMP-ECM fork (GPU stage 1 without CGBN, branch
# gpu-stage1-dynamic of https://github.com/gchilders/gmp-ecm) with gpuecm2 as
# its stage 2: after the fork's GPU stage 1 (-gpu), stage 2 runs with gpuecm2
# instead of the fork's own GPU stage 2 (0001-gpuecm2-stage2.patch). The fork
# itself, unpatched, is the other choice; there is no automatic selection
# between the two stage 2 codes.
#
# Clones the fork at a fixed commit (both projects are under development; the
# patch is made for this commit), applies the patch and builds `ecm` linked to
# libgpuecm2. Build gpuecm2 first (cmake -S . -B build && cmake --build build).
# Needs git, autoconf, automake, libtool and m4 (the fork is a git checkout:
# configure is generated with autoreconf), and the CUDA toolkit.
#
# Usage: gmp-ecm/fork/build_fork.sh [options]
#   --gpuecm2 DIR   gpuecm2 build directory or install prefix (default: build)
#   --workdir DIR   where the fork is cloned and built (default: build-gmp-ecm-fork)
#   --source REPO   clone from this git repository instead of GitHub (a local
#                   clone that has the commit)
#   --commit SHA    the fork's commit (default: the one the patch is made for)
#   --arch NN       CUDA compute capability, e.g. 89 (default: that of GPU 0)
#   --jobs N        parallel make jobs (default: 2; the fork's kernels take about
#                   20 minutes at -j2)
# The result is WORKDIR/gmp-ecm/ecm.
#
# This script is part of gpuecm2 (LGPL-3.0-or-later); the patch it applies
# modifies GPL-3.0-or-later files of GMP-ECM and is itself GPL-3.0-or-later,
# see gmp-ecm/README.md.
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
GPUECM2=$REPO/build
WORK=$REPO/build-gmp-ecm-fork
SOURCE=https://github.com/gchilders/gmp-ecm.git
BRANCH=gpu-stage1-dynamic
COMMIT=bbb6b40c729ed1bb90ace611e830b059bfdc142a
ARCH=
JOBS=2

while [ $# -gt 0 ]; do
  case $1 in
    --gpuecm2) GPUECM2=$(cd "$2" && pwd); shift 2 ;;
    --workdir) WORK=$2; shift 2 ;;
    --source) SOURCE=$2; shift 2 ;;
    --commit) COMMIT=$2; shift 2 ;;
    --arch) ARCH=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    -h|--help) sed -n '2,28p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 1 ;;
  esac
done

# gpuecm2: a build directory (header in the source tree) or an install prefix
if [ -f "$GPUECM2/include/gpuecm2.h" ]; then
  INC=$GPUECM2/include
  LIB=$GPUECM2/lib
  [ -f "$LIB/libgpuecm2.so" ] || LIB=$GPUECM2/lib64
else
  INC=$REPO/src/api
  LIB=$GPUECM2
fi
if [ ! -f "$LIB/libgpuecm2.so" ]; then
  echo "libgpuecm2.so not found in $LIB: build gpuecm2 first (cmake -S . -B build && cmake --build build)" >&2
  exit 1
fi
for t in git autoreconf automake libtoolize m4; do
  command -v $t > /dev/null || { echo "$t not found (needed to build the fork from git)" >&2; exit 1; }
done

# CUDA toolkit and GPU architecture
NVCC=$(command -v nvcc || true)
[ -n "$NVCC" ] || NVCC=/usr/local/cuda/bin/nvcc
[ -x "$NVCC" ] || { echo "nvcc not found (put the CUDA toolkit's bin directory on PATH)" >&2; exit 1; }
CUDA=$(cd "$(dirname "$NVCC")/.." && pwd)
if [ -z "$ARCH" ]; then
  ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  [ -n "$ARCH" ] || { echo "could not detect the GPU; pass --arch NN (e.g. 89)" >&2; exit 1; }
fi

mkdir -p "$WORK"
cd "$WORK"
rm -rf gmp-ecm
echo "cloning the fork ($SOURCE, commit $COMMIT)"
git clone -q --branch "$BRANCH" "$SOURCE" gmp-ecm 2>/dev/null || git clone -q "$SOURCE" gmp-ecm
cd gmp-ecm
git -c advice.detachedHead=false checkout -q "$COMMIT"
git apply "$HERE/0001-gpuecm2-stage2.patch"

LDF="-L$LIB -Wl,-rpath,$LIB"
[ -d /usr/lib/wsl/lib ] && LDF="$LDF -L/usr/lib/wsl/lib"   # libcuda under WSL
ASM=--enable-asm-redc
echo "configuring (GPU compute capability $ARCH)"
autoreconf -i > autoreconf.out 2>&1 || { tail -20 autoreconf.out; echo "autoreconf failed" >&2; exit 1; }
./configure CFLAGS="-O2 -g" CPPFLAGS="-I$INC -DHAVE_GPUECM2" LDFLAGS="$LDF" LIBS="-lgpuecm2" \
  --enable-gpu="$ARCH" --with-cuda="$CUDA" $ASM \
  > configure.out 2>&1 || { tail -20 configure.out; echo "configure failed (see $WORK/gmp-ecm/config.log)" >&2; exit 1; }
echo "building (make -j$JOBS; the fork's GPU kernels take about 20 minutes at -j2)"
make -j"$JOBS" > make.out 2>&1 || { tail -30 make.out; echo "build failed" >&2; exit 1; }
echo 1 | ./ecm -printconfig > /dev/null 2>&1 || true
./ecm 1e3 < /dev/null 2>&1 | head -1 | grep -q "gpuecm2 stage 2" || { echo "built ecm lacks gpuecm2" >&2; exit 1; }
echo "built $WORK/gmp-ecm/ecm (the fork's GPU stage 1, gpuecm2 stage 2 with -gpu)"
