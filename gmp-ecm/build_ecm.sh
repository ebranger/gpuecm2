#!/bin/sh
# Builds GMP-ECM 7.0.6 with GPU stage 1 and gpuecm2 stage 2 (option -gpu2).
#
# Downloads the GMP-ECM 7.0.6 release, checks it, applies the patches of this
# directory and builds `ecm` linked to libgpuecm2. Build gpuecm2 first
# (cmake -S . -B build && cmake --build build).
#
# Usage: gmp-ecm/build_ecm.sh [options]
#   --gpuecm2 DIR   gpuecm2 build directory or install prefix (default: build)
#   --workdir DIR   where GMP-ECM is downloaded and built (default: build-gmp-ecm)
#   --arch NN       CUDA compute capability, e.g. 89 (default: that of GPU 0)
#   --jobs N        parallel make jobs (default: 2)
#   --max-bits B    compile the GPU stage 1 kernels up to B bits (default: the largest
#                   modulus of the gpuecm2 build, 8192 by default; at most 32768, the
#                   large kernels take long to compile, see gmp-ecm/README.md)
#   --no-kernel-sizes   do not apply 0002-cgbn-kernel-sizes.patch
# The result is WORKDIR/ecm-7.0.6/ecm.
#
# This script is part of gpuecm2 (LGPL-3.0-or-later); the patches it applies
# modify GPL-3.0-or-later files of GMP-ECM and are themselves GPL-3.0-or-later,
# see gmp-ecm/README.md.
set -e

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
GPUECM2=$REPO/build
WORK=$REPO/build-gmp-ecm
ARCH=
JOBS=2
MAXBITS=
KERNELS=1
VERSION=7.0.6
URL=https://gitlab.inria.fr/zimmerma/ecm/uploads/ad3e5019fef98819ceae58b78f4cce93/ecm-$VERSION.tar.gz
SHA256=7d20ece61ab6a20ad85f2c18064cabd77dc46a96ff894b5220dbb16e4666e8a5

while [ $# -gt 0 ]; do
  case $1 in
    --gpuecm2) GPUECM2=$(cd "$2" && pwd); shift 2 ;;
    --workdir) WORK=$2; shift 2 ;;
    --arch) ARCH=$2; shift 2 ;;
    --jobs) JOBS=$2; shift 2 ;;
    --max-bits) MAXBITS=$2; shift 2 ;;
    --no-kernel-sizes) KERNELS=0; shift ;;
    -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
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
if [ ! -f ecm-$VERSION.tar.gz ]; then
  echo "downloading GMP-ECM $VERSION"
  if command -v curl > /dev/null; then curl -sSfL -o ecm-$VERSION.tar.gz "$URL"
  else wget -q -O ecm-$VERSION.tar.gz "$URL"; fi
fi
echo "$SHA256  ecm-$VERSION.tar.gz" | sha256sum -c --quiet - || { echo "checksum mismatch" >&2; exit 1; }

rm -rf ecm-$VERSION
tar xzf ecm-$VERSION.tar.gz
cd ecm-$VERSION
patch -s -p1 < "$HERE/0001-gpu2.patch"
[ $KERNELS = 1 ] && patch -s -p1 < "$HERE/0002-cgbn-kernel-sizes.patch"

# GMP-ECM 7.0.6's configure only knows compute capabilities up to 9.0: configure
# for 7.5 and retarget the generated Makefile to the wanted architecture.
# Plain CFLAGS: GMP's suggested flags include -pedantic, which makes the
# configure test of cuda.h fail.
LDF="-L$LIB -Wl,-rpath,$LIB"
[ -d /usr/lib/wsl/lib ] && LDF="$LDF -L/usr/lib/wsl/lib"   # libcuda under WSL
ASM=--enable-asm-redc
command -v m4 > /dev/null || ASM=--disable-asm-redc   # asm-redc needs m4
# GPU stage 1 kernels up to the largest modulus gpuecm2 was built for (its stage 2 refuses
# larger numbers; GMP-ECM then does stage 2 on the CPU)
if [ -z "$MAXBITS" ]; then
  if [ -f "$GPUECM2/gen/gpu/cgbn_size_list.h" ]; then
    MAXBITS=$(sed -n 's/^#define GPUECM2_MAX_BITS \([0-9]*\).*/\1/p' "$GPUECM2/gen/gpu/cgbn_size_list.h")
  elif [ -x "$GPUECM2/bin/gpuecm2" ]; then
    MAXBITS=$("$GPUECM2/bin/gpuecm2" -h 2>&1 | sed -n 's/^Numbers of up to \([0-9]*\) bits.*/\1/p')
  fi
  [ -n "$MAXBITS" ] || MAXBITS=8192
fi
CPPF="-I$INC -DHAVE_GPUECM2 -DCGBN_KERNEL_MAX_BITS=$MAXBITS"
echo "configuring (GPU compute capability $ARCH, stage 1 kernels up to $MAXBITS bits)"
./configure CFLAGS="-O2 -g" CPPFLAGS="$CPPF" LDFLAGS="$LDF" LIBS="-lgpuecm2" \
  --enable-gpu=75 --with-cuda="$CUDA" --with-cgbn-include="$REPO/third_party/cgbn/include/cgbn" $ASM \
  > configure.out 2>&1 || { tail -20 configure.out; echo "configure failed (see $WORK/ecm-$VERSION/config.log)" >&2; exit 1; }
sed -i "s/compute_75/compute_$ARCH/g; s/sm_75/sm_$ARCH/g" Makefile
echo "building (make -j$JOBS; compiling the GPU kernels takes 3 minutes up to 8192 bits, 15 up to 32768)"
make -j"$JOBS" ecm > make.out 2>&1 || { tail -30 make.out; echo "build failed" >&2; exit 1; }
./ecm -h 2>&1 | grep -q -- "-gpu2" || { echo "built ecm lacks -gpu2" >&2; exit 1; }
echo "built $WORK/ecm-$VERSION/ecm (GPU stage 1, -gpu2 for gpuecm2 stage 2)"
