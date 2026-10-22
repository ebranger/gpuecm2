# GMP-ECM with gpuecm2 stage 2 (optional)

This directory turns GMP-ECM into a full GPU program: stage 1 on the GPU (GMP-ECM's own
`-gpu` code) and stage 2 on the GPU with gpuecm2, in one run with the new option `-gpu2`.
It is an addition to the standalone `gpuecm2` program, which works without it.

- `0001-gpu2.patch`: the option `-gpu2` (implies `-gpu`). After stage 1 on the GPU, the
  stage 2 of all curves of the run goes to gpuecm2 in one call (`libgpuecm2`, see
  docs/INTEGRATION.md) instead of GMP-ECM's per-curve CPU loop, with GMP-ECM's own stage 2
  parameters, so the results are those of the CPU loop. If dF does not fit in the free GPU
  memory, the parameters GMP-ECM chooses with the smallest larger `-k` that fits are used
  (as the standalone `gpuecm2` does); if gpuecm2 cannot run the stage 2 at all, the CPU loop
  is used.
- `0002-cgbn-kernel-sizes.patch` (optional, applied by default): GPU stage 1 kernels of 768,
  1280, 2560 and 4608 bits besides GMP-ECM's 512, 1024, 1536, 2048, 3072 and 4096, and of
  5120 to 32768 bits (steps of 1024 bits up to 8192, 2048 up to 16384, 4096 up to 32768,
  the sizes gpuecm2 uses). GMP-ECM needs a kernel with at least bits(N) + 6 bits, so without
  these a 512-bit number uses the 1024-bit kernel, a 2048-bit number the 3072-bit one, and
  numbers of more than 4090 bits cannot use the GPU at all; with them stage 1 is 20–30%
  faster for such numbers and works up to 32762 bits (GMP-ECM's limit for its GPU code).
  The patch only adds kernel instantiations; the stage 1 code is unchanged.
  `ECM_CGBN_BITS=b` forces the kernel of b bits. `build_ecm.sh` compiles the kernels up to the
  largest size of the gpuecm2 build it links (8192 bits by default; `--max-bits B` sets it,
  `-DCGBN_KERNEL_MAX_BITS=B`). The large kernels take long to compile: the whole build takes
  about 3 minutes with the kernels up to 8192 bits, about 16 with all of them, up to 32768 bits.
- `build_ecm.sh`: downloads GMP-ECM 7.0.6 (checksum checked), applies the patches and builds
  `ecm` linked to `libgpuecm2`.
- `test_gpu2.py`: checks a build: `-gpu2` and GMP-ECM's CPU stage 2 must give the same result
  for every curve.

The patches are made against the GMP-ECM 7.0.6 release, the latest with a release tarball
(7.0.7 adds an option to map the stage 1 exponent from a file and fixes for macOS).

## Licence

The patches modify files of GMP-ECM that are under the GNU General Public License, version 3
or later (`main.c`, `cudawrapper.c`, `cgbn_stage1.cu`) or the GNU Lesser General Public
License, version 3 or later (`ecm.h.in`, `factor.c`); like the `ecm` program they produce,
the patches are distributed under the GNU General Public License, version 3 or later
(`COPYING` in the top directory). The rest of this repository, including `libgpuecm2` and
the scripts in this directory, is under the GNU Lesser General Public License, version 3 or
later.
