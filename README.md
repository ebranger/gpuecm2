# gpuecm2: ECM stage 2 on the GPU

gpuecm2 runs the stage 2 of the elliptic curve method (GMP-ECM's FFT continuation) on an
NVIDIA GPU. It reads the residues that GMP-ECM saves after stage 1 (`ecm -save`, or
`ecm -gpu -save` for GPU stage 1) and is a drop-in replacement for `ecm -resume`: it chooses
exactly GMP-ECM's stage 2 parameters, covers the same primes, and finds the same factors.
For every curve tested, its results agree with GMP-ECM's bit for bit (including GMP-ECM's
internal "Product of G(f_i)" value), and it finds again 1461 stage 2 factors from logs of real
factoring work.

On an RTX 4070 Ti SUPER a stage 2 at GMP-ECM's default parameters takes 160–240 times less
time than GMP-ECM on one core of an i7-14700F, for numbers of 512 to 4096 bits and B1 from
3e6 up (e.g. 12 ms instead of 2.3 s for a 1024-bit number at B1 = 3e6, 3.3 s instead of 655 s
for 4096 bits at B1 = 11e7); see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

For a 1024-bit number and the B1 values from GMP-ECM's table (with GMP-ECM's
default B2), on an RTX 4070 Ti SUPER and one core of an i7-14700F the following timings were
obtained, where values are average time for one curve in cases several are batched together:

| Factor size | B1 | B2 (GMP-ECM default) | GPU stage 1 | CPU stage 2 | GPU stage 2 | Stage 2 speedup |
|---|---|---|---|---|---|---|
| 20 digits | 11e3 | 1 873 422 | 0.1 ms | 20 ms | 0.2 ms | 118× |
| 25 digits | 5e4 | 12 746 592 | 0.6 ms | 67 ms | 0.4 ms | 152× |
| 30 digits | 25e4 | 128 992 510 | 3.0 ms | 244 ms | 1.4 ms | 173× |
| 35 digits | 1e6 | 1 045 563 762 | 13 ms | 1.03 s | 5.6 ms | 184× |
| 40 digits | 3e6 | 5 706 890 290 | 39 ms | 2.28 s | 12 ms | 184× |
| 45 digits | 11e6 | 35 133 391 030 | ≈143 ms | 7.39 s | 40 ms | 183× |
| 50 digits | 43e6 | 240 490 660 426 | ≈559 ms | 22.2 s | 118 ms | 188× |
| 55 digits | 11e7 | 776 278 396 540 | 1.34 s | 57.0 s | 324 ms | 176× |
| 60 digits | 26e7 | 3 178 559 884 516 | ≈3.17 s | 118 s | 652 ms | 182× |
| 65 digits | 85e7 | 15 892 628 251 516 | ≈10.4 s | 290 s | 1.53 s | 189× |
| 70 digits | 29e8 | 90 975 170 616 348 | ≈35.3 s | 13 min | 4.07 s | 194× |
| 75 digits | 76e8 | 324 909 696 561 468 | ≈92.6 s | 40 min | 12.2 s | 198× |
| 80 digits | 25e9 | 1 767 494 382 113 478 | ≈305 s | – | 62.9 s | – |

GPU stage 1 is GMP-ECM 7.0.6's `-gpu` (per curve of a full batch of 4224 curves). B1 = 11e7
was measured on numbers of 1037–1226 bits, which use the same 1280-bit CGBN kernel as 1024-bit
ones; values marked ≈ are extrapolated linearly in B1, from B1 = 3e6 for 11e6 and 43e6 and
from 11e7 above (a full batch at those B1 takes hours);
CPU stage 2 is GMP-ECM 7.0.6 on one core (not run at 80 digits); GPU stage 2 is gpuecm2 with
the curves of a GPU stage 1 run processed together. From B1 = 26e7 on, both programs ran with
`-maxmem 8000`. Other sizes (512 to 4096 bits, a speedup of 160–240× from B1 = 3e6 on) are
in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

gpuecm2 can be used in two ways:

- **The `gpuecm2` program** (built by default): stage 2 of residues saved by any GMP-ECM
  run, a drop-in replacement for `ecm -resume`. Nothing else is changed.
- **GMP-ECM with GPU stage 1 and stage 2 in one run** (optional): `gmp-ecm/` holds patches
  for GMP-ECM 7.0.6 that add the option `-gpu2`, and a script that builds such an `ecm`
  linked to gpuecm2's library; see [Full GPU build of GMP-ECM](#full-gpu-build-of-gmp-ecm).

The library has a C interface for other programs as well; see
[docs/INTEGRATION.md](docs/INTEGRATION.md).

## Usage

```
gpuecm2 [-v] [-q] [-k n] [-power n | -dickson n] [-maxmem MB] [-no-ntt | -ntt] [-one]
        [-selfcheck] [-treecache] [-batch n] [-spin] -resume <file> B1 [B2 | B2min-B2max]
```

The options mean what they mean for `ecm` (stage 2 only). In addition:

- `-v -v` prints GMP-ECM's "Product of G(f_i)" value when no factor is found; `-v -v -v` adds
  per-phase times (measuring them serialises the GPU work, so step 2 gets slightly slower).
- `-selfcheck` verifies each curve's stage 2 at random points (product trees, and the final
  values at a few roots of F), and recomputes a few roots (two of F and one of each block of
  G, of the first curve of each batch) on the CPU with an independent method, at a cost of
  5–20%. Off by default.
- `-treecache` keeps the transforms of F's product tree for the final evaluation, which makes
  stage 2 2–6% faster for numbers of up to 1920 bits, at the cost of GPU memory: per curve,
  0.4 GB at 512 bits and 0.8 GB at 1024 bits for B1 = 11e7, 1.6 GB at 1024 bits for 26e7
  (`gpuecm2 -h` has a table). It uses at most the memory left free by the rest of stage 2.
  Off by default.
- Consecutive residues of the same number (as `ecm -save` and `ecm -gpu -save` write them) are
  processed together on the GPU when one curve alone cannot keep it busy (B1 below about 1e7).
  `-batch n` sets the number of residues per batch (default: automatic; `-batch 1` turns
  batching off). The output is the same as without batching; "Step 2 took" then shows the
  batch time divided by the number of residues. After a factor is found, the following
  residues of that number continue with the cofactor, as in GMP-ECM.
- The GPU memory needed grows with dF, not with k. If GMP-ECM's dF does not fit, gpuecm2 uses
  the parameters GMP-ECM would choose with the smallest larger `-k` that fits (more, smaller
  blocks, still covering B2) and prints a note.
- While the GPU works the CPU thread sleeps, so all cores stay available for CPU stage 1;
  `-spin` busy-waits instead (no measurable speed difference on the test system).

Limits: odd numbers of up to 4608 bits (other residues are skipped with a message); any B2
(B2 and B2min may exceed 2^64); parametrizations 0–3 and curves given by A (not GMP-ECM's
`-torsion` curves). Improper options or bounds are refused before anything runs, and
unusable resume lines are skipped with a message (docs/TESTING.md).

## Requirements

- NVIDIA GPU with compute capability 7.5 or newer (Turing, 2018, and later)
- CUDA toolkit 12.x or 13.x to build (the program and library link the CUDA runtime
  statically and need only the driver to run)
- GMP
- CMake 3.24 or newer
- CGBN is included in `third_party/cgbn`

## Build

```
cmake -S . -B build                     # for the GPU in this machine
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="75;86;89;120"   # distribution build
cmake --build build -j 2
cd build && ctest
cmake --install build --prefix /usr/local   # gpuecm2, libgpuecm2.so, gpuecm2.h
```

Compiling the CGBN kernels for all twelve modulus sizes (512 to 4608 bits) takes about seven
minutes. For development, `-DGPUECM2_ONLY_BITS=1024` builds a single size (moduli up to 1024 bits)
in about a minute.

On Windows, use the MSVC toolchain with GMP from vcpkg (`-DGMP_ROOT=...`); the library is then
built as a static library.

## Full GPU build of GMP-ECM

Optional, in addition to the above: GMP-ECM 7.0.6 with its GPU stage 1 and gpuecm2's stage 2,
so that a whole ECM run happens on the GPU. It needs the gpuecm2 build above, the CUDA
toolkit, GMP and an internet connection (for the GMP-ECM release).

```
gmp-ecm/build_ecm.sh                        # makes build-gmp-ecm/ecm-7.0.6/ecm
python3 gmp-ecm/test_gpu2.py                # checks it against GMP-ECM's CPU stage 2 (a few minutes)
```

`build_ecm.sh` downloads the GMP-ECM 7.0.6 release, checks its checksum, applies the patches in
`gmp-ecm/` and builds `ecm` with GPU support for the GPU in the machine (`--arch NN` for
another compute capability, `--gpuecm2 DIR` for an installed gpuecm2, `--no-kernel-sizes` to
leave GMP-ECM's GPU stage 1 kernels as they are; see [gmp-ecm/README.md](gmp-ecm/README.md)).
The patches are under the GPL, like GMP-ECM; see that file.

Usage is that of GMP-ECM's `-gpu` with one more option:

```
echo "<number>" | build-gmp-ecm/ecm-7.0.6/ecm -gpu2 [-gpucurves n] [-sigma 3:s] [-v] B1 [B2]
```

- `-gpu2` implies `-gpu`: stage 1 of a batch of curves (parametrization 3; `-gpucurves`
  curves, by default as many as fill the GPU) runs on the GPU, then stage 2 of all of them on
  the GPU, with GMP-ECM's stage 2 parameters. Factors are reported as with `-gpu` ("GPU: factor
  ... found in Step 2 with curve i"), and `-v` reports "Computing n Step 2 on GPU took ...ms".
- Everything else works as with `-gpu`: `-c`, `-save`/`-savea` (the stage 1 residues),
  `-maxmem`, `-k`, `-power`/`-dickson`, input files, `-one`, and so on.
- If GMP-ECM's dF does not fit in the free GPU memory, stage 2 uses the parameters GMP-ECM
  chooses with the smallest larger `-k` that fits (more, smaller blocks covering the same B2),
  with a message, as the standalone `gpuecm2` does. If gpuecm2 cannot run the stage 2 at all,
  GMP-ECM's CPU stage 2 is used, with a message.
- With the kernel-size patch, numbers of up to 4602 bits can use GPU stage 1 (GMP-ECM 7.0.6
  alone stops at 4090 bits); `ECM_CGBN_BITS=b` forces the stage 1 kernel of b bits.

Stage 1 and stage 2 run one after the other: on a GPU both keep it fully busy, and running
them concurrently was measured to be slower (docs/BENCHMARKS.md).

## Tests

[docs/TESTING.md](docs/TESTING.md) describes the tests. In short:

- `ctest`: unit tests of every part, including the C interface (about a minute).
- `python3 tests/run_hits.py build/gpuecm2`: known stage 2 factors from real factoring work
  (`tests/data/stage2_hits.txt`) must be found again; no GMP-ECM needed.
- With a GMP-ECM binary: `tests/boundary_tests.py` compares both programs on both sides of
  every size and parameter threshold; `tests/compare_stage2.py`, `tests/compare_params.py`,
  `tests/check_curves.py` and `test_roots <ecm>` compare curves, parameters and roots.

Benchmarks are in `bench/`: `bench_stage2` (complete stage 2), `bench_poly`, `bench_roots`,
`bench_conv` (parts), and the scripts behind docs/BENCHMARKS.md.

## License

GNU Lesser General Public License, version 3 or later (`COPYING.LESSER`, `COPYING`).
Parts are ported from GMP-ECM (LGPL-3+; resume-file parsing follows GMP-ECM's MIT-licensed
resume.c). CGBN is under its own MIT-style license (`third_party/cgbn/LICENSE`). The patches
for GMP-ECM in `gmp-ecm/` modify GPL files of GMP-ECM and are under the GNU General Public
License, version 3 or later, like the `ecm` program they produce (see `gmp-ecm/README.md`).
