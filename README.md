# gpuecm2: ECM stage 2 on the GPU

gpuecm2 runs the stage 2 of the elliptic curve method (GMP-ECM's FFT continuation) on an
NVIDIA GPU. It reads the residues that GMP-ECM saves after stage 1 (`ecm -save`, or
`ecm -gpu -save` for GPU stage 1) and is a drop-in replacement for `ecm -resume`: it chooses
exactly GMP-ECM's stage 2 parameters, covers the same primes, and finds the same factors.
For every curve tested, its results agree with GMP-ECM's bit for bit (including GMP-ECM's
internal "Product of G(f_i)" value), and it finds again 1461 stage 2 factors from logs of real
factoring work.

On an RTX 4070 Ti SUPER a stage 2 at GMP-ECM's default parameters takes 100–200 times less
time than GMP-ECM on one core of an i7-14700F, for numbers of 512 to 4096 bits and B1 from
3e6 up (e.g. 16 ms instead of 2.3 s for a 1024-bit number at B1 = 3e6, 4.0 s instead of 655 s
for 4096 bits at B1 = 11e7); see [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

For a 1024-bit number and the B1 values from GMP-ECM's table (with GMP-ECM's
default B2), on an RTX 4070 Ti SUPER and one core of an i7-14700F the following timings were
obtained, where values are average time for one curve in cases several are batched together:

| Factor size | B1 | B2 (GMP-ECM default) | GPU stage 1 | CPU stage 2 | GPU stage 2 | Stage 2 speedup |
|---|---|---|---|---|---|---|
| 35 digits | 1e6 | 1 045 563 762 | 13 ms | 1.0 s | 7.0 ms | 146× |
| 40 digits | 3e6 | 5 706 890 290 | 39 ms | 2.3 s | 16 ms | 143× |
| 45 digits | 11e6 | 35 133 391 030 | 143 ms | 7.4 s | 50 ms | 148× |
| 50 digits | 43e6 | 240 490 660 426 | 559 ms | 22.2 s | 146 ms | 152× |
| 55 digits | 11e7 | 776 278 396 540 | 1.4 s | 57.0 s | 375 ms | 152× |
| 60 digits | 26e7 | 3 178 559 884 516 | 3.4 s | 118 s | 763 ms | 155× |
| 65 digits | 85e7 | 15 892 628 251 516 | 11.0 s | 290 s | 1.8 s | 160× |
| 70 digits | 29e8 | 90 975 170 616 348 | 37.7 s | 789 s | 4.8 s | 163× |
| 75 digits | 76e8 | 324 909 696 561 468 | 98.8 s | 2418 s | 14.5 s | 167× |

GPU stage 1 is GMP-ECM 7.0.6's `-gpu`; CPU stage 2 is GMP-ECM 7.0.6 on one core; GPU stage 2
is gpuecm2 with the curves of a GPU stage 1 run processed together. From B1 = 26e7 on, both
programs ran with `-maxmem 8000`. Other sizes (512 to 4096 bits, the same speedup of 100–200×
from B1 = 3e6 on) are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

It is also a library with a C interface for programs that do stage 1 themselves; see
[docs/INTEGRATION.md](docs/INTEGRATION.md) for how GMP-ECM can use it.

## Usage

```
gpuecm2 [-v] [-q] [-k n] [-power n | -dickson n] [-maxmem MB] [-no-ntt | -ntt] [-one]
        [-selfcheck] [-batch n] [-spin] -resume <file> B1 [B2 | B2min-B2max]
```

The options mean what they mean for `ecm` (stage 2 only). In addition:

- `-v -v` prints GMP-ECM's "Product of G(f_i)" value when no factor is found; `-v -v -v` adds
  per-phase times (measuring them serialises the GPU work, so step 2 gets slightly slower).
- `-selfcheck` verifies each curve's stage 2 at random points (product trees, and the final
  values at a few roots of F), at a cost of a few percent. Off by default.
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
resume.c). CGBN is under its own MIT-style license (`third_party/cgbn/LICENSE`).
