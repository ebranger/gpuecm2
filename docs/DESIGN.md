# gpuecm2: design notes

A stand-alone program that reads GMP-ECM stage-1 save files and runs ECM stage 2
(the FFT/polynomial continuation) on an NVIDIA GPU, as a drop-in replacement for
GMP-ECM's CPU stage 2.

## Requirements

- **Drop-in:** for any B2 that is reasonable with CPU GMP-ECM, cover exactly the same primes
  (same `bestD`/`choose_S` parameters: d, d2, dF, k, i0, Dickson S, B2min).
  Same default B2 for a given B1, same output lines and exit codes.
- **Input:** GMP-ECM resume files (`METHOD=ECM`, PARAM 0–3 and generic A/X Montgomery curves).
  N may be an expression.
- **Size:** N up to 32768 bits, CGBN's largest size (it supports at most TPI = 32 threads of
  TPI limbs each). Larger N would need another big-number arithmetic for the roots and
  products, and more or larger NTT primes (64-bit primes) for long polynomials.
- **Hardware:** NVIDIA, compute capability 7.5+ (Turing and newer); must build with CUDA 12 and 13.
  Keep code HIP-portable where cheap.
- **OS:** Linux first. Host code must also build with MSVC (Windows): CMake, no `__int128`,
  no POSIX-only APIs.
- **Dependencies:** GMP (host) and CGBN (header-only, vendored, pinned). Nothing else at runtime.
  CUB/Thrust from the CUDA toolkit are allowed. GMP-ECM and FLINT are test oracles only.
- **License:** LGPL-3.0-or-later. This allows reuse of GMP-ECM's LGPL stage-2 code
  (stage2.c, ecm2.c, bestd.c, ecm.c, polyeval.c, median.c, mpzspv.c, ...) and the MIT-licensed
  resume.c. Do NOT copy eval.c, main.c, auxi.c, cudawrapper.c or cgbn_stage1.cu, which are GPL;
  use them as inspiration only.

## Architecture

```
cli/        gpuecm2 executable: argument parsing, resume-file I/O, output/exit codes
host/       N context (prime selection, CRT constants), stage-2 parameter selection
            (ported from GMP-ECM), curve setup (sigma/PARAM -> A, x; Montgomery -> Weierstrass),
            expression evaluator (own), CPU reference implementations for tests
gpu/        kernels: RNS NTT, conversions, pointwise ops, poly mul / middle product,
            product tree, PolyInvert, PrerevertDivision, transposed polyeval, EC roots (CGBN)
api/        Stage2Engine (C++), and the C interface gpuecm2.h (libgpuecm2): engine per
            number and parameters, stage 2 for a batch of curves -> per-curve result
tests/      unit tests (vs GMP reference), differential tests (vs GMP-ECM binary)
third_party/cgbn (pinned copy)
```

The C interface is shaped so that GMP-ECM can call it from its per-curve stage 2 loop
(`cudawrapper.c`) in place of `stage2()`, with all curves of the GPU stage 1 at once, or from
`ecm()` for one curve; see docs/INTEGRATION.md.

## Arithmetic

- **Residues mod N:** 32-bit limbs, structure-of-arrays (limb-major) for batches.
- **Polynomials over Z/N:** RNS over P NTT primes 2^28 < p < 2^31, p = c·2^k + 1.
  - Keeping p < 2^31 allows 32-bit Shoup and Montgomery arithmetic without overflow checks.
  - The NTT length is 2^k with k = ceil(log2(2 dF)) (at least 12), unless there are not enough
    primes of the form c·2^k + 1 in the range for this N: then the largest k with enough
    primes is used (2^19 is possible up to 4608 bits, 2^20 up to 2520 bits, 2^21 up to 1186
    bits, 2^22 up to 483 bits) and longer products are split (below).
  - P = smallest count with ∏p > 2·L·N², where L is the longest operand (exact CRT with the x < M/2 trick).
  - Products longer than 2^k are split into pieces of 2^(k−1) coefficients.
    The piece products are summed in the NTT domain and overlap-added before one CRT (option a).
  - Primes, twiddles and CRT tables are computed once per N and shared by all curves.
- **NTT:** batched four-step / radix-8 kernels with twiddles in shared memory
  (prototype: gpu-s2-bench/ntt31v2.cu, 2.2–2.6e11 butterflies/s on an RTX 4070 Ti SUPER).
  - Rows of up to 4096 points (`kRow`, also the second pass of the four-step transforms) are
    done in shared memory with 8 elements per thread. The stride-1 pass (the last of a
    forward transform, the first of an inverse one) holds 8 contiguous elements per thread,
    loaded and stored as two 16-byte vectors; the exchanges between passes use an XOR
    swizzle and the twiddles are stored per level, so there are no bank conflicts. Both
    passes run at about 80% of DRAM bandwidth (Nsight Compute), a transform of 2^18 points
    within 10% of cuFFT's FP32 transform of the same number of bytes.
  - Operands are zero-padded to twice their length: the first pass of the forward transform
    takes the padding (and the leading 1 of a monic operand) as given instead of reading it,
    so it is never written. The pointwise products (plain, adjacent pairs in the product
    tree, sibling swaps in the evaluation) are formed while the first pass of the inverse
    transform loads its input.
- **Conversions:** direct reduction (limbs × table of 2^(32j) mod p) and explicit CRT back to mod N.
  - Both are matrix products over the coefficients, P × nl multiply-adds per coefficient
    each way (P ≈ 2·nl primes), and they dominate the polynomial arithmetic for large N.
  - Limbs → RNS is a tiled matrix product: each thread computes 2–4 primes × 4 coefficients
    from operands staged in shared memory 16 limbs at a time, with registers capped for 3–4
    blocks per SM (latency bound otherwise): 1.7 ns per coefficient at 1024 bits, 3.9 at
    2048, 14 at 4096 (up to 2.3 T multiply-adds/s on an RTX 4070 Ti SUPER).
  - RNS → limbs uses the same tiling for the column sums (k over the primes) of 64
    coefficients, one tile of output limbs at a time: after each tile one thread per
    coefficient adds the sums into the result, carrying from tile to tile in registers,
    and writes those limbs; it then reduces mod N with the quotient estimated in double
    precision (one 64-bit multiply-subtract pass and normally one check). Shared memory is
    one tile of sums whatever the size of N. For 33–48 limbs (1025–1536 bits) a variant that
    keeps the column sums of all limbs is faster, below 32 limbs one dot product per thread.
    2.1 ns per coefficient at 1024 bits, 5.8 at 2048, 22 at 4096, 25 at 4608.
    From 576 limbs (18432 bits) on, two passes: one block per tile of output limbs and
    group of coefficients (the tile index varying fastest, so that the blocks reading the same
    residues run together), the column sums to a scratch buffer, then one thread per
    coefficient for the carries and the reduction. With one block running over all tiles the
    residues of the blocks in flight no longer stay in L2 above about 12000 bits (L2 hit rate
    39% at 32768 bits against 91% at 12288), and the one-dimensional grid has few blocks when
    dF is small: the two passes are 25% faster in the conversion itself at 24576-32768 bits
    (1.1 µs instead of 1.46 µs per coefficient at 32768) and up to 44% faster in the whole
    stage 2 at small dF (32251 bits, B1 = 1e5).
    `bench/bench_conv` measures both directions.
  - Kernels are kept short (tens of ms; Windows' watchdog, a display on the GPU) at large
    sizes: the conversions run in slices of about 2^35 / (P nl) coefficients, the doubling
    chain of the fixed-base tables and the fixed-base multiplications in ranges of windows
    (at 32768 bits a single launch of the chain took 140 ms with S = 3, and grows with the
    scalar size). Smaller sizes are not split in practice.
  - Tried and rejected (M6): the same products as int8 tensor-core GEMMs (bytes of limbs ×
    bytes of constants, exact in int32). With the short inner dimension (4 × limbs) a WMMA
    kernel reached only 13–22 T MAC/s, no faster than the CUDA-core kernels.
- **EC arithmetic for roots:** CGBN behind a thin wrapper, sizes in steps of 256 bits up to
  2048, 512 up to 4608, then 1024 up to 8192, 2048 up to 16384 and 4096 up to 32768 (TPI 4
  at 512 bits, 8 up to 1792 and at 3072, 16 up to 8192, 32 above; chosen by measurement up
  to 8192 bits, above that TPI 32 is the only choice). A modulus of exactly BITS bits uses
  that size. Each size is compiled in its own translation unit (`gen/roots_<BITS>.cu`,
  `gen/prod_<BITS>.cu`, see `src/gpu/cgbn_ops.h`), so the sizes compile in parallel.
  - Above 4608 bits the point addition of the progressions (`k_step`) takes its operands
    from memory where they are used (`padd_mem`): two points and the curve constants no
    longer fit in registers (255 registers and 0.4-1.9 KB of spills per thread from 8192
    bits up, none with `padd_mem` up to 16384 bits); 1-3% faster in the whole stage 2.
  - The curve is GMP-ECM's Montgomery curve, used in its twisted Edwards form
    (a = (A+2)/B, d = (A−2)/B) in extended coordinates: dedicated addition (9M + 1 product
    by a) with the unified formula as fallback, 1.4–2.4× cheaper than projective
    Weierstrass. The roots are converted back to GMP-ECM's Weierstrass x = (3u + A)/(3B),
    so they are the same numbers bit for bit.
  - Dickson/x^S finite-difference tables as in GMP-ECM's `ecm_rootsF`/`ecm_rootsG`,
    started with fixed-base windowed scalar multiplications (one table per curve, shared
    by F and G).
  - One kernel launch per progression step over all (progression, entry) additions,
    double-buffered tables. The number of progressions makes a step a whole number of
    "waves" (concurrent additions, from the occupancy API) and balances the setup
    scalar multiplications against the steps.
  - The output slot of each progression step is the rank of its argument among the wanted
    ones. For F (dF roots) it is a table. For G the arguments along a progression are all
    wanted and their ranks grow by a constant, so the slot is base + n·inc: the plan needs
    O(number of progressions) memory whatever k is, and i0 can have any size (only offsets
    from it are machine integers; B2 up to 1e39 is tested).
  - Projective outputs, one batched inversion per block of dF roots. G roots are produced
    block by block (the progression state is kept), so memory is O(dF) whatever k is.
  - Root generation for block b+1 runs on its own stream while block b's polynomial work
    runs.
- **Batch dimension everywhere** (curves × polynomials × primes), so small-dF runs (small B1)
  are not dominated by launch overhead.

## Stage 2 algorithm (same as GMP-ECM)

For each curve (or batch of curves, see below):
1. Roots of F, then build F's product tree (all levels kept).
2. Compute 1/rev(F) by Newton iteration.
3. For each of the k blocks: roots of G, build G (tree discarded), then H ← G·H mod F (PrerevertDivision).
4. Transposed polyeval of H over F's tree, product of the values, gcd on the CPU.

With `-treecache` (`GPUECM2_TREE_CACHE` in the C interface; off by default because of the
memory: (log2 dF − 2)·(8·P − 4·nl)·dF bytes per curve, 0.8 GB at 1024 bits and dF = 2^17) the
transforms of F's tree levels (the pairs as step 1 multiplied them) are kept for step 4 as
far as the GPU memory left over allows (dF a power of two, nodes of degree 4 and up). On
those levels the evaluation holds its values reversed per node, so that the middle product
needed for a child is coefficients m..2m−1 of (sibling) × (reversed values), with the
sibling's transform as kept: one conversion and one forward transform less per level.

**Curve batching.** For small dF a single curve cannot fill the GPU: kernels have too little
work (or a partly filled last wave), and launch latency dominates. `Stage2Engine::run_batch`
therefore runs B curves with the same N and parameters together. Every array holds the B
curves' data side by side (curve c's n coefficients at c·n, leading dimension B·n), and every
kernel launch covers all B curves: the RNS conversions take a row map (rows grouped per
curve), the polynomial algorithms take a batch count, and root generation keeps per-curve
constants, fixed-base tables and progressions. The output ring of the root generator is laid
out block by block (`[ring slot][curve][blen]`), so one batched inversion covers the block of
every curve (the modulus is shared). The number of progressions per curve is chosen for the
whole batch, which also cuts the per-curve cost of the initial scalar multiplications. If the
shared inversion meets a non-invertible value, the batch is redone curve by curve, so every
curve's result is exactly what a single run would give.

The CLI groups consecutive residues with the same N and parameters (`-batch n`; automatic:
about 2^17·(SMs/66) coefficients per batch, at most 512 curves, within the free GPU memory).
Output is printed in file order once a batch is done; "Step 2 took" is the batch time divided
by the number of curves. When a curve finds a factor, the curves after it in the batch are
redone on the cofactor, which is GMP-ECM's behaviour for consecutive lines with the same N.

## Ideas evaluated (October 2026)

Measured or estimated on the RTX 4070 Ti SUPER during the optimisation of the polynomial
arithmetic; none of them is implemented.

- **All-RNS polynomial arithmetic** (Montgomery reduction in RNS, Kawamura et al. /
  Bajard–Imbert: the primes split into two bases, two base extensions per reduction instead
  of RNS → limbs → RNS). A prototype base extension (half the primes to the other half,
  approximate alpha) takes 0.6–14 ns per coefficient at 512–4608 bits, so one reduction costs
  1.3–28 ns against 1.4–41 ns for the two conversions it replaces. Counting the conversions of
  a whole stage 2 (all but the roots' and the final one become reductions), conversion time
  would drop by 35–48% with the prototype, 55–61% with a base extension as efficient as the
  limbs → RNS kernel: 7–20% of stage 2 at GMP-ECM's default parameters (most for large N and
  small B1). Costs: about twice the memory for stored polynomials (P residues instead of nl
  limbs per coefficient), one or two more primes, and a rewrite of the polynomial layer. A
  candidate milestone; skipping reductions on some tree levels (more primes instead) could
  add to it.
- **NTT lengths 3·2^k** for dF that is not a power of two (above 1920 bits). Full pairs in
  the product trees always have power-of-two lengths; only the top-level products (products
  mod F, 1/F, the start of the evaluation, one partial node per level) would be shorter: 1–3%
  of stage 2 in half of the cases, nothing in the others. Not worth the extra kernels.
- **The two passes of a four-step NTT on L2-resident data** (transforming groups of rows that
  fit in the 48 MB L2 one after the other, so that the second pass reads from L2): would save
  up to one DRAM round trip per transform for T ≥ 2^16; needs a prime offset in the kernels.
  Not tried; at most 10–20% of the NTT time.

## Robustness and limits

- Kernels run ≲ 100 ms each, to stay under the Windows TDR and display watchdogs.
- Sizes: moduli up to 32768 bits, or the largest size the build has (`GPUECM2_MAX_BITS`); a
  residue of a larger number is skipped with a message. No limit on B2 or B2min: the G plan works with offsets from i0,
  and memory is O(dF) on the GPU and O(progressions) on the host whatever k is (tested
  against GMP-ECM with B2min-B2 ranges up to 1e39, and with self-checked runs up to
  k = 2240 blocks).
- GPU memory: GMP-ECM's dF and k are used unless the estimate (`stage2_gpu_bytes`) exceeds
  the free memory reported by `cudaMemGetInfo` (less a margin); then the parameters GMP-ECM
  chooses with the smallest larger `-k` that fits are used. `-maxmem` acts as in GMP-ECM
  (on its CPU memory model, so that both programs choose the same parameters).
- Optional self-check `-selfcheck` (off by default, as decided: hardware errors rarely hit
  the curve that would find a factor, and unstable hardware is usually noticeable anyway):
  each product tree is evaluated at a random point r and compared with ∏(r + root), and the
  final values H(−f_i) at a few random roots of F are compared with ∏ (g_j − f_i) over all
  blocks (reduction mod F keeps the values at F's roots, so this covers the products mod F
  and the transposed evaluation). Those checks use the GPU's roots on both sides, so the root
  generation is checked separately: two roots of F and one root of each block of G of the
  first curve of each batch are recomputed on the CPU (the scalar f(v), an x-only Montgomery
  ladder in GMP, GMP-ECM's Weierstrass x), independently of the GPU's Edwards finite
  differences. Costs 5–20% (measured 11–18% at B1 = 11e7 parameters, of which the CPU
  roots are 6–10%).
- For HIP portability, keep inline PTX in one arithmetic header with portable fallbacks,
  make no warp-size-32 assumptions in our own kernels, and keep CGBN behind a wrapper.
  Review (M6): our kernels use no warp intrinsics and no PTX outside `arith32.cuh` (which has
  C fallbacks); stream-ordered allocation, occupancy queries and launch bounds all have HIP
  equivalents. CGBN is the blocker: its carry chains are PTX (`add.cc`/`madc`) and it relies
  on 32-lane `__shfl_sync`/`__ballot_sync`. A HIP build needs a CGBN port (portable carry
  chains, wave32/64-aware masks) or a replacement for the few operations used
  (Montgomery product, add/sub/compare, conversions) behind `ecc.cuh`.

## Milestones

| Milestone | Content |
|---|---|
| M0 | Project skeleton: CMake (CUDA 12/13, sm_75+), vendored CGBN, GMP, test harness. |
| M1 | Host side: resume parser, expression evaluator, curve setup, ported parameter selection. Must print the same `dF/k/d/d2/i0/S` as `ecm -v` for a grid of N sizes and B1/B2 values. |
| M2 | GPU polynomial engine: prime selection, per-N context, batched NTT for power-of-two lengths 2^6 up to 2^k, conversions, poly multiply with splitting. Unit tests against a GMP reference. **Done.** |
| M3 | Product tree, PolyInvert, PrerevertDivision, transposed polyeval. Validate against the CPU reference, then benchmark the tree. **Done:** on an idle RTX 4070 Ti SUPER the whole polynomial part for c308 (dF = 2^17, k = 5) takes 0.30 s, with one tree at 31 ms. GMP-ECM takes ~45 s on one core and FLINT ~32 s. |
| M4 | EC roots on the GPU (CGBN), validated against GMP-ECM. **Done:** every root of F and G matches GMP-ECM's own (trace output) bit for bit. Uses complete projective formulas, a fixed-base windowed table and one batched inversion. c308 at B1 = 11e7 takes about 0.9 s on the shared GPU, against ~36 s for GMP-ECM on one core. |
| M5 | End-to-end stage 2 and CLI. Differential tests against GMP-ECM: the same curves must find factors in step 2, over N sizes 200–4400 bits and default B2 for B1 from 1e4 to 3e9. Memory auto-sizing and self-check. **Done:** end-to-end results match GMP-ECM exactly on every curve tested, and the 10 logged real hits are reproduced. GPU memory planning raises k when needed, and GMP-ECM's multi-line resume semantics are implemented. Optional `-selfcheck` (off by default; injected faults are caught, see `test_stage2`). |
| M6 | Performance: curve batching for small dF, streams/overlap, tuning, benchmarks against the CPU, a HIP-portability dry run. **Done:** Edwards root arithmetic, per-step kernel, CGBN size ladder, G roots per block, plan and memory-pool tuning, stream overlap, curve batching. Stage 2 is 1.6–2.7× faster than at the end of M5 for one curve (see the table below), and up to 50× for small B1 with batching; 100–200× one CPU core at GMP-ECM's default parameters (docs/BENCHMARKS.md). |
| M7 | Library and tests for publication: C interface (`gpuecm2.h`, `libgpuecm2`), known stage 2 hits from real factoring work, boundary tests at every size and parameter threshold, large-B2 robustness (closed-form G plans). **Done**; see docs/TESTING.md and docs/INTEGRATION.md. |
| M8 | Integration into GMP-ECM (a patch using the C interface in `cudawrapper.c` and `ecm.c`). **Done:** `gmp-ecm/` (option `-gpu2`, `build_ecm.sh`, `test_gpu2.py`). |
| M9 | Large moduli, step 1: up to 32768 bits (CGBN's limit) in the current design, gpuecm2 and GMP-ECM's GPU stage 1 (more kernel sizes only). **Done:** CGBN sizes per translation unit (`GPUECM2_MAX_BITS`), a memory estimate from the allocations, kernels split for the watchdog, a two-pass RNS-to-limbs conversion and memory operands in `k_step` for large sizes; tests up to 32768 bits against GMP-ECM (boundary tests, `tests/large_tests.py`, 40 stage 2 hits of repunit cofactors). Stage 2 is 80-190 times one CPU core from B1 = 1e5 (docs/BENCHMARKS.md). |

### M6 benchmarks (idle RTX 4070 Ti SUPER)

Complete stage 2 for one curve (`bench_stage2`, random curve, engine reused, no per-phase
sync); "end of M5" is the same benchmark before the M6 work.

| N | B1, B2 | polynomial | end of M5 | now |
|---|---|---|---|---|
| RSA-200 (663 bits) | 5e7, 2.4e12 | Dickson(30) | 740 ms | 417 ms |
| RSA-250 (829 bits) | 15e6, 2.4e11 | Dickson(12) | 217 ms | 121 ms |
| RSA-300 (995 bits) | 11e7, default 7.8e11 | Dickson(30) | 563 ms | 368 ms |
| RSA-300 (995 bits) | 3e6, default 5.7e9 | Dickson(6) | 49 ms | 21 ms |
| RSA-1024 | 11e7, default | Dickson(30) | 996 ms | 376 ms |
| RSA-1536 | 11e7, default | Dickson(30) | 1505 ms | 743 ms |
| RSA-2048 | 11e7, default 9.0e11 | Dickson(30) | 2916 ms | 1170 ms |
| RSA-2048 | 3e6, default 4.6e9 | Dickson(6) | 185 ms | 51 ms |
| RSA-2048² × RSA-100 (4425 bits) | 3e6, default | Dickson(6) | — | 180 ms |

Against GMP-ECM 7 on one core of the i7-14700F (same resume lines and parameters,
"Step 2 took", gpuecm2 per curve with the engine reused):

| N | B2 | polynomial | GMP-ECM | gpuecm2 | ratio |
|---|---|---|---|---|---|
| RSA-300 | 7.8e11 | Dickson(30) | 67.7 s | 0.43 s | 158 |
| RSA-300 | 5.7e9 | Dickson(6) | 2.89 s | 25 ms | 115 |
| RSA-2048 | 4.6e9 | Dickson(6) | 7.2 s | 65 ms | 111 |
| c200 (logged hit) | 2e12 | Dickson(30) | 237 s | 0.60 s | ~400 |
| c167–c288 (logged hits) | 2e11 | Dickson(12) | 38–85 s | 0.13–0.30 s | ~300 |

### Curve batching (M6)

Steady-state time per curve with the engine reused (CLI on 256 residues of the same N; batched
= `-batch n`, automatic size in the last column). Idle RTX 4070 Ti SUPER apart from light
desktop use.

| N | B1 | dF, k | one at a time | batched | automatic batch |
|---|---|---|---|---|---|
| 512 bits (RSA-155) | 11e3 | 256, 3 | 5.3 ms | 0.10 ms (B = 128) | 256 |
| 1024 bits (RSA-1024) | 5e4 | 512, 5 | 8.4 ms | 0.55 ms (128) | 256 |
| 2048 bits (RSA-2048) | 11e3 | 288, 2 | 10.1 ms | 0.57 ms (128) | 256 |
| 1024 bits | 25e4 | 2048, 3 | 9.5 ms | 1.8 ms (32) | 64 |
| 4096 bits | 25e4 | 2880, 2 | 43.5 ms | 20 ms (128) | 45 |
| 512 bits | 3e6 | 16384, 2 | 13.5 ms | 6.8 ms (8) | 8 |
| 1024 bits | 3e6 | 16384, 2 | 22.4 ms | 16.0 ms (64) | 8 |
| 2048 bits | 3e6 | 14400, 2 | 52.2 ms | 41.0 ms (64) | 9 |
| 1024 bits | 11e6 | 32768, 3 | 54.5 ms | 48.0 ms (64) | 4 |

For tiny dF the single-curve time is almost all latency (kernel launches, small kernels that
use a fraction of the GPU, host round trips in the root inversions), and batching removes
most of it. Around B1 = 3e6 a single curve's kernels already have thousands of threads; the
gain of 1.3–1.4× there comes from partly filled last waves and launch overhead. From
B1 ≈ 1e7 on (dF ≥ 32768) one curve nearly fills this GPU. Larger GPUs need more parallel
work, so the automatic batch size scales with the number of SMs.
