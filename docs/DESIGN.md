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
    From 96 limbs (3072 bits) on, two passes: one block per tile of output limbs and
    group of coefficients (the tile index varying fastest, so that the blocks reading the same
    residues run together), the column sums to a scratch buffer, then one thread per
    coefficient for the carries and the reduction. With one block running over all tiles the
    residues of the blocks in flight no longer stay in L2 above about 12000 bits (L2 hit rate
    39% at 32768 bits against 91% at 12288), and the one-dimensional grid has few blocks when
    dF is small: the two passes are 25% faster in the conversion itself at 24576-32768 bits
    (1.1 µs instead of 1.46 µs per coefficient at 32768) and up to 44% faster in the whole
    stage 2 at small dF (32251 bits, B1 = 1e5). They were first used from 576 limbs, where
    they are faster for every batch size; below that they win by much less when a batch
    gives the conversions many coefficients (whole stage 2 with GMP-ECM's parameters and
    automatic batches: 1.00 at 2560 bits, 1.01 at 3072-4096, 1.02-1.03 at 6144-8192,
    1.04-1.08 at 10240-16384, 0.96 at 2048) than with few (polynomial part for one curve and dF = 2048-4096: 1.15 at 4096
    bits, 1.7 at 8192, 2.1 at 16384), so the threshold is now 96 limbs.
    `bench/bench_conv` measures both directions.
  - Kernels are kept short (tens of ms; Windows' watchdog, a display on the GPU) at large
    sizes: the conversions run in slices of about 2^35 / (P nl) coefficients, the doubling
    chain of the fixed-base tables and the fixed-base multiplications in ranges of windows
    (at 32768 bits a single launch of the chain took 140 ms with S = 3, and grows with the
    scalar size). Smaller sizes are not split in practice.
  - Tried and rejected (M6): the same products as int8 tensor-core GEMMs (bytes of limbs ×
    bytes of constants, exact in int32). With the short inner dimension (4 × limbs) a WMMA
    kernel reached only 13–22 T MAC/s, no faster than the CUDA-core kernels.
- **RNS storage (the default from 3072 bits on; `-rns` / `-no-rns` for all sizes or none):**
  the polynomial arrays hold, instead of limbs, residues modulo a second set of the NTT
  primes, so that the conversions around each
  product become base extensions between residue systems (`src/gpu/rns.h`, `rns.cu`).
  - The primes are split into base A (the first P_A) and base B (the other P_B), with
    M_A > 2^15 L N and M_B > 2^9 N (P_A + P_B is P to P + 2). A coefficient c is stored as
    the residues mod the primes of B of an integer x ≡ c·M_A (mod N), 0 ≤ x < 64 N
    (Montgomery form with respect to M_A, redundant): P_B words instead of nl, about 6% more
    memory per polynomial.
  - Load (for an NTT): x is extended exactly from B to A as in the limb CRT
    (x = Σ σ_j M_B/b_j − α M_B, α = ⌊Σ σ_j/b_j + 1/4⌋, exact for x < M_B/4); the residues of B
    are copied. Store (a product z, an integer below 2·L·(64 N)²): Montgomery reduction
    r = (z + q N)/M_A with q = −z N⁻¹ mod M_A, extended from A to B approximately and from
    below (q' = q or q + M_A), so r < z/M_A + 2N < 2.25 N. Both are matrix products of
    P_A × P_B multiply-adds per coefficient, half of P × nl, with α accumulated in the same
    tiles.
  - The polynomial algorithms are unchanged. The constant 1 (the leading coefficients of
    monic operands, the wrapped term cancelled in the product tree) becomes M_A mod N
    (`RnsTables::one`); a difference is x + 16N − y, a negation 16N − y (the operands of
    differences are fresh products, < 4N; the largest stored value, after G − F and the
    wrapped reduction mod F, is below 24 N). Roots from CGBN are imported (limbs → RNS of
    all primes, × (M_A² mod N), store) and the evaluated values exported (load,
    × (M_A⁻¹ mod N), limb CRT), both in place; the self-check reads exported copies. The
    results are identical to the limb path (tests/test_stage2 compares the products).
  - Polynomial part, `bench/bench_poly` (dF = 2048-4096, best of 3, limbs/RNS): 1.00× at 1024
    bits, 1.30× at 2048, 1.47× at 4096, 1.78× at 8192, 1.67× at 16384, 1.73× at 32768; with
    dF = 8192 1.55-1.66× from 8192 to 32768 bits. At 1024 bits the conversions are limited by
    memory traffic (each moves about as many words as before).
  - Whole stage 2, `bench/bench_stage2` (GMP-ECM's parameters, automatic batch, best of 3,
    limbs/RNS): 1.07 and 1.05 at 1024 bits (B1 = 1e6, 11e6), 1.08/1.06 at 1536, 1.06/1.04 at
    2048, 1.15/1.12 at 3072 (1e6, 3e6), 1.23/1.17 at 4096, 1.16 at 6144 (1e6), 1.27/1.26 at
    8192 (1e5, 1e6), 1.29 at 10240 (1e5), 1.24 at 12288 (1e5), 1.25/1.25 at 16384 (1e4,
    1e5), 1.27 at 24576 (1e5), 1.27/1.28 at 32768 (1e4, 1e5). At 8192 bits and B1 = 1e6 the polynomial phases go from 68.8 to 47.5
    ms per curve, the root generation (29 ms) is unchanged; the conversions are still 2.6
    times the NTTs.
  - Memory and the default (measured with the Z/N levels, GMP-ECM's parameters, peak of the
    GPU memory pool, limbs → RNS storage): up to 2048 bits 0–5% more (one curve at 11e7:
    375 → 392 MB at 512 bits, 741 → 774 MB at 1024, 2428 → 2517 MB at 2048), from 4096 bits
    the same or less (one curve 4–10% less: 1095 → 1012 MB at 4096 bits, B1 = 11e6; 637 → 570
    MB at 16384 bits, 1e6; batches about equal), because the limb path's two-pass conversion
    needs a scratch buffer and RNS storage's reduction does not. Stage 2 with RNS storage:
    0.94–1.04 at 512 bits, 0.95–1.06 at 1024, 0.99–1.09 at 2048, 1.08–1.19 at 4096,
    1.17–1.22 at 8192, 1.19–1.21 at 16384, 1.23–1.25 at 32768 bits (B1 = 1e4 to 11e7; at
    32768 bits and B1 = 1e4, dF = 256, 0.91: too few coefficients for the extension
    kernels). Hence the automatic choice from 3072 bits.
  - Tried and rejected: summing two products in 64 bits before each 96-bit addition in the
    tiles (fewer additions per multiply-add): 4-20% slower in all conversion kernels.
- **Low levels mod n** (`ZnLevels`; an idea from the GPU stage 2 of Greg Childers' GMP-ECM
  fork): the nodes of the product trees up to degree 2^z, and the matching levels of the
  transposed evaluation, are multiplied with schoolbook products mod n (CGBN Montgomery
  products, `k_zn_tree` and `k_zn_mp` in `prod.cu`) instead of conversions and transforms.
  One thread group per output coefficient: a node of degree 2m costs m/2 products per
  coefficient on average plus one product by R² (the sum of Montgomery products a_i b_j / R),
  the middle products m + 1 per value. Those levels' arrays are limbs with both storages:
  `tree_build` takes the roots as limbs and imports where the RNS levels begin, and
  `poly_eval_tree` gives limb values. The kernels are bound by the per-output work (about
  1 ns per product and 1.5 ns per level and coefficient at 1024 bits, against about 6 ns per
  coefficient for a level through the RNS), so the level of degree 16 still pays at 1024
  bits and the next does not; the RNS levels get relatively cheaper as n grows (and with RNS
  storage), so fewer Z/N levels pay. Default z (measured at GMP-ECM's parameters, B1 = 1e5
  and 1e6): with limbs 4 up to 6144 bits, 3 above; with RNS storage 4 up to 2048 bits, 3 up
  to 6144, 2 up to 12288, 1 above. Stage 2 gets 4–12% faster with limbs (500–32768 bits:
  7–12% at 1000–6000 bits, 4.5% at 32768) and with RNS storage 3–8% up to 6144 bits and 1–2%
  above. `GPUECM2_ZN_LEVELS` overrides z (0: none).
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

Measured or estimated on the RTX 4070 Ti SUPER. The first one is now implemented (RNS
storage, above); the others are not.

- **All-RNS polynomial arithmetic** (Montgomery reduction in RNS, Kawamura et al. /
  Bajard–Imbert). Estimated at 7–20% of stage 2 with about twice the memory for the stored
  polynomials; implemented as RNS storage (`-rns`, see Arithmetic), which keeps only one of
  the two bases and so needs about 6% more memory, and gives 4–29% depending on the size.
- **Curve arithmetic in RNS as well** (the roots of F and G with RNS Montgomery products
  instead of CGBN; removes CGBN, its 32768-bit limit and the per-size kernels). An RNS
  Montgomery product needs two base extensions; measured with the extension kernels of RNS
  storage (65536 values at a time, the whole GPU): 2.9 ns per product at 1024 bits, 7.9 at
  2048, 20 at 4096, 71 at 8192, 267 at 16384 and 970 ns at 32768, against CGBN's Montgomery
  product at 0.68, ≈2.7, 9.7, 32, 130 and 495 ns: 2–4.3 times as expensive (both do about
  2 nl² multiply-adds; CGBN's carry chains run at 3–4.2 T multiply-adds/s, the 96-bit sums of
  the extensions at 0.8–2.3 T). Lazy reduction saves little in Edwards additions (almost
  every product feeds another product). With the roots at 29–47% of stage 2 (RNS storage,
  GMP-ECM's parameters, 1024–32768 bits), stage 2 would be about 1.4–1.55 times slower from
  2048 bits on and 2.2 times at 1024 bits, with 10–20% more memory (the root generation's
  values take P ≈ 2 nl words). Not worth it for speed; it would only pay for portability (HIP)
  and build time.
- **Root arithmetic by transforms for large N** (as in Greg Childers' GMP-ECM fork,
  `gpu_nmul.cuh`: numbers as about 17-bit digits, products by NTTs modulo two 30-bit primes,
  Montgomery with R = B^d − 1, no zero padding, transforms of an operand reused within a point
  addition; it reports stage 2 1.66–1.91 times faster at 32768–65536 bits than its schoolbook
  roots, from about 12–16 kbit on). With our roots at 47% of stage 2 at 32768 bits, a 2–2.8
  times faster root generation would make stage 2 about 1.35–1.4 times faster there. A new
  arithmetic library for the root kernels: a candidate milestone for large numbers.
- **NTT lengths 3·2^k** for dF that is not a power of two (above 1920 bits). Full pairs in
  the product trees always have power-of-two lengths; only the top-level products (products
  mod F, 1/F, the start of the evaluation, one partial node per level) would be shorter: 1–3%
  of stage 2 in half of the cases, nothing in the others. Not worth the extra kernels.
- **The two passes of a four-step NTT on L2-resident data** (transforming groups of rows that
  fit in the 48 MB L2 one after the other, so that the second pass reads from L2): would save
  up to one DRAM round trip per transform for T ≥ 2^16; needs a prime offset in the kernels.
  Not tried; at most 10–20% of the NTT time.
- **Two products summed in 64 bits** before each 96-bit addition in the conversion tiles
  (fewer additions per multiply-add): tried, 4–20% slower in all conversion kernels.
- **The Z/N levels with wide products** (full products summed, one Montgomery reduction per
  output instead of one per product): tried, the same speed (the levels are bound by the
  per-output work, not by the products). Karatsuba for the larger nodes (27 instead of 64
  products for degree 8 × 8) would need a kernel per node with shared-memory sums: estimated
  2–3% of stage 2 at 1024 bits. Keeping the middle Z/N levels in Montgomery form (no product
  by R² per output) about 0.5%.
- **The tree cache by default** (the fork keeps its transforms of F's tree in the free GPU
  memory by default): with the Z/N levels `-treecache` still gives 1–4.4% at 1000–1500 bits
  (B1 = 1e6, 11e6), as before. It stays an option, since it takes most of the GPU memory left.
- **Larger automatic batches** (the fork runs 256 curves at a time at 1000 bits, B1 = 1e5;
  our target is 2^17 coefficients, 128 curves there): twice the automatic batch gains 1–3%
  at small dF and nothing at B1 = 1e6; not changed. (At 8000 bits, B1 = 1e5, twice the
  automatic batch, forced in `bench_stage2`, exceeds the GPU memory and runs 11 times slower;
  the automatic choice and `-batch` are capped by the memory estimate.)

## Robustness and limits

- Kernels run ≲ 100 ms each, to stay under the Windows TDR and display watchdogs.
- Sizes: moduli up to the largest size of the build (`GPUECM2_MAX_BITS`: 8192 bits by default,
  32768 at most); a residue of a larger number is skipped with a message. No limit on B2 or B2min: the G plan works with offsets from i0,
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
| M9 | Large moduli, step 1: up to 32768 bits (CGBN's limit) in the current design, gpuecm2 and GMP-ECM's GPU stage 1 (more kernel sizes only). **Done:** CGBN sizes per translation unit (`GPUECM2_MAX_BITS`), a memory estimate from the allocations, kernels split for the watchdog, a two-pass RNS-to-limbs conversion and memory operands in `k_step` for large sizes; tests up to 32768 bits against GMP-ECM (boundary tests, `tests/large_tests.py`, 51 stage 2 hits of repunit cofactors). Stage 2 is 80-190 times one CPU core from B1 = 1e5 (docs/BENCHMARKS.md). |
| M10 | All-RNS polynomial arithmetic: Montgomery reduction in the RNS by base extensions instead of conversions to and from limbs. **Done** as RNS storage (identical results), the default from 3072 bits on (`STAGE2_RNS_AUTO_BITS`), where stage 2 gets 8–25% faster with the same or up to 10% less GPU memory; below, where it gains or loses a few percent for up to 5% more memory, limbs stay the default (`-rns` / `-no-rns`, `GPUECM2_RNS=1` / `0` force either); and the two-pass RNS-to-limbs conversion from 3072 bits on. |
| M11 | Ideas from Greg Childers' GMP-ECM GPU fork (October 2026): the low levels of the trees and of the evaluation mod n (**done**, 1–12% depending on size and storage); root arithmetic by transforms for large N and the curve arithmetic in RNS evaluated (see Ideas evaluated). Comparison of the two in docs/BENCHMARKS.md. |
| M12 | (Planned.) Root generation with affine Weierstrass additions and one batched inversion per progression step, GMP-ECM's own method: a step is about 10^5 independent additions (all curves, progressions and table entries), so Montgomery's trick over all of them needs one modular inverse per step, about 3 products per addition, and an affine addition 2 products and a square: about 6 products instead of about 10 (Edwards, extended coordinates), and the roots come out as x directly (no batched inversion and conversion per block). Estimated 1.7–2.5 times fewer products for the roots (x^1: 15 → 6 per root, Dickson(6): 65 → 36, Dickson(30): 305 → 180), stage 2 about 1.15–1.35 times faster; half the memory for the difference tables; and the factors GMP-ECM finds while setting up the roots ("Found factor while computing fd[]") would be found the same way. Costs about three kernel phases per step instead of one. |

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
