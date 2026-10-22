# Testing gpuecm2

gpuecm2 must give exactly GMP-ECM's results. The tests check this at three levels: unit
tests of the parts, a set of real stage 2 factors that must be found again, and differential
tests that compare every curve with GMP-ECM, with emphasis on the places where the
implementation changes behaviour (boundary tests).

| Test | Needs | Time | What it checks |
|---|---|---|---|
| `ctest` (unit tests) | GPU | 1 min | arithmetic, NTT, conversions, polynomial algorithms, curve formulas, root plans, stage 2 engine with self-check and fault injection, C interface |
| `tests/run_hits.py` | GPU | 1-2 min | known stage 2 factors (`tests/data/stage2_hits.txt`) are found again, via the program and via the C interface (`capi_resume`) |
| `tests/run_hits.py ... tests/data/stage2_hits_large.txt` | GPU | 1 min | the same for known stage 2 factors of 4609- to 32768-bit numbers (those above the largest size of the build are skipped: all need `-DGPUECM2_MAX_BITS=32768`) |
| `tests/large_tests.py` | GPU; GMP-ECM once | `prepare` about 15 CPU core-hours (once), `check` 2 min | same results as GMP-ECM on cached residues of 4609- to 32768-bit numbers, B1 = 1e4 to 11e6 (cases above the largest size of the build are skipped) |
| `tests/input_tests.py` | GPU | 10 s | improper input (bad bounds and options, malformed or inconsistent resume lines, N even, too small or too large, unsupported curves) is refused with a message, nothing crashes, and the other residues of a file still run |
| `tests/boundary_tests.py` | GPU, GMP-ECM | 10-15 min (`--heavy`: +1 h) | same results as GMP-ECM on both sides of every size and parameter threshold |
| `tests/compare_stage2.py` | GPU, GMP-ECM | 40 min | same results as GMP-ECM for random curves on numbers of 330 to 4425 bits, alone and in batches, including GMP-ECM's handling of consecutive residues after a factor |
| `tests/compare_params.py` | GMP-ECM | 10 min | the parameter lines (`dF`, `k`, `d`, `d2`, `i0`, polynomial) for a grid of sizes, B1, B2 and options |
| `tests/check_curves.py` | GMP-ECM | 5 min | the curve setup of parametrizations 0-3 |
| `test_roots <ecm>` | GPU, GMP-ECM | 1 min | every root of F and G against GMP-ECM's own (`-v -v -v -v -v`), also with G produced block by block |
| `gmp-ecm/test_gpu2.py` | GPU, the `ecm` of `gmp-ecm/build_ecm.sh` | 3 min | the full GPU build of GMP-ECM: `-gpu2` gives the same result as GMP-ECM's CPU stage 2 for every curve of a GPU stage 1 run (512, 1024 and 2048 bits; `--large`: also 7679 and 32251 bits, 5 min more, for builds of gpuecm2 and GMP-ECM up to 32768 bits) |

"Same results" means the same factor, or the same `Product of G(f_i)` value that GMP-ECM
prints at `-v -v` when no factor is found. That value is H(-f_0) = the product over all
blocks and all roots g_j of G of (g_j - f_0). It depends on f_0, on every root of G and on
the polynomial products, inversions and reductions (a wrong coefficient in them changes it),
but not on the other roots of F: a wrong root f_i (i > 0) that is used consistently leaves
it unchanged. Those are covered by `test_roots <ecm>`, which compares every root of F and G
with GMP-ECM's own and, for the smaller cases, compares gpuecm2's product of H(-f_i) over all
roots of F (whose gcd with N is the result) with the product of (g_j - f_i) computed from
GMP-ECM's roots; and, in production runs, by the self-check, which recomputes some roots on
the CPU.

The scripts that need GMP-ECM take its path as an argument (`ecm` 7.x; the tests were run with
7.0.6) and limit its memory use. Nothing is run in parallel.

## Known stage 2 hits

`tests/data/stage2_hits.txt` holds stage 1 residues of curves that found a factor in stage 2
in real factoring work, with the logged B2, polynomial and factor. Most come from the
contributed factorizations on Makoto Kamada's near-repdigit pages (stdkmd.net/nrr); each
line names its source. The residues were made with GMP-ECM 7.0.6 from the logged sigma and
B1, so the test does not need GMP-ECM.

They were chosen from 1457 logged step 2 hits (50 to 575 digits, B1 from 1e4 to 29e8,
parametrizations 0, 1 and 3, x^1, x^2 and Dickson(6/12/30)) to cover every B2 range,
polynomial and parametrization that occurs, and the full range of sizes. gpuecm2 was run on
all 1457: it finds 1449 of the factors with the logged B1, B2 and polynomial, and two more
with the option under which GMP-ECM 7.0.6 also finds them (`-no-ntt`, `-k 3`; these two are in
the file). The other six are not found by GMP-ECM 7.0.6 either, from the same residue with
the same parameters: the logged runs used other versions (7.0.4, 7.0.5-dev; one unstated) whose stage 2
parameters differ, and these curves found their factor through the parts of the stage 2 that
depend on the parameters (the effective B2 above the requested one, and the Brent-Suyama
extension).

```
python3 tests/run_hits.py build/gpuecm2
python3 tests/run_hits.py build/capi_resume      # the same through the C interface
```

## Large numbers

Above 4608 bits GMP-ECM's CPU runs take long (a stage 1 with B1 = 1e6 takes half an hour
at 32768 bits), so the comparisons with GMP-ECM at these sizes are split in two:

- `tests/large_tests.py prepare <ecm>` makes stage 1 residues of numbers at each CGBN size
  above 4608 bits (4609, 5120, ..., 32768, and one bit more at some) for B1 = 1e4, 1e5 and
  1e6, up to 16385 bits also 3e6 and up to 10240 bits 11e6, with parametrizations 0-3, and
  runs GMP-ECM's stage 2 on each with its default B2 and other stage 2 settings
  (Dickson(12), x^1, 100 times the default B2, `-k 5`, Dickson(30)), all stored in a cache
  directory: the residues, GMP-ECM's "Product of G(f_i)" or factor, its stage 2 parameters
  and CPU times. The numbers are products of primes of about 1024 bits and a planted
  smaller prime, close to 2^bits. Runs in parallel (`--jobs`), memory-capped.
- `tests/large_tests.py check <gpuecm2>` runs gpuecm2 on the cached residues (all residues of
  a case in one batch, or `--one-by-one`) and compares every value and parameter line;
  `bench` reports gpuecm2's stage 2 time against GMP-ECM's.

`tests/data/stage2_hits_large.txt` holds stage 2 hits on numbers of 4609 to 32768 bits,
for `run_hits.py`: ECM factors of repunit cofactors reported on Makoto Kamada's repunit
pages (stdkmd.net/nrr/repunit/, the monthly change pages, which give B1 and sigma). The
composites were rebuilt from Phi_n(10) and the factor lists (checked against the reported
digit counts and leading digits); sigmas without a parametrization prefix are GMP-ECM 6's
and Prime95's parametrization 0. Of the 2147 reported ECM factors of composites of 4609 to
32768 bits there, 1828 are found by GMP-ECM 7.0.6 in step 2 with the logged B1 and sigma
(1526 with its default B2, 302 only with 100 times it), 200 in step 1, and 119 not at all
(99 of them 2024-2026 results with sigmas without a prefix, from a program with another
parametrization or a much larger B2). The 51 hits in the file were chosen from the step 2
ones to cover every CGBN size class from 5120 to 32768 bits (3-6 each), B1 from 1e4 to 11e6,
parametrizations 0, 1 and 3, x^1, x^2 and Dickson(3, 6, 12, 30), and default and larger B2
(9 with 100 times the default, 7 of them with `-maxmem 4000`, under which GMP-ECM's stage 2
needs 3.4-3.7 GB instead of 4-13 GB and gpuecm2 chooses the same parameters); for each,
GMP-ECM 7.0.6 found the factor in step 2 on the full composite. `default` in the
B2 field stands for GMP-ECM's default (passing its effective B2 explicitly can give other
stage 2 parameters).

```
python3 tests/run_hits.py build/gpuecm2 tests/data/stage2_hits_large.txt
```

## Boundary tests

`tests/boundary_tests.py` runs both programs on both sides of each point where gpuecm2 changes
its code path or a discrete parameter, so that off-by-one errors and edge effects show up:

- Modulus sizes: each size of the CGBN ladder (512, 768, ..., 2048, 2560, ..., 4608, 5120, ...,
  8192, 10240, ..., 16384, 20480, ..., 32768 bits) and one bit more; GMP-ECM's NTT threshold
  (1920/1921 bits, where dF stops being a power of two); the switches of the RNS-to-limbs
  conversion tile (2254/2258 and 4576/4577 bits); sizes where the RNS primes leave no spare bit
  over the CRT bound (e.g. 1017, 1916, 4008, 4504 bits), with N close to 2^bits; tiny moduli
  (64 bits); 4608 bits at small and larger dF; the largest size of the build (8192 bits by
  default, 32768 at most; `gpuecm2 -h` shows it, and the sizes above it are left out), and one
  bit more must be refused. Above 4608 bits the numbers are products of primes of about 1024
  bits (larger primes take too long to find); `--heavy` adds dF = 16384 at 8193 and 16385
  bits (or at the largest size of the build).
- NTT length: products longer than the longest NTT for the modulus are split, and the product
  tree, inversion and evaluation switch to exact-product fallbacks. `GPUECM2_MAX_LOGT` forces
  short NTTs at small sizes (2^12..2^14 with dF = 16384); `--heavy` also tests the natural
  switches, where the number of NTT primes runs out for the longest NTT (dF = 2^21 at 483/484
  bits, 2^20 at 1186/1187 bits, dF = 362880 at 2520/2521 bits; GMP-ECM needs about 6 GB and
  5 minutes for each).
- Stage 2 parameters: x^1, x^2, x^3, x^6, x^7, Dickson(3, 5, 6, 12, 30, 60); k = 1, 2, 7, 50;
  tiny B2 (dF = 2..64, d2 = 1); B2 = B1 and B2 < B1 (no stage 2); B2min-B2 ranges, a single
  point, and ranges at 2^64 and 10^39 (i0 of 113 bits); `-no-ntt` below and `-ntt` above the
  NTT threshold; parametrizations 0-3.
- Large B2: full B2 of 100 times the default, 1e12 and 1e13 (`--heavy`), against GMP-ECM.
- Batches: batch sizes 1, 2, 3, 7, 64, automatic and 512 on 520 residues of one number, also
  with non-power-of-two dF and with larger dF; a GPU memory budget (`GPUECM2_GPU_MEM`) that
  cuts the batch size; a budget too small for GMP-ECM's dF, where gpuecm2 uses more blocks
  (compared with GMP-ECM run with the same `-k`).

```
python3 tests/boundary_tests.py build/gpuecm2 /path/to/ecm [--heavy] [--only REGEX]
```

## Long runs

Stage 2s too large for GMP-ECM on this machine were run with `-selfcheck` (product trees and
final values checked at random points; these runs were made before the self-check also
recomputed roots on the CPU):

- 4095 bits, B1 = 25e9 (default B2 = 1.8e15, `-maxmem 8000`): dF = 259200, k = 2240 blocks;
  self-check passed, same "Product of G(f_i)" as an earlier run with the previous root
  plans; 84 minutes (with other GPU work running), 217 MB of host memory.
- 512 bits, B2 = 1e17: GMP-ECM's dF = 2^23 does not fit in the GPU memory, so dF = 2^21 with
  k = 2140 blocks (4.5e9 roots of G); self-check passed, 19.5 minutes, 257 MB of host memory.

## Improper input

`tests/input_tests.py` (also run by ctest) checks 64 kinds of improper input. Options and
bounds must be complete numbers in range (B1 up to GMP-ECM's 2^53 - 1, finite B2, k >= 1,
polynomial degree 1 to 1000, -maxmem and -batch not negative); otherwise gpuecm2 stops
before reading the file. A resume line that cannot be used is skipped with a message and
the other lines are processed: unknown fields, missing fields, bad checksum, N not an odd
number above 1 (or above the largest size of the build, 8192 bits by default, or an invalid
expression), a B1 field that is not a finite non-negative number, PARAM outside 0-3,
invalid sigma, non-ECM methods and non-Montgomery curves, and singular curves (A^2 = 4). The
C interface refuses stage 2 parameters that GMP-ECM would never make and reports singular
curves per curve (`test_capi`).

For tiny numbers (below about 2^20) the result can differ from GMP-ECM's in which gcd is
reported: with such small group orders GMP-ECM's affine Weierstrass arithmetic meets
non-invertible values (for example doubling a point of order 2) and reports their gcd,
often N itself, where gpuecm2's Edwards formulas have no exceptional cases. The same can
happen for a small factor p of a large N, when the point's order mod p after stage 1 is
small: GMP-ECM may then find p through a value it cannot invert while setting up the roots
("Found factor while computing fd[]") where gpuecm2 computes the stage 2 values. Among the
tests this happened once, with a planted 17-digit prime at B1 = 1e4 (`large_tests.py`
reports it as a known difference); all other results tested agree exactly.

## Testing aids

Environment variables read by the stage 2 engine (for tests only):

- `GPUECM2_MAX_LOGT=n`: use NTTs of length at most 2^n (n >= 12).
- `GPUECM2_GPU_MEM=MB`: pretend that this much GPU memory is usable.
- `GPUECM2_INJECT_FAULT=1|2|3`: corrupt one coefficient (of H before the evaluation, or of F),
  or one root of G before G is built (a consistently wrong root, which only the self-check's
  CPU recomputation of roots can see), to check that `-selfcheck` catches it.
- `GPUECM2_PARAMS_ONLY=1`: print the stage 2 parameters of each residue, run no stage 2.
- `GPUECM2_TO_LIMBS=dot|cs|tiled|split`: force one of the RNS-to-limbs kernels (benchmarks).
- `GPUECM2_SPLIT_LAUNCHES=1`: split the long kernels (conversions, fixed-base tables and
  multiplications) into very short launches, as at the largest sizes; the results must not
  change (tests the splitting at small sizes).
- `GPUECM2_TREE_CACHE_LEVELS=n`: with `-treecache`, keep the transforms of at most n tree
  levels (tests the evaluation's switch between kept and recomputed levels).
- `GPUECM2_DEBUG=1`: print the root generation plan.
