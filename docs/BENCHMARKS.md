# Stage 2 at GMP-ECM's default parameters: CPU against GPU

GMP-ECM 7.0.6 (Debian build, GMP 6.3.0, `--enable-asm-redc`) on one core of an i7-14700F
under WSL2, against gpuecm2 on an RTX 4070 Ti SUPER, for the B1 values of GMP-ECM's table
of optimal parameters ("digits" is the factor size each B1 targets) and GMP-ECM's default
B2. Produced with `bench/defaults_bench.py`.

- N: RSA-155 (512 bits), RSA-1024, RSA-2048, RSA-2048·RSA-1536·RSA-155 (4095 bits).
- Both programs run with `-maxmem 8000` (the WSL machine has 15 GB), so the largest B1
  values use more, smaller blocks than an unlimited GMP-ECM would; both use exactly the same
  parameters (dF, k).
- CPU stage 1: real GMP-ECM stage 1 where it takes under ~10 minutes. For larger B1 the
  residue of the largest real stage 1 was kept and its B1 field raised (checksum
  recomputed): stage 2 does the same work for any point. Stage 1 times marked ≈ are
  extrapolated linearly in B1 from the largest measured one.
- CPU stage 2: `ecm -v -v -resume`, "Step 2 took", one job at a time while nothing else
  ran on the machine (no compiler, GPU program or other ecm process; a disturbed job is
  repeated). Cases with no CPU time would take hours to days (estimate from the CPU stage 2
  times measured).
- GPU stage 2: steady-state time per curve (`bench_stage2`, engine reused); "batched" is the
  automatic batch size (in parentheses), as `gpuecm2` uses it for consecutive residues of
  one number. For B1 ≥ 26e7 the time is that of the `gpuecm2` run on the same residue.
  The GPU times are those after the optimisation of the polynomial arithmetic of October 2026
  (NTT and conversion kernels; stage 2 at these parameters 11–23% faster than before at
  512–1024 bits, 18–34% at 2048–4096 bits).
- check: "same" means GMP-ECM and gpuecm2 printed the same "Product of G(f_i)" value
  (`-v -v`) for the same residue (rerun with the current version); where GMP-ECM was not run,
  gpuecm2 was run again with `-selfcheck` (with the version before the optimisation; the
  current one gives the same values as that one on all residues tested).

| N | digits | B1 | B2 (GMP-ECM default) | dF, k | CPU stage 1 | CPU stage 2 | GPU stage 2 | GPU, batched | speedup | speedup, batched | check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 20 | 11e3 | 1873422 | 256, 3 | 8 ms | 9 ms | 4.9 ms | 0.08 ms (512) | 2 | 112 | same |
| 512 | 25 | 5e4 | 12746592 | 512, 5 | 34 ms | 33 ms | 6.4 ms | 0.18 ms (256) | 5 | 183 | same |
| 512 | 30 | 25e4 | 128992510 | 2048, 3 | 171 ms | 117 ms | 7.7 ms | 0.58 ms (64) | 15 | 202 | same |
| 512 | 35 | 1e6 | 1045563762 | 4096, 6 | 691 ms | 473 ms | 12 ms | 2.3 ms (32) | 40 | 208 | same |
| 512 | 40 | 3e6 | 5706890290 | 16384, 2 | 2.2 s | 1.1 s | 12 ms | 5.2 ms (8) | 93 | 207 | same |
| 512 | 45 | 11e6 | 35133391030 | 32768, 3 | 8.1 s | 3.3 s | 24 ms | 17 ms (4) | 137 | 201 | same |
| 512 | 50 | 43e6 | 240490660426 | 65536, 5 | 31.8 s | 10.0 s | 53 ms | 48 ms (2) | 190 | 207 | same |
| 512 | 55 | 11e7 | 776278396540 | 131072, 4 | 80.8 s | 24.4 s | 138 ms | 138 ms (1) | 177 | 177 | same |
| 512 | 60 | 26e7 | 3178559884516 | 262144, 4 | 178 s | 51.6 s | 275 ms |  | 188 |  | same |
| 512 | 65 | 85e7 | 15892628251516 | 524288, 5 | ≈580 s | 126 s | 638 ms |  | 197 |  | same |
| 512 | 70 | 29e8 | 105101237217912 | 2097152, 2 | ≈1980 s | 316 s | 1.5 s |  | 206 |  | same |
| 512 | 75 | 76e8 | 367853173761162 | 2097152, 7 | ≈5189 s | 725 s | 3.8 s |  | 193 |  | same |
| 512 | 80 | 25e9 | 1786705836274602 | 2097152, 34 | ≈4.7 h | 3200 s | 15.8 s |  | 202 |  | same |
| 1024 | 20 | 11e3 | 1873422 | 256, 3 | 21 ms | 20 ms | 5.9 ms | 0.17 ms (512) | 3 | 118 | same |
| 1024 | 25 | 5e4 | 12746592 | 512, 5 | 106 ms | 67 ms | 7.8 ms | 0.44 ms (256) | 9 | 152 | same |
| 1024 | 30 | 25e4 | 128992510 | 2048, 3 | 535 ms | 244 ms | 8.4 ms | 1.4 ms (64) | 29 | 173 | same |
| 1024 | 35 | 1e6 | 1045563762 | 4096, 6 | 2.2 s | 1.0 s | 17 ms | 5.6 ms (32) | 62 | 184 | same |
| 1024 | 40 | 3e6 | 5706890290 | 16384, 2 | 6.8 s | 2.3 s | 20 ms | 12 ms (8) | 117 | 184 | same |
| 1024 | 45 | 11e6 | 35133391030 | 32768, 3 | 25.0 s | 7.4 s | 46 ms | 40 ms (4) | 160 | 183 | same |
| 1024 | 50 | 43e6 | 240490660426 | 65536, 5 | 90.4 s | 22.2 s | 119 ms | 118 ms (2) | 187 | 188 | same |
| 1024 | 55 | 11e7 | 776278396540 | 131072, 4 | 230 s | 57.0 s | 324 ms | 324 ms (1) | 176 | 176 | same |
| 1024 | 60 | 26e7 | 3178559884516 | 262144, 4 | ≈545 s | 118 s | 652 ms |  | 182 |  | same |
| 1024 | 65 | 85e7 | 15892628251516 | 524288, 5 | ≈1781 s | 290 s | 1.5 s |  | 189 |  | same |
| 1024 | 70 | 29e8 | 90975170616348 | 1048576, 7 | ≈1.7 h | 789 s | 4.1 s |  | 194 |  | same |
| 1024 | 75 | 76e8 | 324909696561468 | 1048576, 25 | ≈4.4 h | 2418 s | 12.2 s |  | 198 |  | same |
| 1024 | 80 | 25e9 | 1767494382113478 | 1048576, 136 | ≈14.6 h | - | 62.9 s |  |  |  | self-check ok |
| 2048 | 20 | 11e3 | 1684420 | 288, 2 | 92 ms | 49 ms | 8.9 ms | 0.4 ms (455) | 5 | 122 | same |
| 2048 | 25 | 5e4 | 15446350 | 864, 2 | 433 ms | 178 ms | 12 ms | 1.1 ms (151) | 15 | 155 | same |
| 2048 | 30 | 25e4 | 183032866 | 2880, 2 | 2.3 s | 750 ms | 16 ms | 4.7 ms (45) | 47 | 160 | same |
| 2048 | 35 | 1e6 | 974637522 | 6720, 2 | 8.6 s | 2.1 s | 21 ms | 11 ms (19) | 99 | 191 | same |
| 2048 | 40 | 3e6 | 4592487916 | 14400, 2 | 27.6 s | 5.8 s | 41 ms | 30 ms (9) | 141 | 193 | same |
| 2048 | 45 | 11e6 | 30114149530 | 36864, 2 | 102 s | 21.4 s | 116 ms | 111 ms (3) | 184 | 193 | same |
| 2048 | 50 | 43e6 | 198654756318 | 92160, 2 | 370 s | 55.7 s | 278 ms | 279 ms (1) | 200 | 200 | same |
| 2048 | 55 | 11e7 | 900514153782 | 199680, 2 | ≈948 s | 198 s | 949 ms | 950 ms (1) | 209 | 208 | same |
| 2048 | 60 | 26e7 | 3079973376496 | 362880, 2 | ≈2240 s | 364 s | 1.7 s |  | 210 |  | same |
| 2048 | 65 | 85e7 | 15714153955516 | 518400, 5 | ≈2.0 h | 1015 s | 4.4 s |  | 233 |  | same |
| 2048 | 70 | 29e8 | 81712898767516 | 518400, 26 | ≈6.9 h | 4232 s | 17.8 s |  | 238 |  | same |
| 2048 | 75 | 76e8 | 323705466384316 | 518400, 103 | ≈18.2 h | - | 67.3 s |  |  |  | self-check ok |
| 2048 | 80 | 25e9 | 1759936561983616 | 518400, 560 | ≈59.8 h | - | 361 s |  |  |  | self-check ok |
| 4095 | 20 | 11e3 | 1684420 | 288, 2 | 363 ms | 118 ms | 19 ms | 1.1 ms (455) | 6 | 105 | same |
| 4095 | 25 | 5e4 | 15446350 | 864, 2 | 1.7 s | 446 ms | 25 ms | 3.4 ms (151) | 18 | 132 | same |
| 4095 | 30 | 25e4 | 183032866 | 2880, 2 | 8.8 s | 1.9 s | 35 ms | 14 ms (45) | 54 | 134 | same |
| 4095 | 35 | 1e6 | 974637522 | 6720, 2 | 36.0 s | 5.3 s | 50 ms | 33 ms (19) | 105 | 159 | same |
| 4095 | 40 | 3e6 | 4592487916 | 14400, 2 | 109 s | 15.3 s | 110 ms | 94 ms (9) | 139 | 162 | same |
| 4095 | 45 | 11e6 | 30114149530 | 36864, 2 | 371 s | 59.1 s | 354 ms | 337 ms (3) | 167 | 175 | same |
| 4095 | 50 | 43e6 | 198654756318 | 92160, 2 | ≈1452 s | 157 s | 866 ms | 866 ms (1) | 181 | 181 | same |
| 4095 | 55 | 11e7 | 900514153782 | 199680, 2 | ≈3713 s | 655 s | 3.3 s | 3.3 s (1) | 197 | 197 | same |
| 4095 | 60 | 26e7 | 3142865025316 | 259200, 4 | ≈2.4 h | 1383 s | 6.2 s |  | 225 |  | same |
| 4095 | 65 | 85e7 | 14142901172416 | 259200, 18 | ≈8.0 h | 5010 s | 20.6 s |  | 243 |  | same |
| 4095 | 70 | 29e8 | 80927318021566 | 259200, 103 | ≈27.2 h | - | 109 s |  |  |  | self-check ok |
| 4095 | 75 | 76e8 | 321348535858366 | 259200, 409 | ≈71.3 h | - | 427 s |  |  |  | self-check ok |
| 4095 | 80 | 25e9 | 1759936658980516 | 259200, 2240 | ≈234.4 h | - | 2326 s |  |  |  | self-check ok |

Speedup is CPU stage 2 time over GPU stage 2 time. For the target range (B1 ≥ 3e6) the GPU
does a stage 2 160–240 times faster than one core of this CPU, at every size. Below B1 ≈ 1e6
a single curve is too small for the GPU and batching is what makes it fast.

## Stage 2 time over the whole range of sizes

gpuecm2's stage 2 per curve at GMP-ECM's default B2, for random moduli of each size, with
the automatic batch size (`bench/profile_grid.py`, which also covers both sides of every
size where the implementation changes; idle GPU, after the optimisation of the polynomial
arithmetic of October 2026: 19–37% shorter than the earlier version's grid, which was
measured with the desktop in use, so up to about 10% of that is the quieter GPU):

| N bits | B1 1e6 | B1 3e6 | B1 11e6 | B1 43e6 | B1 11e7 | B1 26e7 |
|---|---|---|---|---|---|---|
| 256 | 1.5 ms | 3.6 ms | 12 ms | 34 ms | 111 ms | 208 ms |
| 512 | 2.2 ms | 5.1 ms | 17 ms | 48 ms | 138 ms | 274 ms |
| 768 | 4.0 ms | 9.0 ms | 30 ms | 85 ms | 241 ms | 489 ms |
| 1024 | 5.5 ms | 12 ms | 40 ms | 118 ms | 322 ms | 648 ms |
| 1536 | 11 ms | 23 ms | 80 ms | 217 ms | 646 ms | 1.25 s |
| 2048 | 11 ms | 30 ms | 110 ms | 278 ms | 948 ms | 1.74 s |
| 2560 | 16 ms | 43 ms | 158 ms | 398 ms | 1.36 s | 2.54 s |
| 3072 | 21 ms | 60 ms | 215 ms | 542 ms | 1.90 s | 3.53 s |
| 3584 | 27 ms | 77 ms | 274 ms | 705 ms | 2.67 s | 4.59 s |
| 4096 | 33 ms | 94 ms | 337 ms | 866 ms | 3.33 s | 5.72 s |
| 4608 | 39 ms | 111 ms | 401 ms | 1.03 s | 4.02 s | 6.86 s |

The time grows smoothly with the size of N. Between neighbouring sizes the only steps are
where the CGBN size of the curve arithmetic changes (e.g. 4096 to 4097 bits: roots 25-35%
more expensive) and where GMP-ECM changes its parameters (above 1920 bits dF is no longer a
power of two and fewer, larger blocks are used). Where the time goes (one curve, phases
synchronised):

| N bits | B1 | roots of F and G | trees, 1/F, products mod F | evaluation, product |
|---|---|---|---|---|
| 512 | 3e6 | 48% | 38% | 14% |
| 512 | 11e7 | 61% | 30% | 8% |
| 1024 | 3e6 | 48% | 37% | 15% |
| 1024 | 11e7 | 59% | 32% | 9% |
| 2048 | 3e6 | 44% | 40% | 16% |
| 2048 | 11e7 | 62% | 26% | 12% |
| 4096 | 3e6 | 44% | 40% | 17% |
| 4096 | 11e7 | 69% | 21% | 10% |

Root generation (the curve arithmetic) takes 44–48% of the time at B1 = 3e6 and 59–69% at
11e7, where GMP-ECM's default Brent–Suyama polynomial is Dickson(30) (30 point additions per
root; a lower degree with a larger B2 finds factors faster with GPU stage 2, see the end of
this file). Within the polynomial arithmetic, the NTTs run at about 80% of the DRAM
bandwidth, and the conversions between residues mod N and the RNS (matrix products whose
size grows with the square of the size of N) are the largest part for large N
(docs/DESIGN.md).

For larger B2 the time grows less than proportionally: 4 times GMP-ECM's default B2 costs
2.7-3.8 times the time, 16 times costs 5.2-7.6 times, at every size from 256 to 4608 bits
(B1 = 3e6 and 43e6; measured before the optimisation). Where GMP-ECM's dF does not fit in the GPU memory (4608 bits at
16 times the default for B1 = 43e6) gpuecm2 uses a smaller dF with more blocks.

## Stage 1 and stage 2 on the same GPU

GMP-ECM 7.0.6's GPU stage 1 (`ecm -gpu`, CGBN, parametrization 3, 4224 curves per run) on
the same RTX 4070 Ti SUPER, measured with `bench/defaults_bench.py gpus1` at B1 = 1e5, 1e6
and 3e6 and scaled linearly in B1 (the time per bit of the multiplier was the same at all
three B1 values). GMP-ECM 7.0.6 needs a CGBN kernel of at least bits(N) + 6 bits; its default
list (512, 1024, 1536, 2048, 3072, 4096 bits) has none for the 4095-bit number. For this
measurement GMP-ECM was built with additional kernel sizes (768, 1280, 2560, 4608 bits:
the smallest that CGBN compiles above N + 6 for each number; 640 and 2304 are not supported
by CGBN's division code), and its residues were checked to equal GMP-ECM's CPU stage 1.
Stage 2 is gpuecm2's batched time per curve at GMP-ECM's default B2 (current version).

| N | B1 | GPU stage 1, smallest kernel | GPU stage 1, GMP-ECM 7.0.6 kernel | GPU stage 2 | GPU s2/s1 | CPU s2/s1 |
|---|---|---|---|---|---|---|
| 512 | 1e6 | 7.9 ms (768/8) | 9.5 ms (1024/8) | 2.3 ms | 0.29 | 0.68 |
| 512 | 3e6 | 24 ms (768/8) | 29 ms (1024/8) | 5.2 ms | 0.22 | 0.50 |
| 512 | 11e6 | 87 ms (768/8) | 105 ms (1024/8) | 17 ms | 0.19 | 0.41 |
| 512 | 43e6 | 340 ms (768/8) | 410 ms (1024/8) | 48 ms | 0.14 | 0.32 |
| 512 | 11e7 | 869 ms (768/8) | 1.0 s (1024/8) | 138 ms | 0.16 | 0.30 |
| 512 | 26e7 | 2.1 s (768/8) | 2.5 s (1024/8) | 275 ms | 0.13 | 0.29 |
| 512 | 85e7 | 6.7 s (768/8) | 8.1 s (1024/8) | 638 ms | 0.10 | 0.22 |
| 1024 | 1e6 | 13 ms (1280/8) | 17 ms (1536/8) | 5.6 ms | 0.43 | 0.47 |
| 1024 | 3e6 | 39 ms (1280/8) | 50 ms (1536/8) | 12 ms | 0.32 | 0.34 |
| 1024 | 11e6 | 143 ms (1280/8) | 183 ms (1536/8) | 40 ms | 0.28 | 0.30 |
| 1024 | 43e6 | 559 ms (1280/8) | 716 ms (1536/8) | 118 ms | 0.21 | 0.25 |
| 1024 | 11e7 | 1.4 s (1280/8) | 1.8 s (1536/8) | 324 ms | 0.23 | 0.25 |
| 1024 | 26e7 | 3.4 s (1280/8) | 4.3 s (1536/8) | 652 ms | 0.19 | 0.22 |
| 1024 | 85e7 | 11.0 s (1280/8) | 14.1 s (1536/8) | 1.5 s | 0.14 | 0.16 |
| 2048 | 1e6 | 45 ms (2560/16) | 60 ms (3072/16) | 11 ms | 0.25 | 0.25 |
| 2048 | 3e6 | 134 ms (2560/16) | 179 ms (3072/16) | 30 ms | 0.23 | 0.21 |
| 2048 | 11e6 | 491 ms (2560/16) | 656 ms (3072/16) | 111 ms | 0.23 | 0.21 |
| 2048 | 43e6 | 1.9 s (2560/16) | 2.6 s (3072/16) | 279 ms | 0.15 | 0.15 |
| 2048 | 11e7 | 4.9 s (2560/16) | 6.6 s (3072/16) | 950 ms | 0.19 | 0.21 |
| 2048 | 26e7 | 11.6 s (2560/16) | 15.5 s (3072/16) | 1.7 s | 0.15 | 0.16 |
| 2048 | 85e7 | 38.0 s (2560/16) | 50.7 s (3072/16) | 4.4 s | 0.11 | 0.14 |
| 4095 | 1e6 | 119 ms (4608/16) | - | 33 ms | 0.28 | 0.15 |
| 4095 | 3e6 | 358 ms (4608/16) | - | 94 ms | 0.26 | 0.14 |
| 4095 | 11e6 | 1.3 s (4608/16) | - | 337 ms | 0.26 | 0.16 |
| 4095 | 43e6 | 5.1 s (4608/16) | - | 866 ms | 0.17 | 0.11 |
| 4095 | 11e7 | 13.1 s (4608/16) | - | 3.3 s | 0.25 | 0.18 |
| 4095 | 26e7 | 31.0 s (4608/16) | - | 6.2 s | 0.20 | 0.16 |
| 4095 | 85e7 | 101 s (4608/16) | - | 20.6 s | 0.20 | 0.17 |

With one GPU doing both stages, gpuecm2's stage 2 at the default B2 takes 0.1–0.4 of GPU
stage 1, about the same share as stage 2 has on the CPU (0.1–0.7) or less. The default B2 is
therefore about as well balanced on the GPU as it is on the CPU.

## GPU only against GPU stage 1 with CPU stage 2

The usual way to use a GPU with GMP-ECM so far is GPU stage 1 followed by stage 2 of the
saved residues on CPU cores. With gpuecm2 both stages run on the GPU, one after the other
(running them concurrently does not help: each keeps the GPU busy on its own, see below).
Time per curve, from the measurements above; "GPU stage 1 + 20 cores" assumes that 20 CPU
cores run stage 2 while the GPU runs stage 1 of the next curves, with perfect scaling over the
cores (optimistic: on the test machine eight parallel `ecm` processes each ran 1.5–1.7 times
slower than one alone), so that per curve it takes the larger of the two:

| N bits | B1 | GPU stage 1 | GPU stage 2 | GPU only | CPU stage 2 (one core) | GPU stage 1 + 20 cores of CPU stage 2 | GPU only faster by |
|---|---|---|---|---|---|---|---|
| 512 | 1e6 | 7.9 ms | 2.3 ms | 10 ms | 473 ms | 24 ms | 2.3× |
| 512 | 3e6 | 24 ms | 5.2 ms | 29 ms | 1.1 s | 55 ms | 1.9× |
| 512 | 11e6 | 87 ms | 17 ms | 104 ms | 3.3 s | 165 ms | 1.6× |
| 512 | 43e6 | 340 ms | 48 ms | 388 ms | 10.0 s | 500 ms | 1.3× |
| 512 | 11e7 | 869 ms | 138 ms | 1.0 s | 24.4 s | 1.2 s | 1.2× |
| 1024 | 1e6 | 13 ms | 5.6 ms | 19 ms | 1.0 s | 50 ms | 2.7× |
| 1024 | 3e6 | 39 ms | 12 ms | 51 ms | 2.3 s | 115 ms | 2.2× |
| 1024 | 11e6 | 143 ms | 40 ms | 183 ms | 7.4 s | 370 ms | 2.0× |
| 1024 | 43e6 | 559 ms | 118 ms | 677 ms | 22.2 s | 1.1 s | 1.6× |
| 1024 | 11e7 | 1.4 s | 324 ms | 1.7 s | 57.0 s | 2.9 s | 1.7× |
| 2048 | 1e6 | 45 ms | 11 ms | 56 ms | 2.1 s | 105 ms | 1.9× |
| 2048 | 3e6 | 134 ms | 30 ms | 164 ms | 5.8 s | 290 ms | 1.8× |
| 2048 | 11e6 | 491 ms | 111 ms | 602 ms | 21.4 s | 1.1 s | 1.8× |
| 2048 | 43e6 | 1.9 s | 279 ms | 2.2 s | 55.7 s | 2.8 s | 1.3× |
| 2048 | 11e7 | 4.9 s | 950 ms | 5.9 s | 198 s | 9.9 s | 1.7× |
| 4095 | 1e6 | 119 ms | 33 ms | 152 ms | 5.3 s | 265 ms | 1.7× |
| 4095 | 3e6 | 358 ms | 94 ms | 452 ms | 15.3 s | 765 ms | 1.7× |
| 4095 | 11e6 | 1.3 s | 337 ms | 1.6 s | 59.1 s | 3.0 s | 1.8× |
| 4095 | 43e6 | 5.1 s | 866 ms | 6.0 s | 157 s | 7.8 s | 1.3× |
| 4095 | 11e7 | 13.1 s | 3.3 s | 16.4 s | 655 s | 32.8 s | 2.0× |

In the hybrid setup the CPU limits the throughput everywhere: keeping up with GPU stage 1 at
GMP-ECM's default B2 would need 40 to 60 cores. GPU only is 1.2–2.7 times faster than the
GPU with 20 cores, and leaves the CPU free.

End to end, `ecm -gpu2` (GMP-ECM 7.0.6 with the patches in gmp-ecm/) for 4224 curves of
RSA-1024 at B1 = 3e6 takes 206.4 s (48.9 ms per curve): 36.3 ms of GPU stage 1 and 12.5 ms
of GPU stage 2 per curve, on an idle GPU; the rest of GMP-ECM's work per run is negligible.
(Before the optimisation of the polynomial arithmetic: 260.3 s, 43.2 + 18.4 ms per curve,
measured with the desktop using part of the GPU.)

Running stage 2 of one batch of curves concurrently with stage 1 of the next (in two
threads of one process; separate processes do not run concurrently on a GPU under Windows)
was tried: for 3 × 4224 curves of RSA-1024 at B1 = 1e6 it took 272 s against 263 s one after
the other (283 s with stage 1 at the higher stream priority). Stage 1's kernels occupy every
SM for about 100 ms at a time and stage 2 then gets almost no share, and batched stage 2 alone
keeps the GPU busy 97.5% of the time, so there is no idle time to fill.

## Choosing B2 when stage 2 runs on the GPU

The expected time to find a factor of the targeted size is curves × (stage 1 + stage 2)
per curve, where the number of curves is GMP-ECM's expected number (`ecm -v -param 3`)
for the given B1 and B2. `bench/b2_tradeoff.py` measures gpuecm2's stage 2 for
B2 = m × GMP-ECM's default B2 (m from 0.25 up) and uses the GPU stage 1 times above
(smallest kernel):

| N | digits | B1 | GPU stage 1 | stage 2 at default B2 | best m | stage 2 at best m | curves at best m | time per factor vs default |
|---|---|---|---|---|---|---|---|---|
| 512 | 35 | 1e6 | 7.9 ms | 2.2 ms | 3 | 3.6 ms | 81% | 92% |
| 512 | 40 | 3e6 | 24 ms | 5.4 ms | 5 | 14 ms | 73% | 94% |
| 512 | 45 | 11e6 | 87 ms | 17 ms | 5 | 35 ms | 77% | 91% |
| 512 | 50 | 43e6 | 340 ms | 48 ms | 10 | 226 ms | 67% | 97% |
| 512 | 55 | 11e7 | 869 ms | 137 ms | 10 | 453 ms | 69% | 91% |
| 512 | 60 | 26e7 | 2.1 s | 269 ms | 10 | 745 ms | 72% | 86% |
| 1024 | 35 | 1e6 | 13 ms | 5.5 ms | 3 | 8.8 ms | 81% | 95% |
| 1024 | 40 | 3e6 | 39 ms | 12 ms | 1 | 12 ms | 100% | 100% |
| 1024 | 45 | 11e6 | 143 ms | 39 ms | 5 | 83 ms | 77% | 95% |
| 1024 | 50 | 43e6 | 559 ms | 116 ms | 1 | 116 ms | 100% | 100% |
| 1024 | 55 | 11e7 | 1.4 s | 318 ms | 3 | 535 ms | 84% | 94% |
| 1024 | 60 | 26e7 | 3.4 s | 638 ms | 10 | 1.7 s | 72% | 92% |
| 2048 | 35 | 1e6 | 45 ms | 11 ms | 3 | 23 ms | 77% | 94% |
| 2048 | 40 | 3e6 | 134 ms | 30 ms | 2 | 44 ms | 88% | 96% |
| 2048 | 45 | 11e6 | 491 ms | 109 ms | 5 | 243 ms | 76% | 92% |
| 2048 | 50 | 43e6 | 1.9 s | 272 ms | 5 | 1.0 s | 73% | 98% |
| 2048 | 55 | 11e7 | 4.9 s | 932 ms | 5 | 2.0 s | 79% | 92% |
| 2048 | 60 | 26e7 | 11.6 s | 1.7 s | 5 | 4.0 s | 78% | 91% |
| 4095 | 35 | 1e6 | 119 ms | 33 ms | 3 | 71 ms | 77% | 97% |
| 4095 | 40 | 3e6 | 358 ms | 93 ms | 2 | 134 ms | 88% | 96% |
| 4095 | 45 | 11e6 | 1.3 s | 333 ms | 5 | 746 ms | 76% | 94% |
| 4095 | 50 | 43e6 | 5.1 s | 855 ms | 1 | 855 ms | 100% | 100% |
| 4095 | 55 | 11e7 | 13.1 s | 3.3 s | 5 | 6.4 s | 79% | 94% |
| 4095 | 60 | 26e7 | 31.0 s | 5.7 s | 3 | 10.2 s | 85% | 96% |

The best B2 is 1 to 10 times GMP-ECM's default and saves 0–14% of the time per factor,
mostly 4–9% (measured with the current version; before the optimisation of the polynomial
arithmetic the gain was 0–12%); the jumps between neighbouring rows come from the discrete
choice of dF and k. GMP-ECM's default B2 is therefore a reasonable choice for GPU stage 1
with gpuecm2 stage 2, and two to five times the default is a small, safe gain.

The Brent–Suyama polynomial matters more. GMP-ECM chooses Dickson(30) from B1 = 11e7 on,
which costs 30 point additions per root, while root generation is already the larger part of
gpuecm2's stage 2. With `-dickson 6` and 8 times the default B2 (both options of GMP-ECM and
gpuecm2, so the results stay comparable with GMP-ECM) the expected time per 55-digit factor
at B1 = 11e7 is 82% (512 bits) and 84% (1024 bits) of that with GMP-ECM's defaults; -dickson
12 is about as good, x^1 (no extension) slightly worse.

Running stage 1 on CPU cores and only stage 2 on the GPU also works (gpuecm2 reads the
residues of any GMP-ECM run); the GPU then idles at the default B2 and a larger B2 (`ecm`'s
B2 argument, the same for gpuecm2) makes better use of it. Such B2 values are tested against
GMP-ECM (docs/TESTING.md: full B2 up to 1e13, B2min–B2 ranges up to 1e39).
