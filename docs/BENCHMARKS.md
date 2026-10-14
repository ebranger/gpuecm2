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
  These GPU times were measured before the RNS conversions became tiled matrix products,
  which made stage 2 another 1–17% faster (most at 2048 bits).
- check: "same" means GMP-ECM and gpuecm2 printed the same "Product of G(f_i)" value
  (`-v -v`) for the same residue; where GMP-ECM was not run, gpuecm2 was run again with
  `-selfcheck`.

| N | digits | B1 | B2 (GMP-ECM default) | dF, k | CPU stage 1 | CPU stage 2 | GPU stage 2 | GPU, batched | speedup | speedup, batched | check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 512 | 20 | 11e3 | 1873422 | 256, 3 | 8 ms | 9 ms | 4.8 ms | 0.09 ms (512) | 2 | 100 | same |
| 512 | 25 | 5e4 | 12746592 | 512, 5 | 34 ms | 33 ms | 6.9 ms | 0.23 ms (256) | 5 | 143 | same |
| 512 | 30 | 25e4 | 128992510 | 2048, 3 | 171 ms | 117 ms | 7 ms | 0.74 ms (64) | 17 | 158 | same |
| 512 | 35 | 1e6 | 1045563762 | 4096, 6 | 691 ms | 473 ms | 12 ms | 2.8 ms (32) | 38 | 170 | same |
| 512 | 40 | 3e6 | 5706890290 | 16384, 2 | 2.2 s | 1.1 s | 13 ms | 6.4 ms (8) | 85 | 168 | same |
| 512 | 45 | 11e6 | 35133391030 | 32768, 3 | 8.1 s | 3.3 s | 27 ms | 20 ms (4) | 122 | 165 | same |
| 512 | 50 | 43e6 | 240490660426 | 65536, 5 | 31.8 s | 10.0 s | 61 ms | 59 ms (2) | 164 | 171 | same |
| 512 | 55 | 11e7 | 776278396540 | 131072, 4 | 80.8 s | 24.4 s | 157 ms | 157 ms (1) | 156 | 156 | same |
| 512 | 60 | 26e7 | 3178559884516 | 262144, 4 | 178 s | 51.6 s | 318 ms |  | 162 |  | same |
| 512 | 65 | 85e7 | 15892628251516 | 524288, 5 | ≈580 s | 126 s | 747 ms |  | 169 |  | same |
| 512 | 70 | 29e8 | 105101237217912 | 2097152, 2 | ≈1980 s | 316 s | 1.9 s |  | 171 |  | same |
| 512 | 75 | 76e8 | 367853173761162 | 2097152, 7 | ≈5189 s | 725 s | 4.4 s |  | 165 |  | same |
| 512 | 80 | 25e9 | 1786705836274602 | 2097152, 34 | ≈4.7 h | 3200 s | 18.1 s |  | 176 |  | same |
| 1024 | 20 | 11e3 | 1873422 | 256, 3 | 21 ms | 20 ms | 5.7 ms | 0.2 ms (512) | 3 | 100 | same |
| 1024 | 25 | 5e4 | 12746592 | 512, 5 | 106 ms | 67 ms | 7.6 ms | 0.55 ms (256) | 9 | 122 | same |
| 1024 | 30 | 25e4 | 128992510 | 2048, 3 | 535 ms | 244 ms | 8.8 ms | 1.8 ms (64) | 28 | 133 | same |
| 1024 | 35 | 1e6 | 1045563762 | 4096, 6 | 2.2 s | 1.0 s | 17 ms | 7 ms (32) | 62 | 146 | same |
| 1024 | 40 | 3e6 | 5706890290 | 16384, 2 | 6.8 s | 2.3 s | 22 ms | 16 ms (8) | 103 | 143 | same |
| 1024 | 45 | 11e6 | 35133391030 | 32768, 3 | 25.0 s | 7.4 s | 54 ms | 50 ms (4) | 137 | 148 | same |
| 1024 | 50 | 43e6 | 240490660426 | 65536, 5 | 90.4 s | 22.2 s | 144 ms | 146 ms (2) | 154 | 152 | same |
| 1024 | 55 | 11e7 | 776278396540 | 131072, 4 | 230 s | 57.0 s | 374 ms | 375 ms (1) | 153 | 152 | same |
| 1024 | 60 | 26e7 | 3178559884516 | 262144, 4 | ≈545 s | 118 s | 763 ms |  | 155 |  | same |
| 1024 | 65 | 85e7 | 15892628251516 | 524288, 5 | ≈1781 s | 290 s | 1.8 s |  | 160 |  | same |
| 1024 | 70 | 29e8 | 90975170616348 | 1048576, 7 | ≈1.7 h | 789 s | 4.8 s |  | 163 |  | same |
| 1024 | 75 | 76e8 | 324909696561468 | 1048576, 25 | ≈4.4 h | 2418 s | 14.5 s |  | 167 |  | same |
| 1024 | 80 | 25e9 | 1767494382113478 | 1048576, 136 | ≈14.6 h | - | 74.5 s |  |  |  | self-check ok |
| 2048 | 20 | 11e3 | 1684420 | 288, 2 | 92 ms | 49 ms | 9.6 ms | 0.57 ms (455) | 5 | 86 | same |
| 2048 | 25 | 5e4 | 15446350 | 864, 2 | 433 ms | 178 ms | 13 ms | 1.7 ms (151) | 14 | 105 | same |
| 2048 | 30 | 25e4 | 183032866 | 2880, 2 | 2.3 s | 750 ms | 18 ms | 6.9 ms (45) | 41 | 108 | same |
| 2048 | 35 | 1e6 | 974637522 | 6720, 2 | 8.6 s | 2.1 s | 27 ms | 17 ms (19) | 79 | 127 | same |
| 2048 | 40 | 3e6 | 4592487916 | 14400, 2 | 27.6 s | 5.8 s | 53 ms | 43 ms (9) | 110 | 135 | same |
| 2048 | 45 | 11e6 | 30114149530 | 36864, 2 | 102 s | 21.4 s | 155 ms | 150 ms (3) | 138 | 142 | same |
| 2048 | 50 | 43e6 | 198654756318 | 92160, 2 | 370 s | 55.7 s | 379 ms | 379 ms (1) | 147 | 147 | same |
| 2048 | 55 | 11e7 | 900514153782 | 199680, 2 | ≈948 s | 198 s | 1.2 s | 1.2 s (1) | 167 | 167 | same |
| 2048 | 60 | 26e7 | 3079973376496 | 362880, 2 | ≈2240 s | 364 s | 2.2 s |  | 166 |  | same |
| 2048 | 65 | 85e7 | 15714153955516 | 518400, 5 | ≈2.0 h | 1015 s | 5.4 s |  | 186 |  | same |
| 2048 | 70 | 29e8 | 81712898767516 | 518400, 26 | ≈6.9 h | 4232 s | 22.1 s |  | 191 |  | same |
| 2048 | 75 | 76e8 | 323705466384316 | 518400, 103 | ≈18.2 h | - | 83.5 s |  |  |  | self-check ok |
| 2048 | 80 | 25e9 | 1759936561983616 | 518400, 560 | ≈59.8 h | - | 448 s |  |  |  | self-check ok |
| 4095 | 20 | 11e3 | 1684420 | 288, 2 | 363 ms | 118 ms | 21 ms | 1.6 ms (455) | 6 | 72 | same |
| 4095 | 25 | 5e4 | 15446350 | 864, 2 | 1.7 s | 446 ms | 28 ms | 5.1 ms (151) | 16 | 87 | same |
| 4095 | 30 | 25e4 | 183032866 | 2880, 2 | 8.8 s | 1.9 s | 44 ms | 21 ms (45) | 44 | 90 | same |
| 4095 | 35 | 1e6 | 974637522 | 6720, 2 | 36.0 s | 5.3 s | 72 ms | 51 ms (19) | 73 | 104 | same |
| 4095 | 40 | 3e6 | 4592487916 | 14400, 2 | 109 s | 15.3 s | 148 ms | 135 ms (9) | 103 | 114 | same |
| 4095 | 45 | 11e6 | 30114149530 | 36864, 2 | 371 s | 59.1 s | 474 ms | 456 ms (3) | 125 | 130 | same |
| 4095 | 50 | 43e6 | 198654756318 | 92160, 2 | ≈1452 s | 157 s | 1.2 s | 1.2 s (1) | 134 | 134 | same |
| 4095 | 55 | 11e7 | 900514153782 | 199680, 2 | ≈3713 s | 655 s | 4.0 s | 4.0 s (1) | 162 | 162 | same |
| 4095 | 60 | 26e7 | 3142865025316 | 259200, 4 | ≈2.4 h | 1383 s | 7.5 s |  | 184 |  | same |
| 4095 | 65 | 85e7 | 14142901172416 | 259200, 18 | ≈8.0 h | 5010 s | 25.1 s |  | 199 |  | same |
| 4095 | 70 | 29e8 | 80927318021566 | 259200, 103 | ≈27.2 h | - | 133 s |  |  |  | self-check ok |
| 4095 | 75 | 76e8 | 321348535858366 | 259200, 409 | ≈71.3 h | - | 518 s |  |  |  | self-check ok |
| 4095 | 80 | 25e9 | 1759936658980516 | 259200, 2240 | ≈234.4 h | - | 2828 s |  |  |  | self-check ok |

Speedup is CPU stage 2 time over GPU stage 2 time. For the target range (B1 ≥ 3e6) the GPU
does a stage 2 100–200 times faster than one core of this CPU, at every size. Below B1 ≈ 1e6
a single curve is too small for the GPU and batching is what makes it fast.

## Stage 2 time over the whole range of sizes

gpuecm2's stage 2 per curve at GMP-ECM's default B2, for random moduli of each size, with
the automatic batch size (`bench/profile_grid.py`, which also covers both sides of every
size where the implementation changes; measured with the desktop in use, idle-GPU times are
up to about 10% lower):

| N bits | B1 1e6 | B1 3e6 | B1 11e6 | B1 43e6 | B1 11e7 | B1 26e7 |
|---|---|---|---|---|---|---|
| 256 | 1.9 ms | 4.5 ms | 16 ms | 44 ms | 146 ms | 273 ms |
| 512 | 3.1 ms | 7.2 ms | 23 ms | 66 ms | 195 ms | 377 ms |
| 768 | 5.4 ms | 13 ms | 40 ms | 115 ms | 333 ms | 656 ms |
| 1024 | 7.7 ms | 17 ms | 55 ms | 161 ms | 451 ms | 887 ms |
| 1536 | 14 ms | 31 ms | 104 ms | 287 ms | 872 ms | 1.66 s |
| 2048 | 16 ms | 42 ms | 150 ms | 372 ms | 1.26 s | 2.35 s |
| 2560 | 23 ms | 64 ms | 216 ms | 543 ms | 1.82 s | 3.44 s |
| 3072 | 31 ms | 82 ms | 289 ms | 731 ms | 2.52 s | 4.74 s |
| 3584 | 41 ms | 109 ms | 385 ms | 988 ms | 3.59 s | 6.35 s |
| 4096 | 53 ms | 140 ms | 488 ms | 1.24 s | 4.58 s | 8.06 s |
| 4608 | 58 ms | 157 ms | 550 ms | 1.41 s | 5.39 s | 9.27 s |

The time grows smoothly with the size of N. Between neighbouring sizes the only steps are
where the CGBN size of the curve arithmetic changes (e.g. 4096 to 4097 bits: roots 25-35%
more expensive) and where GMP-ECM changes its parameters (above 1920 bits dF is no longer a
power of two and fewer, larger blocks are used). Where the time goes (one curve, phases
synchronised):

| N bits | B1 | roots of F and G | trees, 1/F, products mod F | evaluation, product |
|---|---|---|---|---|
| 512 | 3e6 | 41% | 37% | 22% |
| 1024 | 3e6 | 43% | 41% | 16% |
| 2048 | 3e6 | 40% | 42% | 18% |
| 4096 | 3e6 | 35% | 45% | 19% |
| 512 | 11e7 | 54% | 36% | 11% |
| 1024 | 11e7 | 51% | 39% | 10% |
| 2048 | 11e7 | 54% | 31% | 14% |
| 4096 | 11e7 | 59% | 28% | 12% |

Root generation (the curve arithmetic) and the polynomial arithmetic share the time about
evenly; with Dickson(30) at large B1 the roots take more than half. Within the polynomial
arithmetic, the conversions between residues mod N and the RNS (matrix products whose
size grows with the square of the size of N) take half or more for large N; they are tiled
matrix products (docs/DESIGN.md).

For larger B2 the time grows less than proportionally: 4 times GMP-ECM's default B2 costs
2.7-3.8 times the time, 16 times costs 5.2-7.6 times, at every size from 256 to 4608 bits
(B1 = 3e6 and 43e6). Where GMP-ECM's dF does not fit in the GPU memory (4608 bits at
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
Stage 2 is gpuecm2's batched time per curve at GMP-ECM's default B2.

| N | B1 | GPU stage 1, smallest kernel | GPU stage 1, GMP-ECM 7.0.6 kernel | GPU stage 2 | GPU s2/s1 | CPU s2/s1 |
|---|---|---|---|---|---|---|
| 512 | 1e6 | 7.9 ms (768/8) | 9.5 ms (1024/8) | 2.8 ms | 0.35 | 0.68 |
| 512 | 3e6 | 24 ms (768/8) | 29 ms (1024/8) | 6.4 ms | 0.27 | 0.50 |
| 512 | 11e6 | 87 ms (768/8) | 105 ms (1024/8) | 20 ms | 0.23 | 0.41 |
| 512 | 43e6 | 340 ms (768/8) | 410 ms (1024/8) | 59 ms | 0.17 | 0.32 |
| 512 | 11e7 | 869 ms (768/8) | 1.0 s (1024/8) | 157 ms | 0.18 | 0.30 |
| 512 | 26e7 | 2.1 s (768/8) | 2.5 s (1024/8) | 318 ms | 0.15 | 0.29 |
| 512 | 85e7 | 6.7 s (768/8) | 8.1 s (1024/8) | 747 ms | 0.11 | 0.22 |
| 1024 | 1e6 | 13 ms (1280/8) | 17 ms (1536/8) | 7 ms | 0.54 | 0.47 |
| 1024 | 3e6 | 39 ms (1280/8) | 50 ms (1536/8) | 16 ms | 0.41 | 0.34 |
| 1024 | 11e6 | 143 ms (1280/8) | 183 ms (1536/8) | 50 ms | 0.35 | 0.30 |
| 1024 | 43e6 | 559 ms (1280/8) | 716 ms (1536/8) | 146 ms | 0.26 | 0.25 |
| 1024 | 11e7 | 1.4 s (1280/8) | 1.8 s (1536/8) | 375 ms | 0.26 | 0.25 |
| 1024 | 26e7 | 3.4 s (1280/8) | 4.3 s (1536/8) | 763 ms | 0.23 | 0.22 |
| 1024 | 85e7 | 11.0 s (1280/8) | 14.1 s (1536/8) | 1.8 s | 0.16 | 0.16 |
| 2048 | 1e6 | 45 ms (2560/16) | 60 ms (3072/16) | 17 ms | 0.37 | 0.25 |
| 2048 | 3e6 | 134 ms (2560/16) | 179 ms (3072/16) | 43 ms | 0.32 | 0.21 |
| 2048 | 11e6 | 491 ms (2560/16) | 656 ms (3072/16) | 150 ms | 0.31 | 0.21 |
| 2048 | 43e6 | 1.9 s (2560/16) | 2.6 s (3072/16) | 379 ms | 0.20 | 0.15 |
| 2048 | 11e7 | 4.9 s (2560/16) | 6.6 s (3072/16) | 1.2 s | 0.24 | 0.21 |
| 2048 | 26e7 | 11.6 s (2560/16) | 15.5 s (3072/16) | 2.2 s | 0.19 | 0.16 |
| 2048 | 85e7 | 38.0 s (2560/16) | 50.7 s (3072/16) | 5.4 s | 0.14 | 0.14 |
| 4095 | 1e6 | 119 ms (4608/16) | - | 51 ms | 0.43 | 0.15 |
| 4095 | 3e6 | 358 ms (4608/16) | - | 135 ms | 0.38 | 0.14 |
| 4095 | 11e6 | 1.3 s (4608/16) | - | 456 ms | 0.35 | 0.16 |
| 4095 | 43e6 | 5.1 s (4608/16) | - | 1.2 s | 0.23 | 0.11 |
| 4095 | 11e7 | 13.1 s (4608/16) | - | 4.0 s | 0.31 | 0.18 |
| 4095 | 26e7 | 31.0 s (4608/16) | - | 7.5 s | 0.24 | 0.16 |
| 4095 | 85e7 | 101 s (4608/16) | - | 25.1 s | 0.25 | 0.17 |

With one GPU doing both stages, gpuecm2's stage 2 at the default B2 takes 0.1–0.5 of GPU
stage 1, about the same share as stage 2 has on the CPU (0.1–0.7). The default B2 is
therefore about as well balanced on the GPU as it is on the CPU.

## Choosing B2 when stage 2 runs on the GPU

The expected time to find a factor of the targeted size is curves × (stage 1 + stage 2)
per curve, where the number of curves is GMP-ECM's expected number (`ecm -v -param 3`)
for the given B1 and B2. `bench/b2_tradeoff.py` measures gpuecm2's stage 2 for
B2 = m × GMP-ECM's default B2 (m from 0.25 up) and uses the GPU stage 1 times above
(smallest kernel):

| N | digits | B1 | GPU stage 1 | stage 2 at default B2 | best m | stage 2 at best m | curves at best m | time per factor vs default |
|---|---|---|---|---|---|---|---|---|
| 512 | 35 | 1e6 | 7.9 ms | 2.8 ms | 3 | 4.5 ms | 81% | 94% |
| 512 | 40 | 3e6 | 24 ms | 6.4 ms | 5 | 16 ms | 73% | 97% |
| 512 | 45 | 11e6 | 87 ms | 20 ms | 5 | 43 ms | 77% | 93% |
| 512 | 50 | 43e6 | 340 ms | 59 ms | 3 | 157 ms | 80% | 99% |
| 512 | 55 | 11e7 | 869 ms | 158 ms | 3 | 266 ms | 84% | 93% |
| 512 | 60 | 26e7 | 2.1 s | 317 ms | 10 | 884 ms | 72% | 89% |
| 1024 | 35 | 1e6 | 13 ms | 7 ms | 2 | 9.4 ms | 87% | 98% |
| 1024 | 40 | 3e6 | 39 ms | 16 ms | 1 | 16 ms | 100% | 100% |
| 1024 | 45 | 11e6 | 143 ms | 49 ms | 3 | 83 ms | 83% | 98% |
| 1024 | 50 | 43e6 | 559 ms | 146 ms | 1 | 146 ms | 100% | 100% |
| 1024 | 55 | 11e7 | 1.4 s | 370 ms | 3 | 631 ms | 84% | 96% |
| 1024 | 60 | 26e7 | 3.4 s | 757 ms | 10 | 2.1 s | 72% | 95% |
| 2048 | 35 | 1e6 | 45 ms | 17 ms | 3 | 34 ms | 77% | 99% |
| 2048 | 40 | 3e6 | 134 ms | 43 ms | 2 | 65 ms | 88% | 99% |
| 2048 | 45 | 11e6 | 491 ms | 151 ms | 2 | 205 ms | 88% | 96% |
| 2048 | 50 | 43e6 | 1.9 s | 380 ms | 1 | 380 ms | 100% | 100% |
| 2048 | 55 | 11e7 | 4.9 s | 1.2 s | 2 | 1.5 s | 91% | 96% |
| 2048 | 60 | 26e7 | 11.6 s | 2.2 s | 5 | 5.1 s | 78% | 94% |
| 4095 | 35 | 1e6 | 119 ms | 51 ms | 1 | 51 ms | 100% | 100% |
| 4095 | 40 | 3e6 | 358 ms | 135 ms | 2 | 197 ms | 88% | 99% |
| 4095 | 45 | 11e6 | 1.3 s | 457 ms | 2 | 634 ms | 88% | 97% |
| 4095 | 50 | 43e6 | 5.1 s | 1.2 s | 1 | 1.2 s | 100% | 100% |
| 4095 | 55 | 11e7 | 13.1 s | 4.1 s | 2 | 5.1 s | 91% | 97% |
| 4095 | 60 | 26e7 | 31.0 s | 7.1 s | 2 | 10.3 s | 90% | 97% |

The best B2 is 1 to 10 times GMP-ECM's default and saves 0–12% of the time per factor,
mostly under 5%; the jumps between neighbouring rows come from the discrete choice of dF and
k. GMP-ECM's default B2 is therefore a good choice for GPU stage 1 with gpuecm2 stage 2, and
two to three times the default is a small, safe gain.

Running stage 1 on CPU cores and only stage 2 on the GPU also works (gpuecm2 reads the
residues of any GMP-ECM run); the GPU then idles at the default B2 and a larger B2 (`ecm`'s
B2 argument, the same for gpuecm2) makes better use of it. Such B2 values are tested against
GMP-ECM (docs/TESTING.md: full B2 up to 1e13, B2min–B2 ranges up to 1e39).
