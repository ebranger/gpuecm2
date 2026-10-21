# Using gpuecm2 from other programs (GMP-ECM integration)

gpuecm2 is also a library, `libgpuecm2`, with a small C interface in
[`src/api/gpuecm2.h`](../src/api/gpuecm2.h). Given GMP-ECM's stage 2 parameters
(dF, k, d1, d2, i0, S) and a batch of curves of one number, it returns per curve exactly
what GMP-ECM's `stage2()` would return: the same factor, or no factor.

## Building and linking

```
cmake -S . -B build && cmake --build build -j 2
cmake --install build --prefix /opt/gpuecm2      # bin/gpuecm2, lib/libgpuecm2.so, include/gpuecm2.h
cc prog.c -I/opt/gpuecm2/include -L/opt/gpuecm2/lib -lgpuecm2 -lgmp
```

The shared library exports only the `gpuecm2_*` functions and brings its own copy of the CUDA
runtime (static), so a program using it needs no CUDA toolkit, only the NVIDIA driver.
On Windows the library is built as a static library instead.

## The interface

```c
gpuecm2_params p;                 /* dF, k, d1, d2, i0, S (and B2min, B2 for information) */
gpuecm2_params_init(p);
/* either GMP-ECM's own values ... */
p->dF = dF; p->k = k; p->d1 = root_params.d1; p->d2 = root_params.d2;
mpz_set(p->i0, root_params.i0); p->S = root_params.S;
/* ... or gpuecm2's port of GMP-ECM's choice (bestD/choose_S), fitted to the GPU memory */
gpuecm2_choose_params(p, n, B1, B2min_or_NULL, B2_or_NULL, k_or_0, S_or_0, maxmem_or_0,
                      1 /* ntt: GMP-ECM default rule */, err, sizeof err);

gpuecm2_stage2 *s = gpuecm2_stage2_new(n, p, err, sizeof err);   /* per number; reuse it */
int nfound = gpuecm2_stage2_run(s, count, A, x, status, f, NULL, 0);
gpuecm2_stage2_free(s);
gpuecm2_params_clear(p);
```

- A curve is GMP-ECM's Montgomery curve `B y^2 = x^3 + A x^2 + x` with the stage 1 point
  `(x : 1)`, which is what GMP-ECM has for parametrizations 0 to 3 at the end of stage 1,
  just before `montgomery_to_weierstrass()`. The torsion curves of `-torsion` (Hessian and
  twisted Hessian models) are not supported; for them GMP-ECM keeps its CPU stage 2.
- `status[i]` is `GPUECM2_NO_FACTOR` (0), `GPUECM2_FACTOR_FOUND` (2, factor in `f[i]`, which
  may equal n), `GPUECM2_ERROR` (-1) or `GPUECM2_CHECK_FAILED` (-2, only with the self-check
  flag). These are GMP-ECM's ECM_NO_FACTOR_FOUND, ECM_FACTOR_FOUND_STEP2 and ECM_ERROR codes.
  A curve for which the conversion to Weierstrass form meets a non-invertible value reports
  that factor, as GMP-ECM does.
- `gpuecm2_stage2_run` splits large batches into pieces that fit the GPU memory.
  `gpuecm2_stage2_batch_size(s)` tells how many curves per call keep the GPU busy;
  for B1 below about 1e6 that is tens to hundreds of curves, and passing them together is
  what makes small stage 2s fast (see docs/BENCHMARKS.md).
- `gpuecm2_stage2_new` costs some milliseconds (NTT primes, root plans). Keep the engine
  while the number and parameters stay the same.
- `GPUECM2_SELFCHECK` verifies each curve's product trees and final values at random points,
  and a few roots of the first curve of each batch against a CPU computation (5–20% slower).
- Not thread-safe: call from one thread (one engine per thread and GPU does work, but the
  default CUDA device of each thread must be set with `gpuecm2_set_device`).
- A CUDA error (e.g. the GPU disappears) terminates the process with a message, as
  GMP-ECM's own GPU code does.

`tests/test_capi.c` is a minimal C program using the interface; `tests/capi_resume.c` runs
stage 2 for a GMP-ECM save file through the interface only (it is also used by the tests:
`tests/run_hits.py build/capi_resume` reproduces the known hits through the library).

## Where GMP-ECM calls it

The following describes the two places in GMP-ECM 7.0.x where its CPU stage 2 is called.
The first is implemented for GMP-ECM 7.0.6 by [`gmp-ecm/0001-gpu2.patch`](../gmp-ecm/0001-gpu2.patch)
(function `stage2_gpuecm2` in `cudawrapper.c`): built with `-DHAVE_GPUECM2` and linked with
`-lgpuecm2`, `ecm` gets the option `-gpu2`. [`gmp-ecm/build_ecm.sh`](../gmp-ecm/build_ecm.sh)
makes such a build; see [gmp-ecm/README.md](../gmp-ecm/README.md) and the README's
"Full GPU build of GMP-ECM". The second is not patched: for CPU stage 1 the standalone
`gpuecm2 -resume` serves.

1. **GPU stage 1 (`-gpu`), `cudawrapper.c`, `gpu_ecm()`.** After stage 1 on the GPU, a loop
   over the `nb_curves` curves sets `P.x` to the curve's result, `P.A` from its sigma
   (`A_from_sigma`), converts to Weierstrass form and calls `stage2()` for each curve.
   With gpuecm2 all curves go to the GPU at once:

   ```c
   /* before the loop, once per number */
   gpuecm2_params gp;  gpuecm2_params_init(gp);
   gp->dF = dF; gp->k = params->k; gp->d1 = root_params.d1; gp->d2 = root_params.d2;
   gp->S = root_params.S; mpz_set(gp->i0, root_params.i0);
   gpuecm2_stage2 *g2 = gpuecm2_stage2_new(n, gp, err, sizeof err);
   if (g2 != NULL) {
     /* A[i] = A_from_sigma(i + firstsigma), x[i] = factors[i] (the stage 1 results) */
     gpuecm2_stage2_run(g2, nb_curves, A, x, status, factors, NULL, 0);
     /* then report as the existing loop does: status[i] == GPUECM2_FACTOR_FOUND
        means a factor factors[i] in step 2 with curve i */
   }
   /* else: parameters with a larger k that fit (see below), or the existing CPU loop */
   ```

   Here the batch matters most: GPU stage 1 produces thousands of curves of one number, and
   gpuecm2 processes them in batches of the size it chooses.

2. **CPU stage 1, `ecm.c`, `ecm()`.** For Montgomery curves (`E->type ==
   ECM_EC_TYPE_MONTGOMERY`, parametrizations 0-3) GMP-ECM calls
   `montgomery_to_weierstrass(f, P.x, P.y, P.A, modulus)` and then
   `stage2(f, &P, modulus, dF, k, &root_params, ...)`. With gpuecm2, take
   `x = mpres_get_z(P.x)` and `A = mpres_get_z(P.A)` before the conversion and call
   `gpuecm2_stage2_run` for this one curve (`count = 1`), keeping the engine in a static
   cache keyed by n and the parameters, because `ecm()` is called once per curve. The
   result maps to `youpi` directly (`GPUECM2_FACTOR_FOUND` is `ECM_FACTOR_FOUND_STEP2`).

In both places GMP-ECM's own dF, k and root parameters are passed unchanged, so the GPU
covers exactly the primes the CPU would. If `gpuecm2_stage2_new` returns NULL because dF
does not fit in the GPU memory, `gpuecm2_choose_params` with the same B1, B2 and options
gives the parameters GMP-ECM would choose with the smallest `-k` that fits; the `-gpu2` patch
does this, and uses the CPU loop only if that fails too.

GMP-ECM's verbose output that depends on stage 2 internals ("Product of G(f_i)" at `-v -v`)
is available as the `value` output of `gpuecm2_stage2_run`.

## Licence

gpuecm2 is LGPL-3.0-or-later, so it can be linked into GMP-ECM (LGPL library, GPL program)
and into other programs under the usual LGPL conditions.
