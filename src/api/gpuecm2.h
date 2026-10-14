/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   gpuecm2.h: C interface of gpuecm2, ECM stage 2 on an NVIDIA GPU with
   GMP-ECM's algorithm and parameters.

   Typical use (see docs/INTEGRATION.md for GMP-ECM):

     gpuecm2_params p;
     gpuecm2_params_init(p);
     gpuecm2_choose_params(p, n, B1, NULL, NULL, 0, 0, 0, 1, err, sizeof err);
     gpuecm2_stage2 *s = gpuecm2_stage2_new(n, p, err, sizeof err);
     gpuecm2_stage2_run(s, count, A, x, status, f, NULL, 0);
     gpuecm2_stage2_free(s);
     gpuecm2_params_clear(p);

   An engine (gpuecm2_stage2) is made for one modulus and one parameter set
   and is meant to be reused for every curve of that number: making it
   selects the NTT primes and prepares the root plans. Curves are passed in
   GMP-ECM's Montgomery form, as GMP-ECM has them at the end of stage 1 for
   parametrizations 0-3 (just before montgomery_to_weierstrass), and the
   result for every curve is exactly what GMP-ECM's stage2() would give for
   the same dF, k, d1, d2, i0 and S.

   The functions are not thread-safe: use one thread per process (or guard
   the calls). They use the current CUDA device of the calling thread. */
#ifndef GPUECM2_H
#define GPUECM2_H

#include <stddef.h>
#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPUECM2_API_VERSION 1

/* result codes; per curve and of the calls (the first three as GMP-ECM's
   ECM_NO_FACTOR_FOUND, ECM_FACTOR_FOUND_STEP2 and ECM_ERROR) */
#define GPUECM2_NO_FACTOR 0
#define GPUECM2_FACTOR_FOUND 2   /* factor in f; may equal n */
#define GPUECM2_ERROR (-1)
#define GPUECM2_CHECK_FAILED (-2) /* GPUECM2_SELFCHECK found an inconsistency (GPU fault) */

/* flags of gpuecm2_stage2_run */
#define GPUECM2_SELFCHECK 1u     /* verify product trees and final values at random points (a few % slower) */

/* Stage 2 parameters as GMP-ECM's stage2() takes them: dF and k, and the
   fields of root_params (d1, d2, i0, S). Stage 2 then covers the primes
   f(i d1) +- f(j d2) with i0 <= i, f(x) = x^S (S > 0) or Dickson(|S|, -1)
   (S < 0); B2min and B2 are the effective bounds (informational). */
typedef struct {
  unsigned long dF, k, d1, d2;
  mpz_t i0;
  int S;
  mpz_t B2min, B2;
} gpuecm2_params_struct;
typedef gpuecm2_params_struct gpuecm2_params[1];

void gpuecm2_params_init(gpuecm2_params p);
void gpuecm2_params_clear(gpuecm2_params p);

/* GMP-ECM's choice of stage 2 parameters for modulus n and stage 1 bound
   B1 (bestD and choose_S, as `ecm` computes them): B2min / B2 NULL or
   negative for the defaults, k = 0 and S = 0 for the defaults, maxmem
   GMP-ECM's -maxmem in bytes (0: none), ntt 1 for GMP-ECM's default
   (power-of-two dF up to 1920 bits), 0 for ecm -no-ntt, 2 for ecm -ntt.
   In addition the parameters are
   made to fit the free GPU memory: if GMP-ECM's dF does not fit, the
   choice for the smallest larger k that fits is used (as with ecm -k).
   Returns 0, or GPUECM2_ERROR with a message in err (if err != NULL). */
int gpuecm2_choose_params(gpuecm2_params p, const mpz_t n, double B1, const mpz_t B2min, const mpz_t B2,
                          unsigned long k, int S, double maxmem, int ntt, char *err, size_t errlen);

typedef struct gpuecm2_stage2 gpuecm2_stage2;

/* An engine for modulus n (odd, at most 4608 bits) and parameters p.
   Returns NULL with a message in err if that is not possible. */
gpuecm2_stage2 *gpuecm2_stage2_new(const mpz_t n, const gpuecm2_params p, char *err, size_t errlen);
void gpuecm2_stage2_free(gpuecm2_stage2 *s);

/* Stage 2 for count curves. Curve i is the Montgomery curve
     B y^2 = x^3 + A[i] x^2 + x
   with the stage 1 point (x[i] : 1), residues mod n. status[i] receives a
   result code; f[i] the factor if status[i] == GPUECM2_FACTOR_FOUND (also
   when the curve could not be converted to Weierstrass form, which GMP-ECM
   reports the same way), GPUECM2_ERROR for a singular curve (A^2 = 4 modulo
   a factor of n). value[i] (if value != NULL) receives GMP-ECM's
   "Product of G(f_i)" debug value (H(-f_0)), or 0 when not computed.
   The curves are processed in batches that fit the GPU memory.
   Returns the number of curves with a factor, or GPUECM2_ERROR. */
int gpuecm2_stage2_run(gpuecm2_stage2 *s, size_t count, mpz_t *A, mpz_t *x, int *status, mpz_t *f, mpz_t *value,
                       unsigned flags);

/* Good number of curves per gpuecm2_stage2_run call: enough to keep the GPU
   busy when one curve is too small (small dF), within the free memory. */
size_t gpuecm2_stage2_batch_size(gpuecm2_stage2 *s);

/* Curve setup of GMP-ECM's parametrizations 0-3 from sigma: A and the
   stage 1 starting point x0. Returns 0, GPUECM2_FACTOR_FOUND (factor in f)
   or GPUECM2_ERROR (invalid sigma). For tools that only have sigma. */
int gpuecm2_curve_from_sigma(mpz_t f, mpz_t A, mpz_t x0, int param, const mpz_t sigma, const mpz_t n);

/* Select the CUDA device (call before the other functions); 0 on success. */
int gpuecm2_set_device(int device);
/* How the calling thread waits for the GPU: 0 (default) sleeps, leaving the
   core to other work such as CPU stage 1; 1 busy-waits. Call first. */
void gpuecm2_cpu_wait(int spin);

const char *gpuecm2_version(void);

#ifdef __cplusplus
}
#endif
#endif /* GPUECM2_H */
