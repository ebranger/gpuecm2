/* Test of the C interface (gpuecm2.h), written in C as a GMP-ECM integration
   would use it: a known stage 2 hit (a c108 of stdkmd.net's 300...008 table,
   also in tests/data/stage2_hits.txt) must find its factor, alone and
   inside a batch, with and without the self-check. */
#include <stdio.h>
#include <string.h>

#include <gmp.h>

#include "api/gpuecm2.h"

static int failures = 0;
#define CHECK(c)                                                         \
  do {                                                                   \
    if (!(c)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); failures++; } \
  } while (0)

int main(void) {
  /* stage 1 residue of GMP-ECM 7.0.6, -sigma 1:3264855362, B1 = 10000 */
  const char *N = "840950548787773567674834886874087725793541531204321772077883170780270982948848586313979245442596902286522491";
  const char *X = "3236feb15d92fe9e3fa289a2f35c28eaec98aedb799b38d82f6941c5c8f9574eec750fa788a5b73b6c5a2a9104";
  const char *FACTOR = "113817707364469273241034443";
  char err[256] = "";
  mpz_t n, sigma, B2, A[3], x[3], f[3], v[3], x0, fac, expect;
  int status[3];
  gpuecm2_params p;
  int i;

  printf("gpuecm2 %s, API version %d\n", gpuecm2_version(), GPUECM2_API_VERSION);
  mpz_inits(n, sigma, B2, x0, fac, expect, NULL);
  for (i = 0; i < 3; i++) mpz_inits(A[i], x[i], f[i], v[i], NULL);
  mpz_set_str(n, N, 10);
  mpz_set_str(expect, FACTOR, 10);
  mpz_set_ui(sigma, 3264855362UL);
  mpz_set_ui(B2, 1873422);

  /* parameters as `ecm -power 1 10000 1873422` chooses them */
  gpuecm2_params_init(p);
  CHECK(gpuecm2_choose_params(p, n, 10000, NULL, B2, 0, 1, 0, 1, err, sizeof err) == 0);
  CHECK(p->dF == 256 && p->k == 3 && p->d1 == 2310 && p->d2 == 13 && p->S == 1);
  gmp_printf("dF=%lu, k=%lu, d=%lu, d2=%lu, i0=%Zd, S=%d, B2=%Zd\n", p->dF, p->k, p->d1, p->d2, p->i0, p->S, p->B2);

  /* the curve from sigma; the point from the residue */
  CHECK(gpuecm2_curve_from_sigma(fac, A[0], x0, 1, sigma, n) == 0);
  mpz_set_str(x[0], X, 16);

  gpuecm2_stage2 *s = gpuecm2_stage2_new(n, p, err, sizeof err);
  CHECK(s != NULL);
  if (!s) { printf("%s\n", err); return 1; }
  printf("batch size %zu\n", gpuecm2_stage2_batch_size(s));

  /* one curve */
  CHECK(gpuecm2_stage2_run(s, 1, A, x, status, f, v, 0) == 1);
  CHECK(status[0] == GPUECM2_FACTOR_FOUND && mpz_cmp(f[0], expect) == 0);

  /* a batch: the hit, another point on the same curve (no factor expected),
     the hit again; with the self-check */
  mpz_set(A[1], A[0]); mpz_add_ui(x[1], x[0], 1);
  mpz_set(A[2], A[0]); mpz_set(x[2], x[0]);
  for (int flags = 0; flags <= 1; flags++) {
    CHECK(gpuecm2_stage2_run(s, 3, A, x, status, f, v, flags ? GPUECM2_SELFCHECK : 0) == 2);
    CHECK(status[0] == GPUECM2_FACTOR_FOUND && mpz_cmp(f[0], expect) == 0);
    CHECK(status[1] == GPUECM2_NO_FACTOR && mpz_sgn(v[1]) != 0);
    CHECK(status[2] == GPUECM2_FACTOR_FOUND && mpz_cmp(f[2], expect) == 0);
  }
  /* a singular curve (A = 2) is refused, the other curves of the batch are not affected */
  mpz_set_ui(A[1], 2);
  CHECK(gpuecm2_stage2_run(s, 3, A, x, status, f, v, 0) == 2);
  CHECK(status[0] == GPUECM2_FACTOR_FOUND && status[1] == GPUECM2_ERROR && status[2] == GPUECM2_FACTOR_FOUND);
  gpuecm2_stage2_free(s);

  /* parameters GMP-ECM would never make are refused */
  {
    gpuecm2_params q;
    gpuecm2_params_init(q);
    q->dF = p->dF; q->k = p->k; q->d1 = p->d1; q->d2 = p->d2; q->S = p->S; mpz_set(q->i0, p->i0);
    q->d1 = 2311;  /* not a multiple of 6 */
    CHECK(gpuecm2_stage2_new(n, q, err, sizeof err) == NULL);
    q->d1 = p->d1; q->dF = 100;  /* fewer roots of F than phi(d1)/2 */
    CHECK(gpuecm2_stage2_new(n, q, err, sizeof err) == NULL);
    q->dF = p->dF; q->d2 = 9;  /* not a prime */
    CHECK(gpuecm2_stage2_new(n, q, err, sizeof err) == NULL);
    q->d2 = 11;  /* divides d1 = 2310 */
    CHECK(gpuecm2_stage2_new(n, q, err, sizeof err) == NULL);
    q->d2 = p->d2; q->S = 0;
    CHECK(gpuecm2_stage2_new(n, q, err, sizeof err) == NULL);
    q->S = p->S; q->k = 0;
    CHECK(gpuecm2_stage2_new(n, q, err, sizeof err) == NULL);
    gpuecm2_params_clear(q);
  }

  /* refused inputs */
  mpz_mul_2exp(fac, n, 4600); mpz_add_ui(fac, fac, 1);
  CHECK(gpuecm2_stage2_new(fac, p, err, sizeof err) == NULL && strstr(err, "at most") != NULL);
  CHECK(gpuecm2_choose_params(p, expect, 10000, NULL, NULL, 0, 0, 0, 1, err, sizeof err) == 0);  /* odd n is fine */
  mpz_set_ui(fac, 1000);
  CHECK(gpuecm2_choose_params(p, fac, 10000, NULL, NULL, 0, 0, 0, 1, err, sizeof err) == GPUECM2_ERROR);

  gpuecm2_params_clear(p);
  for (i = 0; i < 3; i++) mpz_clears(A[i], x[i], f[i], v[i], NULL);
  mpz_clears(n, sigma, B2, x0, fac, expect, NULL);
  printf("%s (%d failures)\n", failures ? "FAIL" : "OK", failures);
  return failures != 0;
}
