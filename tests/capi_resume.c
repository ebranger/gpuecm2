/* Stage 2 for the residues of a GMP-ECM save file through the C interface
   (gpuecm2.h) only: a test of the library and an example of its use.
   Consecutive residues of the same number are run as one batch. Output per
   residue in the format of `ecm -v -v -resume` (the lines the tests read).

   Usage: capi_resume [-power n | -dickson n] [-k n] [-maxmem MB] [-no-ntt | -ntt] [-selfcheck] [-resume] file B1 [B2]
   (-v is accepted and ignored, so that tests/run_hits.py can drive it)
   N must be a decimal number in the file (no expressions); parametrizations
   0-3 with SIGMA, or curves given by A. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gmp.h>

#include "api/gpuecm2.h"

typedef struct {
  mpz_t n, x, A;
  int ok;
} residue;

/* value of field `key` (e.g. "N") in a save-file line, or NULL */
static const char *field(const char *line, const char *key, char *buf, size_t len) {
  size_t kl = strlen(key);
  for (const char *p = line; (p = strstr(p, key)) != NULL; p += kl) {
    if ((p != line && p[-1] != ' ' && p[-1] != ';') || p[kl] != '=') continue;
    const char *v = p + kl + 1, *e = strchr(v, ';');
    size_t n = e ? (size_t)(e - v) : strlen(v);
    if (n >= len) return NULL;
    memcpy(buf, v, n);
    buf[n] = 0;
    return buf;
  }
  return NULL;
}

static int parse(const char *line, residue *r) {
  static char buf[1 << 16];
  mpz_t sigma, f, x0;
  int param = 0, ret = 0;
  if (!field(line, "METHOD", buf, sizeof buf) || strcmp(buf, "ECM") != 0) return 0;
  if (!field(line, "N", buf, sizeof buf) || mpz_set_str(r->n, buf, 10) != 0) return 0;
  if (!field(line, "X", buf, sizeof buf) || mpz_set_str(r->x, buf, 0) != 0) return 0;
  if (field(line, "PARAM", buf, sizeof buf)) param = atoi(buf);
  mpz_inits(sigma, f, x0, NULL);
  if (field(line, "SIGMA", buf, sizeof buf) && mpz_set_str(sigma, buf, 10) == 0)
    ret = gpuecm2_curve_from_sigma(f, r->A, x0, param, sigma, r->n) == 0;
  else if (field(line, "A", buf, sizeof buf) && mpz_set_str(r->A, buf, 10) == 0)
    ret = 1;
  mpz_clears(sigma, f, x0, NULL);
  return ret;
}

int main(int argc, char **argv) {
  int S = 0, i, selfcheck = 0, ntt = 1;
  unsigned long k = 0;
  double maxmem = 0;  /* bytes, as gpuecm2_choose_params takes it (-maxmem: MB, as ecm) */
  const char *file = NULL, *b1 = NULL, *b2 = NULL;
  for (i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-power") && i + 1 < argc) S = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-dickson") && i + 1 < argc) S = -atoi(argv[++i]);
    else if (!strcmp(argv[i], "-k") && i + 1 < argc) k = strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "-maxmem") && i + 1 < argc) maxmem = atof(argv[++i]) * 1048576.;
    else if (!strcmp(argv[i], "-selfcheck")) selfcheck = 1;
    else if (!strcmp(argv[i], "-resume") && i + 1 < argc) file = argv[++i];  /* as gpuecm2 / ecm */
    else if (!strcmp(argv[i], "-v")) continue;
    else if (!strcmp(argv[i], "-no-ntt")) ntt = 0;
    else if (!strcmp(argv[i], "-ntt")) ntt = 2;
    else if (argv[i][0] == '-' && (argv[i][1] < '0' || argv[i][1] > '9')) {
      fprintf(stderr, "capi_resume: unknown option %s\n", argv[i]);
      return 1;
    }
    else if (!file) file = argv[i];
    else if (!b1) b1 = argv[i];
    else if (!b2) b2 = argv[i];
  }
  if (!file || !b1) {
    fprintf(stderr, "usage: capi_resume [-power n | -dickson n] [-k n] [-no-ntt | -ntt] [-selfcheck] file B1 [B2]\n");
    return 1;
  }
  FILE *fp = fopen(file, "r");
  if (!fp) { perror(file); return 1; }
  /* all residues */
  size_t cap = 16, cnt = 0;
  residue *R = malloc(cap * sizeof *R);
  static char line[1 << 17];
  while (fgets(line, sizeof line, fp)) {
    if (cnt == cap) R = realloc(R, (cap *= 2) * sizeof *R);
    mpz_inits(R[cnt].n, R[cnt].x, R[cnt].A, NULL);
    R[cnt].ok = parse(line, &R[cnt]);
    if (R[cnt].ok || strstr(line, "METHOD")) cnt++;
    else mpz_clears(R[cnt].n, R[cnt].x, R[cnt].A, NULL);
  }
  fclose(fp);

  mpz_t B2;
  mpz_init_set_si(B2, -1);
  if (b2) mpz_set_str(B2, b2, 10);
  char err[256];
  int found_any = 0;
  for (size_t at = 0; at < cnt;) {
    size_t end = at + 1;
    while (end < cnt && R[at].ok && R[end].ok && mpz_cmp(R[end].n, R[at].n) == 0) end++;
    const size_t m = end - at;
    if (!R[at].ok) {
      printf("Resuming ECM residue (line %zu)\nError: cannot use this line\n", at + 1);
      at = end;
      continue;
    }
    gpuecm2_params p;
    gpuecm2_params_init(p);
    gpuecm2_stage2 *s = NULL;
    if (gpuecm2_choose_params(p, R[at].n, atof(b1), NULL, b2 ? B2 : NULL, k, S, maxmem, ntt, err, sizeof err) == 0)
      s = gpuecm2_stage2_new(R[at].n, p, err, sizeof err);
    mpz_t *A = malloc(m * sizeof(mpz_t)), *x = malloc(m * sizeof(mpz_t)), *f = malloc(m * sizeof(mpz_t)),
          *v = malloc(m * sizeof(mpz_t));
    int *st = malloc(m * sizeof(int));
    for (size_t j = 0; j < m; j++) {
      mpz_init_set(A[j], R[at + j].A);
      mpz_init_set(x[j], R[at + j].x);
      mpz_inits(f[j], v[j], NULL);
      st[j] = GPUECM2_ERROR;
    }
    if (s) gpuecm2_stage2_run(s, m, A, x, st, f, v, selfcheck ? GPUECM2_SELFCHECK : 0);
    for (size_t j = 0; j < m; j++) {
      printf("Resuming ECM residue (line %zu)\n", at + j + 1);
      gmp_printf("dF=%lu, k=%lu, d=%lu, d2=%lu, i0=%Zd\n", p->dF, p->k, p->d1, p->d2, p->i0);
      if (!s) printf("Error: %s\n", err);
      else if (st[j] == GPUECM2_FACTOR_FOUND) { gmp_printf("********** Factor found in step 2: %Zd\n", f[j]); found_any = 1; }
      else if (st[j] == GPUECM2_NO_FACTOR) gmp_printf("Product of G(f_i) = %Zd\n", v[j]);
      else printf("Error: stage 2 failed (%d)\n", st[j]);
      mpz_clears(A[j], x[j], f[j], v[j], NULL);
    }
    free(A); free(x); free(f); free(v); free(st);
    gpuecm2_stage2_free(s);
    gpuecm2_params_clear(p);
    at = end;
  }
  for (size_t j = 0; j < cnt; j++) mpz_clears(R[j].n, R[j].x, R[j].A, NULL);
  free(R);
  mpz_clear(B2);
  return found_any ? 2 : 0;
}
