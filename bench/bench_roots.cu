/* Benchmark of the stage 2 root generation with GMP-ECM's parameters.
   Usage: bench_roots <bits> <B1> [B2]   (random curve and point) */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>

#include <gmp.h>

#include "gpu/cuda_check.h"
#include "gpu/roots.h"
#include "host/params.h"
#include "host/roots_plan.h"

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: bench_roots bits B1 [B2]\n"); return 1; }
  const int bits = atoi(argv[1]);
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 1);
  MontgomeryCurve E;
  {  // RSA-like modulus: no small factors that would end the computation early
    Z p, q;
    mpz_urandomb(p.get(), rs, bits / 2); mpz_setbit(p.get(), bits / 2 - 1); mpz_nextprime(p.get(), p.get());
    mpz_urandomb(q.get(), rs, bits - bits / 2); mpz_setbit(q.get(), bits - bits / 2 - 1); mpz_nextprime(q.get(), q.get());
    mpz_mul(E.n.get(), p.get(), q.get());
  }
  mpz_urandomm(E.A.get(), rs, E.n.get());
  mpz_urandomm(E.u.get(), rs, E.n.get());
  Stage2Request req;
  req.B1 = atof(argv[2]);
  if (argc > 3) mpz_set_d(req.B2.get(), atof(argv[3]));
  Stage2Params sp;
  std::string err;
  if (!compute_stage2_params(sp, req, E.n.get(), &err)) { printf("%s\n", err.c_str()); return 1; }
  RnsContext ctx;
  ctx.init(E.n.get(), 16, 12, &err);
  printf("%d-bit N, B1=%s: dF=%lu, k=%lu, S=%d\n", bits, argv[2], sp.dF, sp.k, sp.S);
  Z f;
  for (int which = 0; which < 2; which++) {
    const size_t target = std::max<size_t>(256, 8192 / std::abs(sp.S));
    double t0 = now();
    ProgressionPlan pl = which == 0 ? plan_roots_F(sp, target) : plan_roots_G(sp, target);
    double tp = now() - t0;
    uint32_t *d;
    CUDA_CHECK(cudaMalloc(&d, (size_t)ctx.nlimbs() * pl.count * 4));
    compute_roots(ctx, E, pl, d, f.get());  // warm-up
    t0 = now();
    int r = compute_roots(ctx, E, pl, d, f.get());
    CUDA_CHECK(cudaDeviceSynchronize());
    double tg = now() - t0;
    cudaFree(d);
    printf("  %s roots: %zu values, %zu progressions x %zu steps: plan %.0f ms (host), GPU %.0f ms%s\n",
           which ? "G" : "F", pl.count, pl.nprog, pl.steps, tp * 1e3, tg * 1e3, r ? " [factor]" : "");
  }
  return 0;
}
