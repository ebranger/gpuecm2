/* Benchmark of the polynomial part of ECM stage 2 with GMP-ECM's structure:
   F tree, 1/F, k blocks of (G from roots, H = (G-F) H mod F), evaluation.
   Roots are random; the curve arithmetic (root generation) is not included.
   Usage: bench_poly <bits> <dF> <k> [rns]  (rns: RNS storage of the polynomials) */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <gmp.h>

#include "gpu/cuda_check.h"
#include "gpu/limbops.h"
#include "gpu/polyalg.h"
#include "gpu/rns.h"
#include "host/mpz_class_lite.h"

static void pool_report(const char *phase) {
  cudaMemPool_t pool;
  CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, 0));
  uint64_t reserved_hi = 0, used_hi = 0, zero = 0;
  cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &reserved_hi);
  cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_hi);
  cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &zero);
  cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero);
  if (getenv("BENCH_MEM")) printf("  [%s: pool used high %.2f GB, reserved high %.2f GB]\n", phase, used_hi / 1e9, reserved_hi / 1e9);
}

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static uint32_t *random_vec(const RnsContext &ctx, size_t n, const mpz_t N, gmp_randstate_t rs) {
  const int nl = ctx.nlimbs();
  std::vector<uint32_t> h((size_t)nl * n, 0), tmp(nl + 2);
  Z x;
  for (size_t g = 0; g < n; g++) {
    mpz_urandomm(x.get(), rs, N);
    std::fill(tmp.begin(), tmp.end(), 0);
    size_t cnt;
    mpz_export(tmp.data(), &cnt, -1, 4, 0, 0, x.get());
    for (int j = 0; j < nl; j++) h[(size_t)j * n + g] = tmp[j];
  }
  uint32_t *d;
  CUDA_CHECK(cudaMalloc(&d, (size_t)ctx.coef_words() * n * 4));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
  ctx.import_limbs(d, n, n, d, n);
  return d;
}

int main(int argc, char **argv) {
  if (argc < 4) { fprintf(stderr, "usage: bench_poly bits dF k [rns]\n"); return 1; }
  const bool rns = argc > 4 && std::string(argv[4]) == "rns";
  const int bits = atoi(argv[1]);
  const size_t n = (size_t)atol(argv[2]);
  const int k = atoi(argv[3]);
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 5);
  Z N;
  mpz_urandomb(N.get(), rs, bits); mpz_setbit(N.get(), bits - 1); mpz_setbit(N.get(), 0);
  {  // as Stage2Engine: the stream-ordered pool keeps its memory between synchronisations
    cudaMemPool_t pool;
    uint64_t keep = UINT64_MAX;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess)
      cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep);
  }
  RnsContext ctx;
  std::string err;
  int lg = 0;
  while (((size_t)1 << lg) < 2 * n) lg++;
  if (!ctx.init(N.get(), 2 * n + 2, lg, &err, rns)) { printf("init failed: %s\n", err.c_str()); return 1; }
  cudaDeviceProp pr;
  cudaGetDeviceProperties(&pr, 0);
  printf("%s: %d-bit N, dF=%zu, k=%d, %d primes, Tmax=2^%d%s\n", pr.name, bits, n, k, ctx.P(), ctx.logTmax(),
         rns ? ", RNS storage" : "");

  uint32_t *rootsF = random_vec(ctx, n, N.get(), rs);
  std::vector<uint32_t *> rootsG;
  for (int b = 0; b < k; b++) rootsG.push_back(random_vec(ctx, n, N.get(), rs));
  uint32_t *F = limb_alloc(ctx, n), *I = limb_alloc(ctx, n), *G = limb_alloc(ctx, n), *H = limb_alloc(ctx, n),
           *vals = limb_alloc(ctx, n);
  CUDA_CHECK(cudaDeviceSynchronize());

  for (int rep = 0; rep < 2; rep++) {  // first round warms up the memory pool
    double t0 = now(), tF, tI, tG = 0, tM = 0, tE;
    ProductTree tree;
    tree_build(ctx, rootsF, n, F, &tree);
    CUDA_CHECK(cudaDeviceSynchronize()); tF = now() - t0; pool_report("F tree"); t0 = now();
    poly_invert(ctx, F, n, I);
    CUDA_CHECK(cudaDeviceSynchronize()); tI = now() - t0; pool_report("1/F");
    {
      MulModF mm(ctx, F, I, n);
      for (int b = 0; b < k; b++) {
        t0 = now();
        tree_build(ctx, rootsG[b], n, G, nullptr);
        limb_sub(ctx, G, n, F, n, G, n, n);  // G mod F = G - F
        CUDA_CHECK(cudaDeviceSynchronize()); tG += now() - t0; t0 = now();
        if (b == 0) limb_copy(ctx, G, n, H, n, n, false);
        else mm.mulmod(G, H);
        CUDA_CHECK(cudaDeviceSynchronize()); tM += now() - t0;
      }
    }
    t0 = now();
    poly_eval_tree(ctx, tree, I, H, vals);
    CUDA_CHECK(cudaDeviceSynchronize()); tE = now() - t0; pool_report("polyeval");
    tree_free(tree);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (rep == 1)
      printf("F tree %.1f ms, 1/F %.1f ms, G trees %.1f ms (%.1f each), G*H mod F %.1f ms, polyeval %.1f ms"
             " -> total %.0f ms\n", tF * 1e3, tI * 1e3, tG * 1e3, tG * 1e3 / k, tM * 1e3, tE * 1e3,
             (tF + tI + tG + tM + tE) * 1e3);
  }
  return 0;
}
