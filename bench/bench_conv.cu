/* Benchmark of the RNS conversions (limbs -> residues mod the NTT primes and
   back, the matrix products that dominate the polynomial arithmetic for large
   moduli), with a round-trip check.
   Usage: bench_conv [bits ...]   (default 512 1024 2048 3072 4096 4608) */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <gmp.h>

#include "gpu/cuda_check.h"
#include "gpu/rns.h"

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
  std::vector<int> sizes;
  for (int i = 1; i < argc; i++) sizes.push_back(atoi(argv[i]));
  if (sizes.empty()) sizes = {512, 1024, 2048, 3072, 4096, 4608};
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  for (int bits : sizes) {
    mpz_t n;
    mpz_init(n);
    mpz_urandomb(n, rs, bits);
    mpz_setbit(n, bits - 1);
    mpz_setbit(n, 0);
    RnsContext ctx;
    std::string err;
    const size_t dF = 1 << 15;
    if (!ctx.init(n, 2 * dF + 2, 16, &err)) { printf("%s\n", err.c_str()); return 1; }
    const int nl = ctx.nlimbs(), P = ctx.P();
    /* 2^19 coefficients, fewer for large moduli (at most 2^28 residues) */
    const size_t T = (size_t)1 << 16, rows = std::max<size_t>(1, std::min<size_t>(8, ((size_t)1 << 28) / ((size_t)P * T)));
    const size_t K = rows * T;
    std::vector<uint32_t> h((size_t)nl * K);
    for (size_t g = 0; g < K; g++) {  // random residues < n
      mpz_t x;
      mpz_init(x);
      mpz_urandomm(x, rs, n);
      size_t cnt;
      std::vector<uint32_t> w(nl, 0);
      mpz_export(w.data(), &cnt, -1, 4, 0, 0, x);
      for (int j = 0; j < nl; j++) h[(size_t)j * K + g] = w[j];
      mpz_clear(x);
    }
    uint32_t *A, *X, *B;
    CUDA_CHECK(cudaMalloc(&A, h.size() * 4));
    CUDA_CHECK(cudaMalloc(&B, h.size() * 4));
    CUDA_CHECK(cudaMalloc(&X, (size_t)P * K * 4));
    CUDA_CHECK(cudaMemcpy(A, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    double tf = 1e30, tb = 1e30;
    for (int rep = 0; rep < 4; rep++) {
      CUDA_CHECK(cudaDeviceSynchronize());
      double t = now();
      ctx.from_limbs(A, K, rows, T, X, T);
      CUDA_CHECK(cudaDeviceSynchronize());
      tf = std::min(tf, now() - t);
      t = now();
      ctx.to_limbs(X, rows, T, 0, T, B, K, 0, 0);
      CUDA_CHECK(cudaDeviceSynchronize());
      tb = std::min(tb, now() - t);
    }
    std::vector<uint32_t> back(h.size());
    CUDA_CHECK(cudaMemcpy(back.data(), B, back.size() * 4, cudaMemcpyDeviceToHost));
    const bool ok = back == h;
    const double macs = (double)K * P * nl;
    printf("%4d bits: P=%3d nl=%3d  to RNS %6.2f ns/coef (%.2f T MAC/s)  back %6.2f ns/coef (%.2f T MAC/s)  %s\n",
           bits, P, nl, tf / K * 1e9, macs / tf / 1e12, tb / K * 1e9, macs / tb / 1e12, ok ? "round trip ok" : "MISMATCH");
    cudaFree(A); cudaFree(B); cudaFree(X);
    mpz_clear(n);
  }
  return 0;
}
