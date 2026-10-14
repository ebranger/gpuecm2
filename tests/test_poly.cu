/* M2 tests: RNS conversions, NTTs and polynomial multiplication over Z/nZ,
   checked against GMP. Needs a GPU (a few hundred MB). */
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <gmp.h>

#include "gpu/cuda_check.h"
#include "gpu/polymul.h"
#include "gpu/rns.h"
#include "host/mpz_class_lite.h"

static int failures = 0;
static gmp_randstate_t rs;

struct HostPolys {  // K coefficients mod n, limb-major
  int nl;
  size_t K;
  std::vector<uint32_t> limbs;
  std::vector<Z> v;
  HostPolys(int nl_, size_t K_) : nl(nl_), K(K_), limbs((size_t)nl_ * K_, 0), v(K_) {}
  void random(const mpz_t n) {
    for (size_t g = 0; g < K; g++) {
      mpz_urandomm(v[g].get(), rs, n);
      set_limbs(g);
    }
  }
  void set_limbs(size_t g) {
    std::vector<uint32_t> tmp(nl + 1, 0);
    size_t cnt;
    mpz_export(tmp.data(), &cnt, -1, 4, 0, 0, v[g].get());
    for (int j = 0; j < nl; j++) limbs[(size_t)j * K + g] = tmp[j];
  }
  void from_limbs() {
    std::vector<uint32_t> tmp(nl);
    for (size_t g = 0; g < K; g++) {
      for (int j = 0; j < nl; j++) tmp[j] = limbs[(size_t)j * K + g];
      mpz_import(v[g].get(), nl, -1, 4, 0, 0, tmp.data());
    }
  }
  uint32_t *upload() const {
    uint32_t *d;
    CUDA_CHECK(cudaMalloc(&d, limbs.size() * 4 + 4));
    CUDA_CHECK(cudaMemcpy(d, limbs.data(), limbs.size() * 4, cudaMemcpyHostToDevice));
    return d;
  }
  void download(const uint32_t *d) {
    CUDA_CHECK(cudaMemcpy(limbs.data(), d, limbs.size() * 4, cudaMemcpyDeviceToHost));
    from_limbs();
  }
};

static void random_modulus(Z &n, int bits) {
  mpz_urandomb(n.get(), rs, bits);
  mpz_setbit(n.get(), bits - 1);
  mpz_setbit(n.get(), 0);
}

static void test_roundtrip(int bits) {
  Z n;
  random_modulus(n, bits);
  RnsContext ctx;
  std::string err;
  if (!ctx.init(n.get(), 4096, 16, &err)) { printf("FAIL init %d bits: %s\n", bits, err.c_str()); failures++; return; }
  const size_t rows = 5, len = 1000, T = 1024;
  HostPolys a(ctx.nlimbs(), rows * len), b(ctx.nlimbs(), rows * len);
  a.random(n.get());
  uint32_t *dA = a.upload(), *dB = b.upload(), *X;
  CUDA_CHECK(cudaMalloc(&X, (size_t)ctx.P() * rows * T * 4));
  ctx.from_limbs(dA, a.K, rows, len, X, T);
  ctx.to_limbs(X, rows, T, 0, len, dB, b.K, 0, 0);
  CUDA_CHECK(cudaDeviceSynchronize());
  b.download(dB);
  int bad = 0;
  for (size_t g = 0; g < a.K; g++) bad += mpz_cmp(a.v[g].get(), b.v[g].get()) != 0;
  printf("%s roundtrip %4d bits: P=%d primes, NTT up to 2^%d, %d/%zu wrong\n", bad ? "FAIL" : "ok  ", bits, ctx.P(),
         ctx.logTmax(), bad, a.K);
  failures += bad != 0;
  cudaFree(dA); cudaFree(dB); cudaFree(X);
}

/* C(r) == A(r) B(r) at a random point, plus a few coefficients computed directly */
static bool check_product(const HostPolys &A, const HostPolys &B, const HostPolys &C, size_t rows, size_t la,
                          size_t lb, const mpz_t n, bool full_schoolbook) {
  const size_t lc = la + lb - 1;
  Z r, ea, eb, ec, t;
  for (size_t row = 0; row < rows; row++) {
    if (full_schoolbook) {
      std::vector<Z> ref(lc);
      for (size_t i = 0; i < la; i++)
        for (size_t j = 0; j < lb; j++) mpz_addmul(ref[i + j].get(), A.v[row * la + i].get(), B.v[row * lb + j].get());
      for (size_t k = 0; k < lc; k++) {
        mpz_mod(ref[k].get(), ref[k].get(), n);
        if (mpz_cmp(ref[k].get(), C.v[row * lc + k].get()) != 0) { printf("  row %zu coef %zu differs\n", row, k); return false; }
      }
      continue;
    }
    mpz_urandomm(r.get(), rs, n);
    auto horner = [&](Z &out, const HostPolys &P, size_t off, size_t len) {
      mpz_set_ui(out.get(), 0);
      for (size_t i = len; i-- > 0;) {
        mpz_mul(out.get(), out.get(), r.get());
        mpz_add(out.get(), out.get(), P.v[off + i].get());
        mpz_mod(out.get(), out.get(), n);
      }
    };
    horner(ea, A, row * la, la);
    horner(eb, B, row * lb, lb);
    horner(ec, C, row * lc, lc);
    mpz_mul(t.get(), ea.get(), eb.get());
    mpz_mod(t.get(), t.get(), n);
    if (mpz_cmp(t.get(), ec.get()) != 0) { printf("  row %zu: C(r) != A(r)B(r)\n", row); return false; }
    for (size_t k : {(size_t)0, lc / 2, lc - 1}) {  // direct coefficients
      mpz_set_ui(t.get(), 0);
      for (size_t i = k >= lb - 1 ? k - (lb - 1) : 0; i <= k && i < la; i++)
        mpz_addmul(t.get(), A.v[row * la + i].get(), B.v[row * lb + (k - i)].get());
      mpz_mod(t.get(), t.get(), n);
      if (mpz_cmp(t.get(), C.v[row * lc + k].get()) != 0) { printf("  row %zu coef %zu differs\n", row, k); return false; }
    }
  }
  return true;
}

static void test_mul(int bits, size_t la, size_t lb, size_t rows, int want_logT, bool schoolbook) {
  Z n;
  random_modulus(n, bits);
  RnsContext ctx;
  std::string err;
  size_t maxop = la < lb ? la : lb;
  if (!ctx.init(n.get(), maxop, want_logT, &err)) { printf("FAIL init: %s\n", err.c_str()); failures++; return; }
  const size_t lc = la + lb - 1;
  HostPolys A(ctx.nlimbs(), rows * la), B(ctx.nlimbs(), rows * lb), C(ctx.nlimbs(), rows * lc);
  A.random(n.get());
  B.random(n.get());
  uint32_t *dA = A.upload(), *dB = B.upload(), *dC = C.upload();
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  poly_mul(ctx, dA, A.K, la, dB, B.K, lb, dC, C.K, rows);
  cudaEventRecord(e1);
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  C.download(dC);
  bool ok = check_product(A, B, C, rows, la, lb, n.get(), schoolbook);
  const bool split = lc > ((size_t)1 << ctx.logTmax());
  printf("%s mul %4d bits, %zu x %zu (x%zu rows), P=%d, Tmax=2^%d%s: %.2f ms\n", ok ? "ok  " : "FAIL", bits, la, lb,
         rows, ctx.P(), ctx.logTmax(), split ? " [split]" : "", ms);
  failures += !ok;
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char **argv) {
  bool quick = argc > 1 && strcmp(argv[1], "--quick") == 0;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 2026);
  for (int bits : {30, 64, 200, 1023, 1024, 1500, 2048, 3000, 4400}) test_roundtrip(bits);

  const size_t small[][2] = {{1, 1}, {1, 5}, {2, 2}, {3, 7}, {17, 33}, {63, 2}, {64, 64}, {65, 100}, {200, 300}};
  for (int bits : {64, 1024, 2048, 4400})
    for (auto &p : small) test_mul(bits, p[0], p[1], 3, 16, true);
  test_mul(1024, 1000, 999, 2, 16, true);
  test_mul(1024, 3000, 2500, 2, 12, true);                   // split, Tmax 4096
  test_mul(1024, 5000, 5000, 2, 14, false);                  // four-step 2^14
  test_mul(4400, 3000, 3000, 1, 12, false);                  // split at 4400 bits
  if (!quick) {
    test_mul(1024, 1 << 16, 1 << 16, 1, 17, false);          // four-step 2^17
    test_mul(1024, 1 << 17, 1 << 17, 1, 16, false);          // split into 4 pieces each
    test_mul(1024, (1 << 17) + 12345, 777, 1, 18, false);    // unbalanced
    test_mul(2048, 100000, 100000, 1, 18, false);
    test_mul(4400, 1 << 15, 1 << 15, 1, 16, false);
  }
  printf("%s (%d failures)\n", failures ? "FAIL" : "OK", failures);
  return failures != 0;
}
