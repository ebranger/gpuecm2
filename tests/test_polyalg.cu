/* M3 tests: product tree, series inversion, reduction mod F and the
   transposed multipoint evaluation, checked against GMP; with limbs and with
   RNS storage (the arrays imported from / exported to limbs). */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <gmp.h>

#include "gpu/cuda_check.h"
#include "gpu/limbops.h"
#include "gpu/polyalg.h"
#include "gpu/rns.h"
#include "host/mpz_class_lite.h"

static int failures = 0;
static gmp_randstate_t rs;
static bool rns_store = false;  // the mode of the current run

struct Vec {  // n residues, host copy + device array (ld = n) of the context's representation
  const RnsContext &ctx;
  size_t n;
  std::vector<Z> v;
  uint32_t *d = nullptr;
  Vec(const RnsContext &c, size_t n_) : ctx(c), n(n_), v(n_) { CUDA_CHECK(cudaMalloc(&d, words() * 4)); }
  size_t words() const { return (size_t)ctx.coef_words() * (n ? n : 1); }
  ~Vec() { cudaFree(d); }
  void random(const mpz_t N) { for (auto &x : v) mpz_urandomm(x.get(), rs, N); upload(); }
  void upload() {
    const int nl = ctx.nlimbs();
    std::vector<uint32_t> h((size_t)nl * n, 0), tmp(nl + 2);
    for (size_t g = 0; g < n; g++) {
      std::fill(tmp.begin(), tmp.end(), 0);
      size_t cnt;
      mpz_export(tmp.data(), &cnt, -1, 4, 0, 0, v[g].get());
      for (int j = 0; j < nl; j++) h[(size_t)j * n + g] = tmp[j];
    }
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    ctx.import_limbs(d, n, n, d, n);  // in place (coef_words() >= nlimbs())
  }
  void download() {
    const int nl = ctx.nlimbs();
    std::vector<uint32_t> h((size_t)nl * n), tmp(nl);
    uint32_t *e;
    CUDA_CHECK(cudaMalloc(&e, words() * 4));
    ctx.export_limbs(d, n, n, e, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h.data(), e, h.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(e));
    for (size_t g = 0; g < n; g++) {
      for (int j = 0; j < nl; j++) tmp[j] = h[(size_t)j * n + g];
      mpz_import(v[g].get(), nl, -1, 4, 0, 0, tmp.data());
    }
  }
};

/* x^deg + sum c_i x^i (monic) or sum c_i x^i */
static void horner(Z &out, const std::vector<Z> &c, size_t len, const mpz_t x, const mpz_t N, bool monic, size_t off = 0) {
  mpz_set_ui(out.get(), monic ? 1 : 0);
  for (size_t i = len; i-- > 0;) {
    mpz_mul(out.get(), out.get(), x);
    mpz_add(out.get(), out.get(), c[off + i].get());
    mpz_mod(out.get(), out.get(), N);
  }
}

static std::vector<Z> host_from_roots(const std::vector<Z> &r, size_t off, size_t n, const mpz_t N) {
  std::vector<Z> c(n + 1);  // explicit, c[n] = 1
  mpz_set_ui(c[0].get(), 1);
  for (size_t k = 0; k < n; k++) {  // multiply by (x + r)
    for (size_t i = k + 1; i-- > 0;) {
      if (i + 1 <= k + 1) mpz_add(c[i + 1].get(), c[i + 1].get(), c[i].get());
      mpz_mul(c[i].get(), c[i].get(), r[off + k].get());
      mpz_mod(c[i].get(), c[i].get(), N);
    }
    mpz_mod(c[k + 1].get(), c[k + 1].get(), N);
  }
  c.pop_back();
  for (auto &z : c) mpz_mod(z.get(), z.get(), N);
  return c;
}

static std::vector<size_t> sample(size_t n, bool full) {
  std::vector<size_t> idx;
  if (full) { for (size_t i = 0; i < n; i++) idx.push_back(i); return idx; }
  for (size_t i : {(size_t)0, (size_t)1, n / 3, n / 2, n - 2, n - 1}) if (i < n) idx.push_back(i);
  for (int k = 0; k < 6; k++) idx.push_back(gmp_urandomm_ui(rs, n));
  return idx;
}

static void report(bool ok, const char *what, int bits, size_t n, int logTmax, const char *extra = "") {
  printf("%s %-9s %4d bits, n=%-7zu Tmax=2^%d%s%s\n", ok ? "ok  " : "FAIL", what, bits, n, logTmax, extra,
         rns_store ? " [rns]" : "");
  failures += !ok;
}

static void run(int bits, size_t n, int want_logT) {
  Z N, x, y, t;
  mpz_urandomb(N.get(), rs, bits); mpz_setbit(N.get(), bits - 1); mpz_setbit(N.get(), 0);
  RnsContext ctx;
  std::string err;
  if (!ctx.init(N.get(), 2 * n + 2, want_logT, &err, rns_store)) { printf("FAIL init: %s\n", err.c_str()); failures++; return; }
  const bool full = n <= 1500, fb = n * 2 > ((size_t)1 << ctx.logTmax());
  const char *tag = fb ? " [fallback products]" : "";
  const int lt = ctx.logTmax();

  Vec roots(ctx, n), F(ctx, n);
  roots.random(N.get());
  ProductTree tree;
  tree_build(ctx, roots.d, n, F.d, &tree);
  F.download();
  bool ok = true;
  if (full) {
    auto ref = host_from_roots(roots.v, 0, n, N.get());
    for (size_t i = 0; i < n; i++) ok &= mpz_cmp(ref[i].get(), F.v[i].get()) == 0;
    /* every tree node */
    for (int l = 0; l < tree.L && ok; l++) {
      Vec lev(ctx, n);
      CUDA_CHECK(cudaMemcpy(lev.d, tree.level[l], lev.words() * 4, cudaMemcpyDeviceToDevice));
      lev.download();
      const size_t m = (size_t)1 << l;
      for (size_t j = 0; j * m < n && ok; j++) {
        size_t deg = std::min(m, n - j * m);
        auto r = host_from_roots(roots.v, j * m, deg, N.get());
        for (size_t i = 0; i < deg; i++) ok &= mpz_cmp(r[i].get(), lev.v[j * m + i].get()) == 0;
      }
    }
  } else {
    for (int k = 0; k < 3; k++) {
      mpz_urandomm(x.get(), rs, N.get());
      horner(y, F.v, n, x.get(), N.get(), true);
      mpz_set_ui(t.get(), 1);
      Z xr;
      for (size_t i = 0; i < n; i++) {
        mpz_add(xr.get(), x.get(), roots.v[i].get());
        mpz_mul(t.get(), t.get(), xr.get());
        mpz_mod(t.get(), t.get(), N.get());
      }
      ok &= mpz_cmp(y.get(), t.get()) == 0;
    }
  }
  report(ok, "tree", bits, n, lt, tag);

  /* inverse: (rev(F) * I)_t = [t == 0] */
  Vec I(ctx, n);
  poly_invert(ctx, F.d, n, I.d);
  I.download();
  ok = true;
  for (size_t tt : sample(n, n <= 300)) {
    mpz_set(t.get(), I.v[tt].get());  // R_0 = 1
    for (size_t j = 1; j <= tt; j++) mpz_addmul(t.get(), F.v[n - j].get(), I.v[tt - j].get());
    mpz_mod(t.get(), t.get(), N.get());
    ok &= mpz_cmp_ui(t.get(), tt == 0 ? 1 : 0) == 0;
  }
  report(ok, "invert", bits, n, lt, tag);

  /* mulmod: (G*H mod F)(-f_i) = G(-f_i) H(-f_i) */
  Vec G(ctx, n), H(ctx, n), H2(ctx, n);
  G.random(N.get());
  H.random(N.get());
  CUDA_CHECK(cudaMemcpy(H2.d, H.d, H2.words() * 4, cudaMemcpyDeviceToDevice));
  {
    MulModF mm(ctx, F.d, I.d, n);
    mm.mulmod(G.d, H2.d);
  }
  H2.download();
  ok = true;
  Z a, b, c;
  for (size_t i : sample(n, full && n <= 300)) {
    mpz_neg(x.get(), roots.v[i].get()); mpz_mod(x.get(), x.get(), N.get());
    horner(a, G.v, n, x.get(), N.get(), false);
    horner(b, H.v, n, x.get(), N.get(), false);
    horner(c, H2.v, n, x.get(), N.get(), false);
    mpz_mul(t.get(), a.get(), b.get()); mpz_mod(t.get(), t.get(), N.get());
    ok &= mpz_cmp(t.get(), c.get()) == 0;
  }
  report(ok, "mulmod", bits, n, lt, tag);

  /* polyeval: values[i] = H(-f_i) */
  Vec vals(ctx, n);
  poly_eval_tree(ctx, tree, I.d, H.d, vals.d);
  vals.download();
  ok = true;
  for (size_t i : sample(n, full && n <= 300)) {
    mpz_neg(x.get(), roots.v[i].get()); mpz_mod(x.get(), x.get(), N.get());
    horner(a, H.v, n, x.get(), N.get(), false);
    ok &= mpz_cmp(a.get(), vals.v[i].get()) == 0;
  }
  report(ok, "polyeval", bits, n, lt, tag);

  /* end to end (small n): H = G - F with G = prod (x + g_j); prod_i H(-f_i) = prod_{i,j} (g_j - f_i) */
  if (n <= 200) {
    Vec groots(ctx, n), Gp(ctx, n), Hp(ctx, n), v2(ctx, n);
    groots.random(N.get());
    tree_build(ctx, groots.d, n, Gp.d, nullptr);
    limb_sub(ctx, Gp.d, n, F.d, n, Hp.d, n, n);
    poly_eval_tree(ctx, tree, I.d, Hp.d, v2.d);
    v2.download();
    Z prod, ref;
    mpz_set_ui(prod.get(), 1); mpz_set_ui(ref.get(), 1);
    for (size_t i = 0; i < n; i++) {
      mpz_mul(prod.get(), prod.get(), v2.v[i].get()); mpz_mod(prod.get(), prod.get(), N.get());
      for (size_t j = 0; j < n; j++) {
        mpz_sub(t.get(), groots.v[j].get(), roots.v[i].get());
        mpz_mul(ref.get(), ref.get(), t.get()); mpz_mod(ref.get(), ref.get(), N.get());
      }
    }
    report(mpz_cmp(prod.get(), ref.get()) == 0, "e2e", bits, n, lt, tag);
  }
  tree_free(tree);
  CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char **argv) {
  /* [--quick] [--limbs | --rns] [--case bits n want_logT]; both modes by default */
  bool quick = false, modes[2] = {true, true};
  int a = 1;
  for (; a < argc && argv[a][0] == '-' && strcmp(argv[a], "--case"); a++) {
    if (!strcmp(argv[a], "--quick")) quick = true;
    else if (!strcmp(argv[a], "--limbs")) modes[1] = false;
    else if (!strcmp(argv[a], "--rns")) modes[0] = false;
  }
  gmp_randinit_default(rs);
  for (int m = 0; m < 2; m++) {
    if (!modes[m]) continue;
    rns_store = m == 1;
    gmp_randseed_ui(rs, 77);
    if (a + 4 == argc && strcmp(argv[a], "--case") == 0) {  // --case bits n want_logT
      run(atoi(argv[a + 1]), (size_t)atol(argv[a + 2]), atoi(argv[a + 3]));
      continue;
    }
    for (size_t n : {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 100, 255, 257})
      run(1024, n, 20);
    run(64, 100, 20);
    run(2048, 77, 20);
    run(4400, 50, 20);
    run(16000, 100, 20);
    run(32768, 40, 20);
    run(1024, 1000, 20);
    run(1024, 3000, 12);  // fallback products (Tmax = 4096)
    run(2048, 2500, 12);
    if (!quick) {
      run(1024, 1500, 20);
      run(1024, 5000, 12);
      run(1024, 4096, 20);
      run(1024, 65536, 20);
      run(2048, 36864, 20);   // GMP-ECM's dF for 2048-bit N, B1 = 11e6
      run(4400, 5000, 20);
      run(32768, 3000, 20);
      run(1024, 131072, 20);
    }
  }
  printf("%s (%d failures)\n", failures ? "FAIL" : "OK", failures);
  return failures != 0;
}
