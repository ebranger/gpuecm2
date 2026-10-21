/* M4 tests: curve arithmetic and stage 2 root generation.
   1. complete addition/doubling vs affine host arithmetic (all edge cases)
   2. the roots of F and G vs GMP-ECM's own (printed at verbosity 5), bit exact.
   Usage: test_roots [ecm-binary]   (part 2 is skipped without a GMP-ECM binary) */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <map>
#include <string>
#include <vector>

#include <gmp.h>

#include "gpu/cuda_check.h"
#include "gpu/roots.h"
#include "host/curve.h"
#include "host/params.h"
#include "host/resume.h"
#include "host/roots_plan.h"

static int failures = 0;
static gmp_randstate_t rs;

/* ---- part 1: curve formulas ---------------------------------------------- */
typedef AffinePoint AffPt;

static AffPt aff_add(const AffPt &P, const AffPt &Q, const mpz_t a, const mpz_t p) {
  if (P.inf) return Q;
  if (Q.inf) return P;
  Z l, t, u;
  AffPt R;
  if (mpz_cmp(P.x, Q.x) == 0) {
    mpz_add(t, P.y, Q.y); mpz_mod(t, t, p);
    if (mpz_sgn(t.get()) == 0) { R.inf = true; return R; }
    mpz_mul(l, P.x, P.x); mpz_mul_ui(l, l, 3); mpz_add(l, l, a);          // 3x^2 + a
    mpz_mul_ui(u, P.y, 2); mpz_invert(u, u, p); mpz_mul(l, l, u);
  } else {
    mpz_sub(l, Q.y, P.y); mpz_sub(u, Q.x, P.x); mpz_invert(u, u, p); mpz_mul(l, l, u);
  }
  mpz_mod(l, l, p);
  mpz_mul(R.x, l, l); mpz_sub(R.x, R.x, P.x); mpz_sub(R.x, R.x, Q.x); mpz_mod(R.x, R.x, p);
  mpz_sub(R.y, P.x, R.x); mpz_mul(R.y, R.y, l); mpz_sub(R.y, R.y, P.y); mpz_mod(R.y, R.y, p);
  return R;
}

static void test_formulas(int bits) {
  /* random Montgomery curve and point mod a prime, and GMP-ECM's Weierstrass model of it */
  MontgomeryCurve C;
  mpz_urandomb(C.n.get(), rs, bits); mpz_setbit(C.n.get(), bits - 1);
  mpz_nextprime(C.n.get(), C.n.get());
  mpz_urandomm(C.A.get(), rs, C.n.get());
  mpz_urandomm(C.u.get(), rs, C.n.get());
  Z fac, a, B, t;
  AffinePoint P, Q, Q2, O, mP, T2;
  mpz_set(P.x, C.u); mpz_set_ui(P.y, 1); mpz_set(a, C.A);
  montgomery_to_weierstrass(fac, P.x, P.y, a, C.n);
  Q = aff_add(P, P, a, C.n);
  Q2 = aff_add(Q, P, a, C.n);
  O.inf = true;
  mP = P; mpz_neg(mP.y, P.y); mpz_mod(mP.y, mP.y, C.n.get());
  /* the 2-torsion point (0, 0) of the Montgomery curve: x = A/(3B), y = 0 */
  mpz_add(B, C.u, C.A); mpz_mul(B, B, C.u); mpz_add_ui(B, B, 1); mpz_mul(B, B, C.u);
  mpz_mul_ui(t, B, 3); mpz_invert(t, t, C.n); mpz_mul(T2.x, t, C.A); mpz_mod(T2.x, T2.x, C.n.get());
  mpz_set_ui(T2.y, 0);
  AffinePoint PT = aff_add(P, T2, a, C.n);
  struct Case { const AffinePoint *p, *q; const char *name; } cases[] = {
      {&P, &Q, "P+Q"}, {&Q, &Q2, "Q+R"}, {&P, &P, "P+P"}, {&P, &mP, "P+(-P)"}, {&P, &O, "P+O"},
      {&O, &P, "O+P"}, {&O, &O, "O+O"}, {&P, &T2, "P+T2"}, {&T2, &T2, "T2+T2"}, {&PT, &P, "(P+T2)+P"},
      {&T2, &O, "T2+O"}};
  bool ok = true;
  auto same = [](const AffinePoint &x, const AffinePoint &y) {
    return x.inf == y.inf && (x.inf || (mpz_cmp(x.x, y.x) == 0 && mpz_cmp(x.y, y.y) == 0));
  };
  for (auto &c : cases) {
    AffinePoint R, D, M;
    bool r = ec_test_ops(C, *c.p, *c.q, R, D, M);
    AffinePoint S = aff_add(*c.p, *c.q, a, C.n), D2 = aff_add(*c.p, *c.p, a, C.n);
    bool good = r && same(R, S) && same(D, D2) && same(M, S);
    if (!good) printf("  formula case %s wrong (%s%s%s%s)\n", c.name, r ? "" : "not representable ", r && !same(R, S) ? "add " : "",
                      r && !same(D, D2) ? "dbl " : "", r && !same(M, S) ? "mixed" : "");
    ok &= good;
  }
  printf("%s curve formulas, %d-bit prime modulus\n", ok ? "ok  " : "FAIL", bits);
  failures += !ok;
}

/* The root generator's batch inversion: correct inverses (zeros are unused
   slots), and for values that are not invertible the gcd of the first one,
   also when two of them multiply to 0 mod N = p q (one divisible by p, the
   other by q) in the same chunk or the same higher-level product. */
static void test_batch_inverse(int bits) {
  Z p, q, n, f, t;
  mpz_urandomb(p, rs, bits / 2); mpz_setbit(p, bits / 2 - 1); mpz_nextprime(p, p);
  mpz_urandomb(q, rs, bits - bits / 2); mpz_setbit(q, bits - bits / 2 - 1); mpz_nextprime(q, q);
  mpz_mul(n, p, q);
  bool ok = true;
  for (size_t count : {20, 64, 300, 5000}) {  // host only; one, two, three GPU levels (chunks of 8)
    std::vector<Z> v(count), inv;
    for (size_t i = 0; i < count; i++)
      if (i % 7 != 3) do mpz_urandomm(v[i], rs, n); while (mpz_sgn(v[i]) == 0 || (mpz_gcd(t, v[i], n), mpz_cmp_ui(t, 1) != 0));
    int r = batch_inverse_test(n, v, inv, f);
    bool good = r == ROOTS_OK;
    for (size_t i = 0; good && i < count; i++)
      if (mpz_sgn(v[i])) { mpz_mul(t, v[i], inv[i]); mpz_mod(t, t, n); good = mpz_cmp_ui(t, 1) == 0; }
    if (!good) printf("  batch inverse, %zu values: wrong inverses (status %d)\n", count, r);
    ok &= good;
    /* (index of a multiple of p, index of a multiple of q or none) */
    const std::pair<size_t, size_t> cases[] = {{16, 17}, {16, 40}, {5, count - 2}, {count / 2, SIZE_MAX}};
    for (auto c : cases) {
      if (c.first >= count || (c.second != SIZE_MAX && c.second >= count)) continue;
      std::vector<Z> w(count);
      for (size_t i = 0; i < count; i++) mpz_set(w[i], v[i]);
      mpz_urandomm(t, rs, q); mpz_add_ui(t, t, 1); mpz_mul(w[c.first], p, t);   // p * (1..q): nonzero mod n
      if (c.second != SIZE_MAX) { mpz_urandomm(t, rs, p); mpz_add_ui(t, t, 1); mpz_mul(w[c.second], q, t); }
      mpz_set_ui(f, 0);
      r = batch_inverse_test(n, w, inv, f);
      const bool g2 = r == ROOTS_FACTOR && mpz_cmp(f, p) == 0;
      if (!g2) gmp_printf("  batch inverse, %zu values, multiples of p at %zu and of q at %zu: status %d, factor %Zd "
                          "(expected p)\n", count, c.first, c.second, r, f.get());
      ok &= g2;
    }
  }
  printf("%s batch inversion, %d-bit N = p q\n", ok ? "ok  " : "FAIL", bits);
  failures += !ok;
}

/* k*P mod N (affine, k may be negative); returns false if an inversion fails */
static bool host_mul(AffPt &R, const AffPt &P, const mpz_t k_in, const mpz_t a, const mpz_t N) {
  Z k;
  mpz_abs(k, k_in);
  R.inf = true;
  for (long i = (long)mpz_sizeinbase(k, 2) - 1; i >= 0; i--) {
    R = aff_add(R, R, a, N);
    if (mpz_tstbit(k, i)) R = aff_add(R, P, a, N);
  }
  if (mpz_sgn(k_in) < 0 && !R.inf) { mpz_neg(R.y, R.y); mpz_mod(R.y, R.y, N); }
  return true;
}

/* ---- part 2: roots vs GMP-ECM ------------------------------------------- */
static std::string run_cmd(const std::string &cmd) {
  std::string out;
  FILE *f = popen(cmd.c_str(), "r");
  if (!f) return out;
  char buf[8192];
  while (fgets(buf, sizeof buf, f)) out += buf;
  pclose(f);
  return out;
}

static void test_roots_vs_ecm(const char *ecm, const char *nstr, int param, unsigned long sigma, double B1,
                              const char *B2, const char *sopt) {
  const std::string tmp = "/tmp/gpuecm2_test_roots_" + std::to_string(getpid()) + ".txt";
  const std::string lim = "ulimit -v 2000000; ";
  remove(tmp.c_str());
  char cmd[4096];
  snprintf(cmd, sizeof cmd, "%secho %s | %s -q -param %d -sigma %d:%lu -save %s %.0f 1 > /dev/null 2>&1", lim.c_str(),
           nstr, ecm, param, param, sigma, tmp.c_str(), B1);
  run_cmd(cmd);
  snprintf(cmd, sizeof cmd, "%s%s -v -v -v -v -v %s -resume %s %.0f %s 2>&1", lim.c_str(), ecm, sopt, tmp.c_str(), B1, B2);
  std::string trace = run_cmd(cmd);
  /* GMP-ECM's roots: f_i, and g_i per block (index restarts each block) */
  std::vector<std::string> fv, gv;
  for (size_t pos = 0; pos < trace.size();) {
    size_t e = trace.find('\n', pos);
    std::string line = trace.substr(pos, e - pos);
    pos = e == std::string::npos ? trace.size() : e + 1;
    if (line.compare(0, 2, "f_") == 0) fv.push_back(line.substr(line.find('=') + 2));
    else if (line.compare(0, 2, "g_") == 0) gv.push_back(line.substr(line.find('=') + 2));
  }
  /* our side */
  FILE *f = fopen(tmp.c_str(), "r");
  if (!f) { printf("FAIL could not run GMP-ECM\n"); failures++; return; }
  ResumeReader rd(f);
  ResumeRecord rec;
  bool have = rd.next(rec);
  fclose(f);
  remove(tmp.c_str());
  if (!have) { printf("FAIL no save line (factor found in stage 1?)\n"); failures++; return; }
  Z A, x0, fac;
  curve_from_param(fac, A, x0, rec.param, rec.sigma, rec.n);
  Stage2Request req;
  req.B1 = B1;
  mpz_set_d(req.B2.get(), atof(B2));
  if (strncmp(sopt, "-power", 6) == 0) req.S = atoi(sopt + 7);
  if (strncmp(sopt, "-dickson", 8) == 0) req.S = -atoi(sopt + 9);
  Stage2Params sp;
  std::string err;
  compute_stage2_params(sp, req, rec.n, &err);
  MontgomeryCurve C;
  mpz_set(C.n, rec.n);
  mpz_set(C.A, A);
  mpz_set(C.u, rec.x);
  struct { Z n, a, x, y; } E;  // GMP-ECM's Weierstrass model, for the host check
  mpz_set(E.n, rec.n);
  mpz_set(E.x, rec.x);
  mpz_set_ui(E.y, 1);
  mpz_set(E.a, A);
  montgomery_to_weierstrass(fac, E.x, E.y, E.a, E.n);
  RnsContext ctx;
  ctx.init(E.n, 16, 12, &err);
  bool ok = fv.size() == sp.dF && gv.size() == sp.k * sp.dF;
  if (!ok) printf("  GMP-ECM printed %zu f and %zu g roots (expected %lu, %lu)%s\n", fv.size(), gv.size(), sp.dF, sp.k * sp.dF,
                  trace.find("actor found") != std::string::npos ? " -- it found a factor" : "");
  for (int which = 0; which < 2 && ok; which++) {
    ProgressionPlan pl = which == 0 ? plan_roots_F(sp, 64) : plan_roots_G(sp, 64);
    const int nl = ctx.nlimbs();
    std::vector<uint32_t> h((size_t)nl * pl.count), tmpl(nl);
    int r;
    if (which == 0) {  // all at once
      uint32_t *d;
      CUDA_CHECK(cudaMalloc(&d, (size_t)nl * pl.count * 4));
      r = compute_roots(ctx, C, pl, d, fac);
      CUDA_CHECK(cudaMemcpy(h.data(), d, h.size() * 4, cudaMemcpyDeviceToHost));
      cudaFree(d);
    } else {  // block by block, as stage 2 does
      RootGenerator gen;
      r = gen.init(C, plan_scalar_bits(pl), fac);
      RootPlanGPU gp(pl, sp.dF);
      if (r == ROOTS_OK) r = gen.start(gp);
      uint32_t *d;
      CUDA_CHECK(cudaMalloc(&d, (size_t)nl * sp.dF * 4));
      std::vector<uint32_t> hb((size_t)nl * sp.dF);
      for (size_t b = 0; b < sp.k && r == ROOTS_OK; b++) {
        r = gen.next_block(ctx, d, fac);
        CUDA_CHECK(cudaMemcpy(hb.data(), d, hb.size() * 4, cudaMemcpyDeviceToHost));
        for (int j = 0; j < nl; j++)
          for (size_t i = 0; i < sp.dF; i++) h[(size_t)j * pl.count + b * sp.dF + i] = hb[(size_t)j * sp.dF + i];
      }
      cudaFree(d);
    }
    if (r != ROOTS_OK) { printf("  compute_roots returned %d\n", r); ok = false; break; }
    const auto &ref = which == 0 ? fv : gv;
    Z v, w;
    for (size_t i = 0; i < pl.count && ok; i++) {
      for (int j = 0; j < nl; j++) tmpl[j] = h[(size_t)j * pl.count + i];
      mpz_import(v, ctx.nlimbs(), -1, 4, 0, 0, tmpl.data());
      mpz_set_str(w, ref[i].c_str(), 10);
      if (mpz_cmp(v, w) != 0) {
        printf("  %s root %zu differs\n", which ? "G" : "F", i);
        ok = false;
        /* which one is right?  key of root i, then f(key * d) * P on the host */
        long long key = 0;
        if (which == 0) { size_t c = 0; for (long long j = 1;; j += 6) { Z g; mpz_set_si(g, j); mpz_gcd_ui(g, g, sp.d1); if (mpz_cmp_ui(g, 1) == 0) { if (c == i) { key = j; break; } c++; } } }
        else { size_t c = 0; for (long long j = mpz_get_si(sp.i0.get());; j++) { if (((j % (long long)sp.d2) + (long long)sp.d2) % (long long)sp.d2 != 0 || sp.d2 == 1) { if (c == i) { key = j; break; } c++; } } }
        Z arg, fv2;
        mpz_set_si(arg, key);
        mpz_mul_ui(arg, arg, which == 0 ? sp.d2 : sp.d1);
        if (sp.S < 0) dickson(fv2, arg, -sp.S, -1); else mpz_pow_ui(fv2, arg, sp.S);
        AffPt P0, R;
        mpz_set(P0.x, E.x); mpz_set(P0.y, E.y);
        host_mul(R, P0, fv2, E.a, E.n);
        gmp_printf("    key %lld: gpu %s, gmp-ecm %s (host x = %Zd...)\n", key,
                   (!R.inf && mpz_cmp(R.x, v) == 0) || (R.inf && mpz_sgn(v.get()) == 0) ? "RIGHT" : "wrong",
                   (!R.inf && mpz_cmp(R.x, w) == 0) || (R.inf && mpz_sgn(w.get()) == 0) ? "RIGHT" : "wrong", R.x.get());
      }
    }
  }
  printf("%s roots vs GMP-ECM: %zu-bit N, param %d, B2=%s %s: dF=%lu k=%lu d1=%lu d2=%lu i0=%ld S=%d\n",
         ok ? "ok  " : "FAIL", mpz_sizeinbase(rec.n.get(), 2), param, B2, sopt, sp.dF, sp.k, sp.d1, sp.d2,
         mpz_get_si(sp.i0.get()), sp.S);
  failures += !ok;
}

int main(int argc, char **argv) {
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 99);
#ifdef GPUECM2_ONLY_BITS
  const int maxbits = GPUECM2_ONLY_BITS;
#else
  const int maxbits = 4608;
#endif
  /* one size per CGBN size, and exact sizes (n with as many bits as the CGBN size) */
  for (int bits : {64, 500, 512, 600, 768, 1000, 1024, 1100, 1280, 1500, 1536, 1700, 1792, 2000, 2048, 2300, 2560,
                   3000, 3072, 3300, 3584, 4000, 4096, 4500, 4608})
    if (bits <= maxbits) test_formulas(bits);
  for (int bits : {200, 1000, 2000, 4000})
    if (bits <= maxbits) test_batch_inverse(bits);
  if (argc > 1) {
    const char *ecm = argv[1];
    const char *n154 = "2025465328936207440818703294301272886064421597821177778962584974524173076992682818130942359552750438209930281680713872986662397903964218259128794011267773";
    /* 2^1061-1 cofactor and other sizes */
    /* RSA-like composites (no small factors that GMP-ECM would find) */
    auto rsa = [&](int bits, int seed) {
      Z p, q, nn;
      gmp_randstate_t r2;
      gmp_randinit_default(r2);
      gmp_randseed_ui(r2, seed);
      mpz_urandomb(p, r2, bits / 2); mpz_setbit(p, bits / 2 - 1); mpz_nextprime(p, p);
      mpz_urandomb(q, r2, bits - bits / 2); mpz_setbit(q, bits - bits / 2 - 1); mpz_nextprime(q, q);
      mpz_mul(nn, p, q);
      char *str = mpz_get_str(nullptr, 10, nn);
      std::string out = str;
      free(str);
      gmp_randclear(r2);
      return out;
    };
    std::string n1024 = rsa(1022, 3), n2048 = rsa(2046, 4), n300 = rsa(298, 5), n1024x = rsa(1024, 6);
    std::string n1600 = rsa(1600, 7), n3200 = rsa(3200, 8), n4500 = rsa(4500, 9);
    test_roots_vs_ecm(ecm, n154, 3, 12345, 1e4, "1000000", "");           // x^1, i0 <= 0
    test_roots_vs_ecm(ecm, n154, 1, 777, 2e4, "50000000", "");            // x^2
    test_roots_vs_ecm(ecm, n154, 0, 4242, 5e4, "500000000", "");          // Dickson(3)
    test_roots_vs_ecm(ecm, n1024.c_str(), 2, 9, 1e5, "3000000000", "");   // Dickson(6)
    test_roots_vs_ecm(ecm, n1024.c_str(), 3, 99, 1e4, "1000000", "-dickson 30");
    test_roots_vs_ecm(ecm, n1024.c_str(), 3, 98, 1e4, "2000000", "-dickson 12");
    if (maxbits >= 2046) test_roots_vs_ecm(ecm, n2048.c_str(), 3, 7, 1e4, "1000000", "-dickson 30");
    test_roots_vs_ecm(ecm, n300.c_str(), 3, 5, 3e3, "20000000", "-power 3");
    if (maxbits >= 1024) test_roots_vs_ecm(ecm, n1024x.c_str(), 1, 31, 1e4, "5000000", "-dickson 6");  // exactly 1024 bits
    /* one size per remaining CGBN size class (1792, 3584, 4608 bits) */
    if (maxbits >= 1600) test_roots_vs_ecm(ecm, n1600.c_str(), 3, 41, 1e4, "1000000", "-dickson 12");
    if (maxbits >= 3200) test_roots_vs_ecm(ecm, n3200.c_str(), 3, 42, 5e3, "500000", "");
    if (maxbits >= 4500) test_roots_vs_ecm(ecm, n4500.c_str(), 3, 43, 5e3, "500000", "-dickson 6");
  }
  printf("%s (%d failures)\n", failures ? "FAIL" : "OK", failures);
  return failures != 0;
}
