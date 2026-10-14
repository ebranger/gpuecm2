/* Unit tests for the host-side code (no GPU needed). */
#include <cstdio>
#include <cstring>
#include <string>

#include "host/curve.h"
#include "host/expr.h"
#include "host/params.h"
#include "host/resume.h"
#include "host/roots_plan.h"

#include <vector>

static int failures = 0;
#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) { printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); failures++; } \
  } while (0)

static bool expr_eq(const char *e, const char *expect) {
  Z v, w;
  std::string err;
  if (!eval_expression(v.get(), e, &err)) { printf("  eval(%s) failed: %s\n", e, err.c_str()); return false; }
  mpz_set_str(w.get(), expect, 10);
  if (mpz_cmp(v.get(), w.get()) != 0) gmp_printf("  eval(%s) = %Zd, expected %s\n", e, v.get(), expect);
  return mpz_cmp(v.get(), w.get()) == 0;
}
static bool expr_fails(const char *e) {
  Z v; std::string err;
  return !eval_expression(v.get(), e, &err);
}

static void test_expr() {
  CHECK(expr_eq("2^127-1", "170141183460469231731687303715884105727"));
  CHECK(expr_eq("2^3^2", "512"));
  CHECK(expr_eq("-2^2+5", "1"));
  CHECK(expr_eq("[3+{4*(2)}]", "11"));
  CHECK(expr_eq("6.7", "42"));
  CHECK(expr_eq("10%3", "1"));
  CHECK(expr_eq("11#", "2310"));
  CHECK(expr_eq("17#5", "85085"));
  CHECK(expr_eq("14!3", "12320"));
  CHECK(expr_eq("10!", "3628800"));
  CHECK(expr_eq("Phi(3,5)", "31"));
  CHECK(expr_eq("GCD(120, 28)", "4"));
  CHECK(expr_eq("(10^20+5)/7", "14285714285714285715"));
  CHECK(expr_fails("(10^20+7)/7"));
  CHECK(expr_fails("2^"));
  CHECK(expr_fails("U(1,2,3)"));
  CHECK(expr_fails("12)"));
  /* prod_{d | n} Phi_d(x) = x^n - 1 */
  Z prod, t, ref;
  mpz_set_ui(prod.get(), 1);
  for (unsigned d = 1; d <= 105; d++)
    if (105 % d == 0) {
      std::string err;
      CHECK(eval_expression(t.get(), "Phi(" + std::to_string(d) + ",2)", &err));
      mpz_mul(prod.get(), prod.get(), t.get());
    }
  mpz_ui_pow_ui(ref.get(), 2, 105); mpz_sub_ui(ref.get(), ref.get(), 1);
  CHECK(mpz_cmp(prod.get(), ref.get()) == 0);
}

static void test_arith() {
  CHECK(eulerphi(1) == 1);
  CHECK(eulerphi(12) == 4);
  CHECK(eulerphi(2310) == 480);
  CHECK(eulerphi(1345890) == 258048);  /* 2*3*5*7*13*17*29 */
  CHECK(ceil_log2(1) == 0 && ceil_log2(2) == 1 && ceil_log2(3) == 2);
  CHECK(ceil_log2(1024) == 10 && ceil_log2(1025) == 11);
}

/* A real line written by GMP-ECM 7.0.6 (ecm -param 3 -sigma 3:12345 -save) */
static const char *SAVED =
    "METHOD=ECM; PARAM=3; SIGMA=12345; B1=10000; N=2025465328936207440818703294301272886064421597821177778962584974524173076992682818130942359552750438209930281680713872986662397903964218259128794011267773; X=0x13c0db1d95bb12e32390230e86dccb0c154882215774ebd5dacd086febfd5db63879e7bbd7d2a3d71f9ecf0c4bc1086d079fa8235b2d7445ddfe3b340d5dc5e6; CHECKSUM=3295356700; PROGRAM=GMP-ECM 7.0.6; X0=0x0; Y0=0x0; WHO=user@host; TIME=Mon Oct 12 19:33:45 2026;\n";

static int count_records(const std::string &text) {
  FILE *f = tmpfile();
  fputs(text.c_str(), f);
  rewind(f);
  ResumeReader r(f);
  ResumeRecord rec;
  int n = 0;
  while (r.next(rec)) n++;
  fclose(f);
  return n;
}

static void test_resume() {
  std::string good = SAVED;
  CHECK(count_records(good) == 1);
  CHECK(count_records("# comment\n\n" + good + "\r\n" + good) == 2);
  std::string bad = good;
  bad[bad.find("X=0x13c0") + 6] = '4';  // corrupt X -> checksum mismatch
  CHECK(count_records(bad) == 0);
  CHECK(count_records("METHOD=ECM; SIGMA=5; B1=100; X=0x5;\n") == 0);  // lacks N

  FILE *f = tmpfile();
  fputs(SAVED, f);
  rewind(f);
  ResumeReader r(f);
  ResumeRecord rec;
  CHECK(r.next(rec));
  CHECK(rec.param == 3 && rec.B1 == 10000 && mpz_cmp_ui(rec.sigma.get(), 12345) == 0);
  CHECK(rec.who == "user@host" && rec.program == "GMP-ECM 7.0.6");
  fclose(f);
}

static void test_curves() {
  /* param 3: A = 4*sigma/2^32 - 2 mod n */
  Z n, sigma, A, x0, f, t;
  mpz_set_str(n.get(), "2025465328936207440818703294301272886064421597821177778962584974524173076992682818130942359552750438209930281680713872986662397903964218259128794011267773", 10);
  mpz_set_ui(sigma.get(), 12345);
  CHECK(curve_from_param(f.get(), A.get(), x0.get(), 3, sigma.get(), n.get()) == CURVE_OK);
  mpz_ui_pow_ui(t.get(), 2, 32);
  mpz_invert(t.get(), t.get(), n.get());
  mpz_mul_ui(t.get(), t.get(), 4 * 12345);
  mpz_sub_ui(t.get(), t.get(), 2);
  mpz_mod(t.get(), t.get(), n.get());
  CHECK(mpz_cmp(A.get(), t.get()) == 0 && mpz_cmp_ui(x0.get(), 2) == 0);
  /* parametrization 0 rejects sigma in {0, 1, 3, 5} */
  mpz_set_ui(sigma.get(), 5);
  CHECK(curve_from_param(f.get(), A.get(), x0.get(), 0, sigma.get(), n.get()) == CURVE_ERROR);
  /* g*y^2 = x^3 + A*x^2 + x with y = 1 maps to Y^2 = X^3 + a*X + b,
     b = (2A^3 - 9A) / (27 g^3) */
  mpz_set_ui(sigma.get(), 7);
  CHECK(curve_from_param(f.get(), A.get(), x0.get(), 0, sigma.get(), n.get()) == CURVE_OK);
  Z x, y, a, g, b, lhs, rhs;
  mpz_set_str(x.get(), "123456789123456789", 10);
  mpz_set(a.get(), A.get());
  mpz_add(g.get(), x.get(), A.get()); mpz_mul(g.get(), g.get(), x.get());
  mpz_add_ui(g.get(), g.get(), 1); mpz_mul(g.get(), g.get(), x.get()); mpz_mod(g.get(), g.get(), n.get());
  mpz_pow_ui(t.get(), A.get(), 3); mpz_mul_ui(t.get(), t.get(), 2);
  mpz_submul_ui(t.get(), A.get(), 9);                                   /* 2A^3 - 9A */
  mpz_powm_ui(b.get(), g.get(), 3, n.get()); mpz_mul_ui(b.get(), b.get(), 27);
  CHECK(mpz_invert(b.get(), b.get(), n.get()) != 0);
  mpz_mul(b.get(), b.get(), t.get()); mpz_mod(b.get(), b.get(), n.get());
  mpz_set_ui(y.get(), 1);
  CHECK(montgomery_to_weierstrass(f.get(), x.get(), y.get(), a.get(), n.get()) == CURVE_OK);
  mpz_mul(lhs.get(), y.get(), y.get()); mpz_mod(lhs.get(), lhs.get(), n.get());
  mpz_powm_ui(rhs.get(), x.get(), 3, n.get()); mpz_addmul(rhs.get(), a.get(), x.get());
  mpz_add(rhs.get(), rhs.get(), b.get()); mpz_mod(rhs.get(), rhs.get(), n.get());
  CHECK(mpz_cmp(lhs.get(), rhs.get()) == 0);
}

/* The root plans against a brute-force list of the wanted arguments: every
   output index in [0, count) exactly once, with the argument of that rank,
   for G also with a negative i0 and an i0 far beyond 64 bits. */
static uint64_t gcd64(uint64_t a, uint64_t b) { while (b) { uint64_t t = a % b; a = b; b = t; } return a; }
static void check_plan(const ProgressionPlan &pl, const std::vector<uint64_t> &want, const char *what) {
  std::vector<int> seen(pl.count, 0);
  bool ok = true;
  size_t last_out = 0;
  for (size_t n = 0; n < pl.steps; n++)
    for (size_t q = 0; q < pl.nprog; q++) {
      const int64_t ix = pl.index(n, q);
      if (ix < 0) continue;
      last_out = n;
      if ((uint64_t)ix >= pl.count) { ok = false; continue; }
      seen[ix]++;
      if (pl.arg0[q] + n * pl.arg_step != want[ix]) ok = false;
    }
  for (int c : seen) ok &= c == 1;
  ok &= last_out + 1 == pl.steps;
  if (!ok) printf("  %s plan: count %zu, nprog %zu, steps %zu\n", what, pl.count, pl.nprog, pl.steps);
  CHECK(ok);
}

static void test_plans() {
  const unsigned long d1s[] = {12, 30, 60, 210, 330, 2310, 4620, 9240, 30030};
  const unsigned long d2s[] = {1, 5, 7, 11, 13, 17, 19, 23};
  const char *i0s[] = {"0", "1", "-7", "-1000", "123457", "99999999999999999999999999999999", "18446744073709551615"};
  unsigned seed = 1;
  auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return seed >> 8; };
  for (int it = 0; it < 60; it++) {
    Stage2Params p;
    p.d1 = d1s[rnd() % 9];
    p.d2 = d2s[rnd() % 8];
    p.dF = 1 + rnd() % 300;
    p.k = 1 + rnd() % 5;
    p.S = (rnd() % 2) ? -3 : 2;
    mpz_set_str(p.i0.get(), i0s[rnd() % 7], 10);
    const size_t target = 1 + rnd() % 200;
    /* F: j == 1 (mod 6), gcd(j, d1) = 1 */
    std::vector<uint64_t> wf;
    for (uint64_t j = 1; wf.size() < p.dF; j += 6)
      if (gcd64(j, p.d1) == 1) wf.push_back(j);
    check_plan(plan_roots_F(p, target), wf, "F");
    /* G: offsets t >= 0 with gcd(i0 + t, d2) = 1 */
    const uint64_t r0 = mpz_fdiv_ui(p.i0.get(), p.d2);
    std::vector<uint64_t> wg;
    for (uint64_t t = 0; wg.size() < p.k * p.dF; t++)
      if (p.d2 == 1 || (r0 + t) % p.d2 != 0) wg.push_back(t);
    ProgressionPlan g = plan_roots_G(p, target);
    check_plan(g, wg, "G");
    /* the first table entry of each progression is f((i0 + arg0) d1) */
    Z x, fx;
    for (size_t q = 0; q < g.nprog; q += 1 + g.nprog / 4) {
      mpz_add_ui(x.get(), p.i0.get(), g.arg0[q]);
      mpz_mul_ui(x.get(), x.get(), p.d1);
      if (p.S < 0) dickson(fx.get(), x.get(), 3, -1);
      else mpz_pow_ui(fx.get(), x.get(), 2);
      CHECK(mpz_cmp(fx.get(), g.coeffs[q * (g.S + 1)].get()) == 0);
    }
  }
}

int main() {
  test_expr();
  test_arith();
  test_resume();
  test_curves();
  test_plans();
  printf("%s (%d failures)\n", failures ? "FAIL" : "OK", failures);
  return failures != 0;
}
