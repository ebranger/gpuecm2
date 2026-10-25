/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Stage 2 engine tests that need no GMP-ECM binary:
   1. random curves pass the self-check (and give the same values without it);
   2. injected faults are caught by the self-check;
   3. RNS storage gives the same values as limbs (batches, with the
      self-check, with the tree cache, with the long-product fallback), and
      its self-check catches the injected faults too;
   4. the low levels mod n (ZnLevels: none, the default 4, 6) give the same
      values, with both storages.
   Usage: test_stage2 */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <gmp.h>

#include "api/stage2.h"
#include "gpu/cgbn_sizes.h"
#include "host/params.h"
#include "random_number.h"

static int failures = 0;

static void check(bool ok, const char *what) {
  printf("%s %s\n", ok ? "ok  " : "FAIL", what);
  failures += !ok;
}

static void set_env(const char *name, const char *value) {
#ifdef _WIN32
  _putenv_s(name, value ? value : "");
#else
  if (value) setenv(name, value, 1);
  else unsetenv(name);
#endif
}

/* the same random curves (a batch) with limbs and with RNS storage */
static void compare_rns(const MontgomeryCurve &C0, const Stage2Params &sp, gmp_randstate_t rs, const char *tag) {
  stage2_set_rns_store(false);
  Stage2Engine eng(C0.n.get(), sp);
  stage2_set_rns_store(true);
  Stage2Engine engr(C0.n.get(), sp);
  stage2_set_rns_store_auto();
  char what[240];
  const int bits = (int)mpz_sizeinbase(C0.n.get(), 2);
  if (!eng.ok() || !engr.ok() || !engr.rns().rns_store()) {
    snprintf(what, sizeof what, "%d-bit N%s: engines (%s)", bits, tag, (eng.ok() ? engr : eng).error().c_str());
    check(false, what);
    return;
  }
  std::vector<MontgomeryCurve> Cs(3);
  for (MontgomeryCurve &c : Cs) {
    mpz_set(c.n.get(), C0.n.get());
    mpz_urandomm(c.A.get(), rs, c.n.get());
    mpz_urandomm(c.u.get(), rs, c.n.get());
  }
  const std::vector<Stage2Outcome> a = eng.run_batch(Cs.data(), Cs.size(), false);
  for (int variant = 0; variant < 3; variant++) {  // plain, self-check, tree cache
    engr.set_tree_cache(variant == 2);
    const std::vector<Stage2Outcome> b = engr.run_batch(Cs.data(), Cs.size(), false, variant == 1);
    bool ok = true;
    for (size_t c = 0; c < Cs.size(); c++)
      ok &= a[c].status == b[c].status && !b[c].check_failed && a[c].have_full && b[c].have_full &&
            mpz_cmp(a[c].full_product.get(), b[c].full_product.get()) == 0 &&
            mpz_cmp(a[c].first_value.get(), b[c].first_value.get()) == 0;
    snprintf(what, sizeof what, "%d-bit N, dF=%lu, k=%lu%s: RNS storage%s gives the same products (batch of %zu)",
             bits, sp.dF, sp.k, tag, variant == 1 ? " with self-check" : variant == 2 ? " with tree cache" : "",
             Cs.size());
    check(ok, what);
  }
  engr.set_tree_cache(false);
  for (const char *fault : {"1", "2", "3"}) {
    set_env("GPUECM2_INJECT_FAULT", fault);
    Stage2Outcome o = engr.run(Cs[0], false, true);
    set_env("GPUECM2_INJECT_FAULT", nullptr);
    snprintf(what, sizeof what, "%d-bit N%s: RNS storage, injected fault %s is caught", bits, tag, fault);
    check(o.status == -1 && o.check_failed, what);
  }
}

/* the same random curves (a batch) without and with the low levels mod n */
static void compare_zn(const MontgomeryCurve &C0, const Stage2Params &sp, gmp_randstate_t rs) {
  std::vector<MontgomeryCurve> Cs(3);
  for (MontgomeryCurve &c : Cs) {
    mpz_set(c.n.get(), C0.n.get());
    mpz_urandomm(c.A.get(), rs, c.n.get());
    mpz_urandomm(c.u.get(), rs, c.n.get());
  }
  const int bits = (int)mpz_sizeinbase(C0.n.get(), 2);
  char what[240];
  for (int rns = 0; rns < 2; rns++) {
    stage2_set_rns_store(rns != 0);
    std::vector<Stage2Outcome> ref;
    for (const char *zl : {"0", "4", "6"}) {
      set_env("GPUECM2_ZN_LEVELS", zl);
      Stage2Engine eng(C0.n.get(), sp);
      set_env("GPUECM2_ZN_LEVELS", nullptr);
      const std::vector<Stage2Outcome> o = eng.run_batch(Cs.data(), Cs.size(), false, zl[0] == '4');
      if (ref.empty()) { ref = o; continue; }
      bool ok = eng.ok();
      for (size_t c = 0; ok && c < Cs.size(); c++)
        ok &= o[c].status == ref[c].status && !o[c].check_failed && o[c].have_full &&
              mpz_cmp(o[c].full_product.get(), ref[c].full_product.get()) == 0;
      snprintf(what, sizeof what, "%d-bit N, dF=%lu, k=%lu%s: %s levels mod n give the same products%s", bits,
               sp.dF, sp.k, rns ? " [rns]" : "", zl, zl[0] == '4' ? " (with self-check)" : "");
      check(ok, what);
    }
  }
  stage2_set_rns_store_auto();
}

static void run_case(int bits, double B1, int S) {
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 7 + bits);
  MontgomeryCurve C;
  {  // RSA-like modulus
    Z p, q;
    random_hard_number(p.get(), rs, bits / 2);
    random_hard_number(q.get(), rs, bits - bits / 2);
    mpz_mul(C.n.get(), p.get(), q.get());
  }
  Stage2Request req;
  req.B1 = B1;
  req.S = S;
  Stage2Params sp;
  std::string err;
  if (!compute_stage2_params(sp, req, C.n.get(), &err)) { check(false, err.c_str()); return; }
  Stage2Engine eng(C.n.get(), sp);
  if (!eng.ok()) { check(false, eng.error().c_str()); return; }
  char what[200];
  for (int curve = 0; curve < 2; curve++) {
    mpz_urandomm(C.A.get(), rs, C.n.get());
    mpz_urandomm(C.u.get(), rs, C.n.get());
    Stage2Outcome plain = eng.run(C, false, false), checked = eng.run(C, false, true);
    snprintf(what, sizeof what, "%d-bit N, B1=%.0f, dF=%lu, k=%lu, S=%d: self-check passes, same result", bits, B1,
             sp.dF, sp.k, sp.S);
    check(checked.status == plain.status && !checked.check_failed && plain.have_first && checked.have_first &&
              mpz_cmp(plain.first_value.get(), checked.first_value.get()) == 0,
          what);
  }
  for (const char *fault : {"1", "2", "3"}) {
    set_env("GPUECM2_INJECT_FAULT", fault);
    Stage2Outcome o = eng.run(C, false, true);
    set_env("GPUECM2_INJECT_FAULT", nullptr);
    snprintf(what, sizeof what, "%d-bit N: injected fault %s is caught", bits, fault);
    check(o.status == -1 && o.check_failed, what);
  }
  compare_rns(C, sp, rs, "");
  compare_zn(C, sp, rs);
  gmp_randclear(rs);
}

/* the long-product fallback (NTTs of at most 2^12 points) with both storages */
static void fallback_case(int bits, double B1) {
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 11 + bits);
  MontgomeryCurve C;
  Z p, q;
  random_hard_number(p.get(), rs, bits / 2);
  random_hard_number(q.get(), rs, bits - bits / 2);
  mpz_mul(C.n.get(), p.get(), q.get());
  Stage2Request req;
  req.B1 = B1;
  Stage2Params sp;
  std::string err;
  if (!compute_stage2_params(sp, req, C.n.get(), &err)) { check(false, err.c_str()); return; }
  set_env("GPUECM2_MAX_LOGT", "12");
  compare_rns(C, sp, rs, " [NTTs <= 2^12]");
  set_env("GPUECM2_MAX_LOGT", nullptr);
  gmp_randclear(rs);
}

int main() {
  run_case(400, 2e4, 0);
  run_case(1000, 1e5, -12);
  if (GPUECM2_MAX_BITS >= 2100) run_case(2100, 5e4, 0);
  if (GPUECM2_MAX_BITS >= 12000) run_case(12000, 2e3, -6);
  fallback_case(1000, 3e6);
  printf("%s (%d failures)\n", failures ? "FAIL" : "OK", failures);
  return failures != 0;
}
