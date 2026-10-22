/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Stage 2 engine tests that need no GMP-ECM binary:
   1. random curves pass the self-check (and give the same values without it);
   2. injected faults are caught by the self-check.
   Usage: test_stage2 */
#include <cstdio>
#include <cstdlib>
#include <string>

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
  gmp_randclear(rs);
}

int main() {
  run_case(400, 2e4, 0);
  run_case(1000, 1e5, -12);
  if (GPUECM2_MAX_BITS >= 2100) run_case(2100, 5e4, 0);
  if (GPUECM2_MAX_BITS >= 12000) run_case(12000, 2e3, -6);
  printf("%s (%d failures)\n", failures ? "FAIL" : "OK", failures);
  return failures != 0;
}
