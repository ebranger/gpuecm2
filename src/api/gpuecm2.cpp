/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   The C interface (gpuecm2.h) on top of Stage2Engine. */
#include "api/gpuecm2.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "api/stage2.h"
#include "gpu/cgbn_sizes.h"
#include "host/curve.h"
#include "host/params.h"
#include "host/version.h"

namespace {
constexpr size_t MAX_BITS = GPUECM2_MAX_BITS;

int fail(char *err, size_t errlen, const std::string &msg) {
  if (err && errlen) snprintf(err, errlen, "%s", msg.c_str());
  return GPUECM2_ERROR;
}

void to_params(Stage2Params &sp, const gpuecm2_params_struct *p) {
  sp.dF = p->dF; sp.k = p->k; sp.d1 = p->d1; sp.d2 = p->d2; sp.S = p->S;
  mpz_set(sp.i0.get(), p->i0);
  mpz_set(sp.B2min.get(), p->B2min);
  mpz_set(sp.B2.get(), p->B2);
}
}  // namespace

struct gpuecm2_stage2 {
  Z n;
  Stage2Params sp;
  Stage2Engine *engine = nullptr;
  size_t batch = 0;  // curves per engine call (0: not yet chosen)
};

extern "C" {

void gpuecm2_params_init(gpuecm2_params p) {
  p->dF = p->k = p->d1 = p->d2 = 0;
  p->S = 1;
  mpz_init(p->i0);
  mpz_init(p->B2min);
  mpz_init(p->B2);
}

void gpuecm2_params_clear(gpuecm2_params p) {
  mpz_clear(p->i0);
  mpz_clear(p->B2min);
  mpz_clear(p->B2);
}

int gpuecm2_choose_params(gpuecm2_params p, const mpz_t n, double B1, const mpz_t B2min, const mpz_t B2,
                          unsigned long k, int S, double maxmem, int ntt, char *err, size_t errlen) {
  if (mpz_sgn(n) <= 0 || mpz_even_p(n)) return fail(err, errlen, "n must be odd and positive");
  Stage2Request req;
  req.B1 = B1;
  if (B2 && mpz_sgn(B2) >= 0) mpz_set(req.B2.get(), B2);
  if (B2min && mpz_sgn(B2min) >= 0) mpz_set(req.B2min.get(), B2min);
  req.k = k;
  req.S = S;
  req.maxmem = maxmem;
  req.use_ntt = ntt == 0 ? 0 : ntt == 2 ? 2 : 1;
  Stage2Params sp;
  std::string e;
  if (!stage2_params_for_gpu(sp, req, n, stage2_gpu_budget(), nullptr, &e)) return fail(err, errlen, e);
  p->dF = sp.dF; p->k = sp.k; p->d1 = sp.d1; p->d2 = sp.d2; p->S = sp.S;
  mpz_set(p->i0, sp.i0.get());
  mpz_set(p->B2min, sp.B2min.get());
  mpz_set(p->B2, sp.B2.get());
  return 0;
}

gpuecm2_stage2 *gpuecm2_stage2_new(const mpz_t n, const gpuecm2_params p, char *err, size_t errlen) {
  if (mpz_sgn(n) <= 0 || mpz_even_p(n)) { fail(err, errlen, "n must be odd and positive"); return nullptr; }
  if (mpz_sizeinbase(n, 2) > MAX_BITS) {
    fail(err, errlen, "n has " + std::to_string(mpz_sizeinbase(n, 2)) + " bits; at most " + std::to_string(MAX_BITS) + " are supported");
    return nullptr;
  }
  /* parameters as GMP-ECM makes them (bestD, choose_S): d1 a multiple of 6 with dF >= phi(d1)/2
     roots of F, d2 = 1 or a prime 5 <= d2 < 25 coprime to d1, 1 <= |S| <= 1000 */
  {
    const unsigned long d2 = p->d2;
    const bool d2_ok = d2 == 1 || ((d2 == 5 || d2 == 7 || d2 == 11 || d2 == 13 || d2 == 17 || d2 == 19 ||
                                    d2 == 23) && p->d1 % d2 != 0);
    if (!p->k || !p->d1 || p->d1 % 6 != 0 || !p->dF || p->dF < eulerphi(p->d1) / 2 || !d2_ok || p->S == 0 ||
        p->S > 1000 || p->S < -1000) {
      fail(err, errlen, "invalid stage 2 parameters (not as GMP-ECM chooses them)");
      return nullptr;
    }
  }
  gpuecm2_stage2 *s = new gpuecm2_stage2;
  mpz_set(s->n.get(), n);
  to_params(s->sp, p);
  const size_t budget = stage2_gpu_budget();
  if (budget && stage2_gpu_bytes(n, s->sp, 1) > budget) {
    char buf[200];
    snprintf(buf, sizeof buf, "not enough GPU memory for dF=%lu (%.1f GB usable); use a larger k", p->dF,
             budget / 1e9);
    fail(err, errlen, buf);
    delete s;
    return nullptr;
  }
  s->engine = new Stage2Engine(n, s->sp);
  if (!s->engine->ok()) {
    fail(err, errlen, s->engine->error());
    delete s->engine;
    delete s;
    return nullptr;
  }
  return s;
}

void gpuecm2_stage2_free(gpuecm2_stage2 *s) {
  if (!s) return;
  delete s->engine;
  delete s;
}

size_t gpuecm2_stage2_batch_size(gpuecm2_stage2 *s) {
  if (!s) return 1;
  if (!s->batch) s->batch = std::max<size_t>(1, stage2_auto_batch(s->n.get(), s->sp, stage2_gpu_budget()));
  return s->batch;
}

int gpuecm2_stage2_run(gpuecm2_stage2 *s, size_t count, mpz_t *A, mpz_t *x, int *status, mpz_t *f, mpz_t *value,
                       unsigned flags) {
  if (!s || (count && (!A || !x || !status || !f))) return GPUECM2_ERROR;
  const mpz_srcptr n = s->n.get();
  /* GMP-ECM converts each curve to Weierstrass form first; a non-invertible
     value there is a factor */
  std::vector<MontgomeryCurve> curves;
  std::vector<size_t> idx;
  curves.reserve(count);
  for (size_t i = 0; i < count; i++) {
    status[i] = GPUECM2_NO_FACTOR;
    if (value) mpz_set_ui(value[i], 0);
    Z xa, y, a, fac;
    mpz_mod(xa.get(), x[i], n);
    mpz_mod(a.get(), A[i], n);
    mpz_set_ui(y.get(), 1);
    if (montgomery_to_weierstrass(fac.get(), xa.get(), y.get(), a.get(), n) == CURVE_FACTOR_FOUND) {
      mpz_set(f[i], fac.get());
      status[i] = GPUECM2_FACTOR_FOUND;
      continue;
    }
    /* a singular curve (A^2 = 4 modulo a factor of n) has no Edwards model */
    mpz_mod(a.get(), A[i], n);
    mpz_mul(fac.get(), a.get(), a.get());
    mpz_sub_ui(fac.get(), fac.get(), 4);
    mpz_gcd(fac.get(), fac.get(), n);
    if (mpz_cmp_ui(fac.get(), 1) != 0) {
      status[i] = GPUECM2_ERROR;
      continue;
    }
    MontgomeryCurve C;
    mpz_set(C.n.get(), n);
    mpz_mod(C.A.get(), A[i], n);
    mpz_mod(C.u.get(), x[i], n);
    curves.push_back(std::move(C));
    idx.push_back(i);
  }
  const size_t B = gpuecm2_stage2_batch_size(s);
  for (size_t at = 0; at < curves.size(); at += B) {
    const size_t cnt = std::min(B, curves.size() - at);
    std::vector<Stage2Outcome> out =
        s->engine->run_batch(curves.data() + at, cnt, false, (flags & GPUECM2_SELFCHECK) != 0);
    for (size_t c = 0; c < cnt; c++) {
      const size_t i = idx[at + c];
      const Stage2Outcome &o = out[c];
      if (o.check_failed) status[i] = GPUECM2_CHECK_FAILED;
      else if (o.status == 2) { status[i] = GPUECM2_FACTOR_FOUND; mpz_set(f[i], o.factor.get()); }
      else if (o.status < 0) status[i] = GPUECM2_ERROR;
      if (value && o.have_first) mpz_set(value[i], o.first_value.get());
    }
  }
  int found = 0;
  for (size_t i = 0; i < count; i++) found += status[i] == GPUECM2_FACTOR_FOUND;
  return found;
}

int gpuecm2_curve_from_sigma(mpz_t f, mpz_t A, mpz_t x0, int param, const mpz_t sigma, const mpz_t n) {
  if (param < 0 || param > 3) return GPUECM2_ERROR;
  const int r = curve_from_param(f, A, x0, param, sigma, n);
  return r == CURVE_OK ? 0 : r == CURVE_FACTOR_FOUND ? GPUECM2_FACTOR_FOUND : GPUECM2_ERROR;
}

int gpuecm2_set_device(int device) { return cudaSetDevice(device) == cudaSuccess ? 0 : GPUECM2_ERROR; }

void gpuecm2_cpu_wait(int spin) { stage2_cpu_wait(spin != 0); }

/* gpuecm2_version() is defined in the core library (host/version.cpp); this
   reference makes the linker put it into libgpuecm2 */
const char *(*volatile libgpuecm2_version_ref)(void) = gpuecm2_version;

}  // extern "C"
