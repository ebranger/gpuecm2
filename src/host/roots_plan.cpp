/* Copyright 2026 Erik Branger.
   dickson() and fin_diff_coeff() are ported from GMP-ECM's stage2.c
   (Copyright 2001-2012 Paul Zimmermann, Alexander Kruppa et al., GNU LGPL v3+).
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "host/roots_plan.h"

#include <algorithm>
#include <cstdlib>

namespace {

void sub_si(mpz_t r, const mpz_t x, long a) {
  if (a >= 0) mpz_sub_ui(r, x, (unsigned long)a);
  else mpz_add_ui(r, x, (unsigned long)(-a));
}

uint64_t gcd_u(uint64_t a, uint64_t b) {
  while (b) { uint64_t t = a % b; a = b; b = t; }
  return a;
}

/* rank of key among the sorted wanted keys, or -1 */
template <class K>
int32_t rank_of(const std::vector<K> &keys, K key) {
  auto it = std::lower_bound(keys.begin(), keys.end(), key);
  return (it != keys.end() && *it == key) ? (int32_t)(it - keys.begin()) : -1;
}

}  // namespace

void dickson(mpz_t r, const mpz_t x, unsigned n, int a) {
  unsigned i, b = 0;
  Z t, u;
  while (n > 2 && (n & 1) == 0) { b++; n >>= 1; }
  mpz_set(r, x);
  if (n > 1) {
    mpz_mul(r, x, x);
    sub_si(r, r, a);
    sub_si(r, r, a);  /* r = dickson(x, 2, a) */
    mpz_set(t, x);    /* t = dickson(x, 1, a) */
    for (i = 2; i < n; i++) {
      mpz_mul_si(u, t, a);
      mpz_set(t, r);  /* t = dickson(x, i, a) */
      mpz_mul(r, r, x);
      mpz_sub(r, r, u);  /* r = dickson(x, i+1, a) */
    }
  }
  for (; b > 0; b--) {
    mpz_mul(t, r, r);  /* t = dickson(x, n, a)^2 */
    mpz_ui_pow_ui(u, (unsigned long)std::abs(a), n);
    if ((n & 1) && a < 0) mpz_neg(u, u);
    mpz_mul_2exp(u, u, 1);  /* u = 2 a^n */
    mpz_sub(r, t, u);       /* r = dickson(x, 2n, a) */
    n <<= 1;
  }
}

void fin_diff_coeff(Z *coeffs, const mpz_t s, const mpz_t D, unsigned E, int dickson_a) {
  Z t;
  mpz_set(t, s);
  for (unsigned i = 0; i <= E; i++) {
    if (dickson_a != 0) dickson(coeffs[i], t, E, dickson_a);
    else mpz_pow_ui(coeffs[i], t, E);
    mpz_add(t, t, D);
  }
  for (unsigned k = 1; k <= E; k++)
    for (unsigned i = E; i >= k; i--) mpz_sub(coeffs[i], coeffs[i], coeffs[i - 1]);
}

ProgressionPlan plan_roots_F(const Stage2Params &p, size_t nprog_target) {
  ProgressionPlan pl;
  pl.S = std::abs(p.S);
  pl.dickson_a = p.S < 0 ? -1 : 0;
  pl.count = p.dF;
  if (p.dF == 0) return pl;
  /* wanted keys: j == 1 (mod 6), gcd(j, d1) = 1, first dF of them */
  std::vector<uint64_t> keys;
  keys.reserve(p.dF);
  for (uint64_t j = 1; keys.size() < p.dF; j += 6)
    if (gcd_u(j, p.d1) == 1) keys.push_back(j);
  /* residues mod L = 6*M coprime to M (M: product of small primes dividing d1) */
  uint64_t M = 1, phiM = 1;
  for (uint64_t q : {5, 7, 11, 13})
    if (p.d1 % q == 0 && phiM * (q - 1) <= std::max<size_t>(nprog_target, 1)) { M *= q; phiM *= q - 1; }
  const uint64_t L = 6 * M;
  std::vector<uint64_t> res;
  for (uint64_t r = 1; r < L; r += 6)
    if (gcd_u(r, M) == 1) res.push_back(r);
  const size_t c = std::max<size_t>(1, nprog_target / res.size());
  pl.nprog = res.size() * c;
  const uint64_t step = c * L;
  std::vector<uint64_t> start(pl.nprog);
  for (size_t u = 0; u < c; u++)
    for (size_t ri = 0; ri < res.size(); ri++) start[u * res.size() + ri] = res[ri] + u * L;
  pl.arg0 = start;
  pl.arg_step = step;
  const uint64_t kmax = keys.back();
  pl.steps = 0;
  for (uint64_t s0 : start)
    if (s0 <= kmax) pl.steps = std::max<size_t>(pl.steps, (kmax - s0) / step + 1);
  pl.slot.assign(pl.steps * pl.nprog, -1);
  for (size_t n = 0; n < pl.steps; n++)
    for (size_t q = 0; q < pl.nprog; q++) pl.slot[n * pl.nprog + q] = rank_of(keys, start[q] + n * step);
  /* finite difference tables of f(j*d2) along each progression */
  pl.coeffs.resize(pl.nprog * (pl.S + 1));
  Z t, D;
  mpz_set_ui(D, step);
  mpz_mul_ui(D, D, p.d2);
  for (size_t q = 0; q < pl.nprog; q++) {
    mpz_set_ui(t, start[q]);
    mpz_mul_ui(t, t, p.d2);
    fin_diff_coeff(&pl.coeffs[q * (pl.S + 1)], t, D, pl.S, pl.dickson_a);
  }
  return pl;
}

ProgressionPlan plan_roots_G(const Stage2Params &p, size_t nprog_target) {
  ProgressionPlan pl;
  pl.S = std::abs(p.S);
  pl.dickson_a = p.S < 0 ? -1 : 0;
  pl.count = p.k * p.dF;
  if (pl.count == 0) return pl;
  /* wanted: the first k*dF integers i >= i0 with gcd(i, d2) = 1 (d2 = 1 or a
     prime). Progression (u, r) runs over i = i0 + r + u*d2 + n*c*d2, with
     gcd(i0 + r, d2) = 1, so all its values are wanted and their ranks grow
     by c*phi(d2) per step. i0 may be any size; only offsets from it are
     machine integers. */
  const uint64_t d2 = p.d2, phi2 = d2 > 1 ? d2 - 1 : 1;
  const uint64_t r0 = d2 > 1 ? mpz_fdiv_ui(p.i0.get(), d2) : 0;  // i0 mod d2, in [0, d2)
  auto wanted = [&](uint64_t off) { return d2 == 1 || (r0 + off) % d2 != 0; };
  auto rank = [&](uint64_t off) -> uint64_t {  // #{t < off : i0 + t wanted}
    if (d2 == 1) return off;
    const uint64_t t0 = (d2 - r0) % d2;  // first t with d2 | i0 + t
    return off - (off > t0 ? (off - t0 - 1) / d2 + 1 : 0);
  };
  std::vector<uint64_t> res;  // offsets r in [0, d2) with gcd(i0 + r, d2) = 1
  for (uint64_t r = 0; r < d2; r++)
    if (wanted(r)) res.push_back(r);
  const size_t c = std::max<size_t>(1, nprog_target / res.size());
  pl.nprog = res.size() * c;
  const uint64_t step = c * d2;
  pl.inc = (int64_t)(c * phi2);
  pl.base.resize(pl.nprog);
  std::vector<uint64_t> start(pl.nprog);
  pl.steps = 0;
  for (size_t u = 0; u < c; u++)
    for (size_t ri = 0; ri < res.size(); ri++) {
      const size_t q = u * res.size() + ri;
      start[q] = res[ri] + u * d2;
      pl.base[q] = (int64_t)rank(start[q]);
      if ((uint64_t)pl.base[q] < pl.count)  // outputs while base + n*inc < count
        pl.steps = std::max<size_t>(pl.steps, (pl.count - (uint64_t)pl.base[q] + pl.inc - 1) / pl.inc);
    }
  pl.arg0 = start;
  pl.arg_step = step;
  pl.coeffs.resize(pl.nprog * (pl.S + 1));
  Z t, D;
  mpz_set_ui(D, step);
  mpz_mul_ui(D, D, p.d1);
  for (size_t q = 0; q < pl.nprog; q++) {
    mpz_add_ui(t, p.i0.get(), start[q]);
    mpz_mul_ui(t, t, p.d1);
    fin_diff_coeff(&pl.coeffs[q * (pl.S + 1)], t, D, pl.S, pl.dickson_a);
  }
  return pl;
}
