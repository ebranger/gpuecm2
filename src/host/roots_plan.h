/* Copyright 2026 Erik Branger.
   dickson() and fin_diff_coeff() are ported from GMP-ECM's stage2.c
   (Copyright 2001-2012 Paul Zimmermann, Alexander Kruppa et al., GNU LGPL v3+).
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Planning of the root computations of ECM stage 2. The roots are
     F: x( f(j*d2) P )  for j == 1 (mod 6), gcd(j, d1) = 1, the first dF such j,
     G: x( f(i*d1) P )  for i >= i0, gcd(i, d2) = 1, the first k*dF such i,
   in increasing order of j resp. i, f = Dickson_{S,-1} or x^S, exactly as in
   GMP-ECM (ecm2.c). They are generated with several interleaved arithmetic
   progressions of arguments, each evaluated with a degree-S table of finite
   differences: the table of a progression with start t and step D holds
   Delta^i f(t) * P (i = 0..S); one step adds entry i+1 to entry i (i < S). */
#pragma once
#include <cstdint>
#include <vector>

#include "host/mpz_class_lite.h"
#include "host/params.h"

struct ProgressionPlan {
  int S = 1;                  // degree |S|
  int dickson_a = 0;          // 0: x^S, -1: Dickson
  size_t nprog = 0;           // number of progressions
  std::vector<Z> coeffs;      // nprog * (S+1) signed scalars Delta^i f(t_p)
  size_t steps = 0;           // steps to run per progression (outputs per progression)
  /* Output index of (step n, progression p), one of two forms:
     table (F): slot[n*nprog + p], or -1 if that value is not wanted;
     affine (G, slot empty): base[p] + n*inc if that is in [0, count), else
     not wanted. The affine form needs O(nprog) memory whatever count is. */
  std::vector<int32_t> slot;
  std::vector<int64_t> base;
  int64_t inc = 0;
  size_t count = 0;           // number of outputs (dF for F, k*dF for G)
  /* argument of output (n, p): j = arg0[p] + n*arg_step (F), i = i0 + arg0[p]
     + n*arg_step (G); kept for tests */
  std::vector<uint64_t> arg0;
  uint64_t arg_step = 0;
  int64_t index(size_t n, size_t p) const {
    if (!slot.empty()) return slot[n * nprog + p];
    const int64_t v = base[p] + (int64_t)n * inc;
    return v >= 0 && (uint64_t)v < count ? v : -1;
  }
};

/* nprog_target: roughly how many progressions to use (more = more parallelism,
   but more initial scalar multiplications) */
ProgressionPlan plan_roots_F(const Stage2Params &p, size_t nprog_target);
ProgressionPlan plan_roots_G(const Stage2Params &p, size_t nprog_target);

/* r = Dickson_{n,a}(x) (GMP-ECM's dickson()) */
void dickson(mpz_t r, const mpz_t x, unsigned n, int a);
/* coeffs[0..E] = finite difference table of f(s + i D), f = Dickson_{E,a} or x^E */
void fin_diff_coeff(Z *coeffs, const mpz_t s, const mpz_t D, unsigned E, int dickson_a);
