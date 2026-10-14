/* Copyright 2026 Erik Branger.
   Ported from GMP-ECM's parametrizations.c (Copyright 2012 Cyril Bouvier) and
   ecm.c (Copyright 2001-2012 Paul Zimmermann, Alexander Kruppa et al.),
   GNU LGPL version 3 or later. The arithmetic is the same, done with plain
   mpz_t residues mod n instead of GMP-ECM's mpres_t.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "host/curve.h"

#include "host/mpz_class_lite.h"

namespace {

/* r = a*b mod n etc., all residues in [0, n) */
struct Mod {
  const mpz_srcptr n;
  void mul(mpz_ptr r, mpz_srcptr a, mpz_srcptr b) const { mpz_mul(r, a, b); mpz_mod(r, r, n); }
  void sqr(mpz_ptr r, mpz_srcptr a) const { mpz_mul(r, a, a); mpz_mod(r, r, n); }
  void mul_ui(mpz_ptr r, mpz_srcptr a, unsigned long b) const { mpz_mul_ui(r, a, b); mpz_mod(r, r, n); }
  void add(mpz_ptr r, mpz_srcptr a, mpz_srcptr b) const { mpz_add(r, a, b); mpz_mod(r, r, n); }
  void sub(mpz_ptr r, mpz_srcptr a, mpz_srcptr b) const { mpz_sub(r, a, b); mpz_mod(r, r, n); }
  void add_ui(mpz_ptr r, mpz_srcptr a, unsigned long b) const { mpz_add_ui(r, a, b); mpz_mod(r, r, n); }
  void sub_ui(mpz_ptr r, mpz_srcptr a, unsigned long b) const { mpz_sub_ui(r, a, b); mpz_mod(r, r, n); }
  void neg(mpz_ptr r, mpz_srcptr a) const { mpz_neg(r, a); mpz_mod(r, r, n); }
  void set_si(mpz_ptr r, long a) const { mpz_set_si(r, a); mpz_mod(r, r, n); }
  /* returns false (and f = gcd) if a is not invertible */
  bool invert(mpz_ptr r, mpz_srcptr a, mpz_ptr f) const {
    if (mpz_invert(r, a, n)) return true;
    mpz_gcd(f, a, n);
    return false;
  }
};

/* --- parametrization 2: multiples of (-3:3:1) on y^2 = x^3 + 36 (Jacobian) --- */
void dbl_param(const Mod &M, Z &x, Z &y, Z &z, Z &t, Z &u, Z &v) {
  M.mul(z, y, z);  M.mul_ui(z, z, 2);   /* Z3 = 2*Y1*Z1 */
  M.sqr(u, x);                           /* A = X1^2 */
  M.sqr(t, y);                           /* B = Y1^2 */
  M.sqr(y, t);                           /* C = B^2 */
  M.add(v, x, t); M.sqr(v, v); M.sub(v, v, u); M.sub(v, v, y);
  M.mul_ui(v, v, 2);                     /* D = 2*((X1+B)^2-A-C) */
  M.mul_ui(u, u, 3);                     /* E = 3*A */
  M.sqr(t, u);                           /* F = E^2 */
  M.mul_ui(x, v, 2); M.sub(x, t, x);     /* X3 = F-2*D */
  M.sub(v, v, x);
  M.mul_ui(y, y, 8);
  M.mul(t, u, v);
  M.sub(y, t, y);                        /* Y3 = E*(D-X3)-8*C */
}

/* Add sgn*P = (-3 : sgn*3 : 1) to Q = (x:y:z) */
void add_param(const Mod &M, Z &x, Z &y, Z &z, int sgn, Z &t, Z &u, Z &v, Z &w) {
  M.sqr(t, z);                           /* Z1Z1 */
  M.mul_ui(u, t, 3); M.neg(u, u);        /* U2 = -3*Z1Z1 */
  M.mul(v, z, t); M.mul_ui(v, v, 3);     /* S2 = 3*Z1*Z1Z1 */
  if (sgn == -1) M.neg(v, v);
  M.sub(u, u, x);                        /* H = U2-X1 */
  M.sqr(w, u);                           /* HH */
  M.add(z, z, u); M.sqr(z, z); M.sub(z, z, t); M.sub(z, z, w);  /* Z3 */
  M.mul_ui(t, w, 4);                     /* I = 4*HH */
  M.mul(u, u, t);                        /* J = H*I */
  M.sub(v, v, y); M.mul_ui(v, v, 2);     /* r = 2*(S2-Y1) */
  M.mul(t, x, t);                        /* V = X1*I */
  M.sqr(x, v); M.mul_ui(w, t, 2); M.sub(x, x, u); M.sub(x, x, w);  /* X3 = r^2-J-2*V */
  M.sub(w, t, x);
  M.mul(y, y, u); M.mul_ui(y, y, 2);
  M.mul(w, v, w);
  M.sub(y, w, y);                        /* Y3 = r*(V-X3)-2*Y1*J */
}

void addchain_param(const Mod &M, Z &x, Z &y, Z &z, Z &s, Z &t, Z &u, Z &v, Z &w) {
  if (mpz_cmp_ui(s, 1) == 0) {
    M.set_si(x, -3); M.set_si(y, 3); M.set_si(z, 1);
  } else if (mpz_cmp_ui(s, 3) == 0) {
    mpz_sub_ui(s, s, 1);
    addchain_param(M, x, y, z, s, t, u, v, w);
    add_param(M, x, y, z, +1, t, u, v, w);
  } else if (mpz_divisible_2exp_p(s, 1)) {
    mpz_tdiv_q_2exp(s, s, 1);
    addchain_param(M, x, y, z, s, t, u, v, w);
    dbl_param(M, x, y, z, t, u, v);
  } else if (mpz_congruent_ui_p(s, 1, 4)) {
    mpz_sub_ui(s, s, 1);
    addchain_param(M, x, y, z, s, t, u, v, w);
    add_param(M, x, y, z, +1, t, u, v, w);
  } else {
    mpz_add_ui(s, s, 1);
    addchain_param(M, x, y, z, s, t, u, v, w);
    add_param(M, x, y, z, -1, t, u, v, w);
  }
}

int param0(const Mod &M, mpz_t f, mpz_t A, mpz_t x, const mpz_t sigma) {
  Z t, u, v, b, z;
  mpz_mod(t, sigma, M.n);
  if (mpz_cmp_ui(t, 5) == 0 || mpz_cmp_ui(t, 3) == 0 || mpz_cmp_ui(t, 1) == 0 || mpz_sgn(t) == 0)
    return CURVE_ERROR;
  mpz_mod(u, sigma, M.n);
  M.mul_ui(v, u, 4);                 /* v = 4*sigma */
  M.sqr(t, u); M.sub_ui(u, t, 5);    /* u = sigma^2-5 */
  M.sqr(t, u); M.mul(x, t, u);       /* x = u^3 */
  M.sqr(t, v); M.mul(z, t, v);       /* z = v^3 */
  M.mul(t, x, v); M.mul_ui(b, t, 4); /* b = 4*x*v */
  M.mul_ui(t, u, 3);
  M.sub(u, v, u);                    /* u' = v-u */
  M.add(v, t, v);                    /* v' = 3*u+v */
  M.sqr(t, u); M.mul(u, t, u);       /* u'' = (v-u)^3 */
  M.mul(A, u, v);                    /* A = (v-u)^3 * (3*u+v) */
  M.mul(v, b, z);
  if (!M.invert(u, v, f))            /* u = (b*z)^-1 */
    return mpz_cmp(f, M.n) == 0 ? CURVE_ERROR : CURVE_FACTOR_FOUND;
  M.mul(v, u, b); M.mul(x, x, v);    /* x = x/z */
  M.mul(v, u, z); M.mul(t, A, v); M.sub_ui(A, t, 2);  /* A = A/b - 2 */
  return CURVE_OK;
}

int param1(const Mod &M, mpz_t A, mpz_t x0, const mpz_t sigma) {
  Z t;
  mpz_ui_pow_ui(t, 2, 64);
  mpz_invert(t, t, M.n);
  mpz_mul(t, t, sigma); mpz_mul(t, t, sigma); mpz_mod(t, t, M.n);  /* d = sigma^2/2^64 */
  if (mpz_sgn(t) == 0 || mpz_cmp_ui(t, 1) == 0) return CURVE_ERROR;
  mpz_mul_2exp(t, t, 2); mpz_sub_ui(t, t, 2);                       /* 4d-2 */
  mpz_mod(A, t, M.n);
  mpz_set_ui(x0, 2);
  return CURVE_OK;
}

int param2(const Mod &M, mpz_t f, mpz_t A, mpz_t x0, const mpz_t sigma) {
  Z t, u, v, w, x, y, z, k;
  mpz_set(k, sigma);
  if (mpz_cmp_ui(k, 2) < 0) return CURVE_ERROR;
  addchain_param(M, x, y, z, k, t, u, v, w);  /* (x:y:z) = k*(-3:3:1) */
  if (!M.invert(u, z, f)) return CURVE_FACTOR_FOUND;
  M.sqr(v, u); M.mul(u, v, u);
  M.mul(x, x, v); M.mul(y, y, u);             /* affine */
  M.sub_ui(t, y, 3); M.mul_ui(t, t, 2);
  if (!M.invert(u, t, f)) return CURVE_FACTOR_FOUND;
  M.mul_ui(w, x, 3); M.add(w, w, y); M.add_ui(w, w, 6);
  M.mul(x, w, u);                             /* x3 = (3x+y+6)/(2(y-3)) */
  /* A = -(3*x3^4+6*x3^2-1)/(4*x3^3) */
  M.sqr(u, x); M.mul(v, u, x); M.sqr(w, u);
  M.mul_ui(u, u, 6); M.neg(u, u);
  M.mul_ui(v, v, 4);
  M.mul_ui(w, w, 3); M.neg(w, w);
  if (!M.invert(t, v, f)) return CURVE_FACTOR_FOUND;
  M.add(w, w, u); M.add_ui(w, w, 1);
  M.mul(A, w, t);
  mpz_set_ui(x0, 2);
  return CURVE_OK;
}

int param3(const Mod &M, mpz_t A, mpz_t x0, const mpz_t sigma) {
  Z t;
  mpz_ui_pow_ui(t, 2, 32);
  mpz_invert(t, t, M.n);
  mpz_mul(t, sigma, t); mpz_mod(t, t, M.n);   /* d = sigma/2^32 */
  if (mpz_sgn(t) == 0 || mpz_cmp_ui(t, 1) == 0) return CURVE_ERROR;
  mpz_mul_2exp(t, t, 2); mpz_sub_ui(t, t, 2); /* 4d-2 */
  mpz_mod(A, t, M.n);
  mpz_set_ui(x0, 2);
  mpz_mul_2exp(t, t, 1); mpz_add_ui(t, t, 5); /* 8d+1 */
  if (mpz_divisible_p(t, M.n)) return CURVE_ERROR;
  return CURVE_OK;
}

}  // namespace

int curve_from_param(mpz_t f, mpz_t A, mpz_t x0, int param, const mpz_t sigma, const mpz_t n) {
  Mod M{n};
  switch (param) {
    case 0: return param0(M, f, A, x0, sigma);
    case 1: return param1(M, A, x0, sigma);
    case 2: return param2(M, f, A, x0, sigma);
    case 3: return param3(M, A, x0, sigma);
    default: return CURVE_ERROR;
  }
}

int montgomery_to_weierstrass(mpz_t f, mpz_t x, mpz_t y, mpz_t A, const mpz_t n) {
  Mod M{n};
  Z g;
  M.add(g, x, A); M.mul(g, g, x); M.add_ui(g, g, 1); M.mul(g, g, x);  /* g = x^3+A*x^2+x */
  M.mul_ui(y, g, 3); M.mul(y, y, g);                                   /* y = 3g^2 */
  if (!M.invert(y, y, f)) return CURVE_FACTOR_FOUND;                   /* y = 1/(3g^2) */
  M.mul_ui(x, x, 3); M.add(x, x, A); M.mul(x, x, g); M.mul(x, x, y);   /* x = (3x+A)/(3g) */
  M.sqr(A, A); M.sub_ui(A, A, 3); M.neg(A, A); M.mul(A, A, y);         /* a = (3-A^2)/(3g^2) */
  M.mul_ui(g, g, 3); M.mul(y, y, g);                                   /* y = 1/g */
  return CURVE_OK;
}
