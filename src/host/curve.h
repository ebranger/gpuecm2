/* Copyright 2026 Erik Branger.
   Ported from GMP-ECM's parametrizations.c (Copyright 2012 Cyril Bouvier) and
   ecm.c (Copyright 2001-2012 Paul Zimmermann, Alexander Kruppa et al.),
   GNU LGPL version 3 or later.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#pragma once
#include <gmp.h>

/* Return codes, as in GMP-ECM */
enum { CURVE_OK = 0, CURVE_FACTOR_FOUND = 1, CURVE_ERROR = -1 };

/* Montgomery curve B*y^2 = x^3 + A*x^2 + x from GMP-ECM's parametrization
   `param` (0..3) and `sigma`. Sets A and the stage-1 starting point x0.
   On CURVE_FACTOR_FOUND the factor is in f. */
int curve_from_param(mpz_t f, mpz_t A, mpz_t x0, int param, const mpz_t sigma, const mpz_t n);

/* Converts the point (x : 1) on the Montgomery curve with parameter A to the
   short Weierstrass form Y^2 = X^3 + a*X + b used by GMP-ECM's stage 2:
   on return x = X, y = Y (with the input y treated as 1, as GMP-ECM does) and
   A = a. Returns CURVE_FACTOR_FOUND (factor in f) if 3*g^2 is not invertible. */
int montgomery_to_weierstrass(mpz_t f, mpz_t x, mpz_t y, mpz_t A, const mpz_t n);
