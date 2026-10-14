/* Copyright 2026 Erik Branger.
   Ported from GMP-ECM's bestd.c, ecm.c (set_stage_2_params, choose_S) and
   stage2.c (memory_use): Copyright 2001-2012 Paul Zimmermann, Alexander Kruppa
   et al., GNU LGPL version 3 or later.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#pragma once
#include <string>

#include "host/mpz_class_lite.h"

/* Stage 2 parameters exactly as GMP-ECM computes them.
   Stage 2 covers p = f(i*d1) +- f(j*d2) with i0 <= i, gcd(i, d2) = 1,
   gcd(j, d1) = 1, j == 1 (mod 6), 1 <= j <= d1 - 5; f(x) = x^S or Dickson(|S|). */
struct Stage2Params {
  double B1 = 0;
  Z B2min;            // lower bound actually used (default B1)
  Z B2;               // effective upper bound B2' (>= requested B2)
  unsigned long d1 = 0, d2 = 0, dF = 0, k = 0;
  Z i0;
  int S = 1;          // > 0: x^S, < 0: Dickson(-S)
  bool po2 = false;   // dF rounded up to a power of 2 (always true with NTT)
  bool use_ntt = false;
};

struct Stage2Request {
  double B1 = 0;                // stage 1 bound (used for default B2 and B2min)
  Z B2;                         // -1: default
  Z B2min;                      // -1: default (= B1)
  unsigned long k = 0;          // 0: automatic (GMP-ECM's ECM_DEFAULT_K)
  int S = 0;                    // 0: automatic (GMP-ECM's ECM_DEFAULT_S)
  double maxmem = 0;            // bytes; 0: no limit (GMP-ECM's CPU memory model)
  int use_ntt = 1;              // 1: GMP-ECM default rule, 0: never, 2: always
  Stage2Request() { mpz_set_si(B2.get(), -1); mpz_set_si(B2min.get(), -1); }
};

/* Computes the parameters GMP-ECM would use for ECM stage 2 on modulus n.
   Returns false (with a message in *err) where GMP-ECM reports an error. */
bool compute_stage2_params(Stage2Params &out, const Stage2Request &req, const mpz_t n, std::string *err);

/* GMP-ECM's default choice of the Brent-Suyama degree for B2 - B2min. */
int choose_S(const mpz_t B2len);

unsigned long eulerphi(unsigned long n);
unsigned int ceil_log2(unsigned long n);
