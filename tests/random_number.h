/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#pragma once
#include <gmp.h>

/* A random number of `bits` bits with no small factors: a prime up to 4608
   bits, above that a product of primes of about 1024 bits (large primes take
   minutes to find). */
static inline void random_hard_number(mpz_t r, gmp_randstate_t rs, int bits) {
  if (bits <= 4608) {
    mpz_urandomb(r, rs, bits); mpz_setbit(r, bits - 1); mpz_nextprime(r, r);
    return;
  }
  mpz_t f, lo;
  mpz_inits(f, lo, NULL);
  do {
    mpz_set_ui(r, 1);
    for (int k = 0; k < (bits - 1) / 1024 - 1; k++) {
      mpz_urandomb(f, rs, 1024); mpz_setbit(f, 1023); mpz_nextprime(f, f);
      mpz_mul(r, r, f);
    }
    /* the last prime in [lo, 2 lo), lo = 2^(bits-1) / r */
    mpz_set_ui(lo, 0); mpz_setbit(lo, bits - 1);
    mpz_cdiv_q(lo, lo, r);
    mpz_urandomm(f, rs, lo); mpz_add(f, f, lo); mpz_nextprime(f, f);
    mpz_mul(r, r, f);
  } while (mpz_sizeinbase(r, 2) != (size_t)bits);
  mpz_clears(f, lo, NULL);
}
