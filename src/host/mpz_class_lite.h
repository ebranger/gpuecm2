/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#pragma once
#include <gmp.h>
#include <utility>

/* Minimal RAII wrapper around mpz_t (we avoid a dependency on gmpxx). */
class Z {
 public:
  Z() { mpz_init(v_); }
  explicit Z(unsigned long x) { mpz_init_set_ui(v_, x); }
  Z(const Z &o) { mpz_init_set(v_, o.v_); }
  Z(Z &&o) noexcept { mpz_init(v_); mpz_swap(v_, o.v_); }
  Z &operator=(const Z &o) { if (this != &o) mpz_set(v_, o.v_); return *this; }
  Z &operator=(Z &&o) noexcept { mpz_swap(v_, o.v_); return *this; }
  ~Z() { mpz_clear(v_); }
  mpz_ptr get() { return v_; }
  mpz_srcptr get() const { return v_; }
  operator mpz_ptr() { return v_; }
  operator mpz_srcptr() const { return v_; }
  /* lets GMP's function-like macros (mpz_sgn, mpz_cmp_ui, ...) accept a Z */
  __mpz_struct *operator->() { return v_; }
  const __mpz_struct *operator->() const { return v_; }

 private:
  mpz_t v_;
};
