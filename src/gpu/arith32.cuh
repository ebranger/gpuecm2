/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Word-level modular arithmetic for the 31-bit NTT primes (p < 2^31).
   All inline PTX lives in this header; define GPUECM2_NO_PTX (or build for
   HIP) to use the portable C++ versions instead. */
#pragma once
#include <cstdint>

#if defined(__CUDA_ARCH__) && !defined(GPUECM2_NO_PTX) && !defined(__HIP_PLATFORM_AMD__)
#define GPUECM2_PTX 1
#endif

__device__ __forceinline__ uint32_t add_mod(uint32_t a, uint32_t b, uint32_t p) {
  uint32_t s = a + b;  // < 2^32 since a, b < p < 2^31
  return s >= p ? s - p : s;
}
__device__ __forceinline__ uint32_t sub_mod(uint32_t a, uint32_t b, uint32_t p) {
  return a >= b ? a - b : a + p - b;
}

/* b * w mod p with wp = floor(w * 2^32 / p) precomputed (Shoup) */
__device__ __forceinline__ uint32_t mul_shoup(uint32_t b, uint32_t w, uint32_t wp, uint32_t p) {
  uint32_t q = __umulhi(wp, b);
  uint32_t r = w * b - q * p;  // in [0, 2p)
  return r >= p ? r - p : r;
}

/* a * b * 2^-32 mod p (Montgomery), pinv = -p^-1 mod 2^32 */
__device__ __forceinline__ uint32_t mul_mont(uint32_t a, uint32_t b, uint32_t p, uint32_t pinv) {
  uint32_t lo = a * b, hi = __umulhi(a, b), m = lo * pinv;
  uint32_t r = hi + __umulhi(m, p) + (lo != 0);  // < 2p
  return r >= p ? r - p : r;
}

/* (hi:mid:lo) += a * b, a 96-bit accumulator */
__device__ __forceinline__ void mac96(uint32_t &lo, uint32_t &mid, uint32_t &hi, uint32_t a, uint32_t b) {
#ifdef GPUECM2_PTX
  asm("mad.lo.cc.u32 %0, %3, %4, %0;\n\t"
      "madc.hi.cc.u32 %1, %3, %4, %1;\n\t"
      "addc.u32 %2, %2, 0;"
      : "+r"(lo), "+r"(mid), "+r"(hi) : "r"(a), "r"(b));
#else
  uint64_t prod = (uint64_t)a * b;
  uint64_t s = (uint64_t)lo + (uint32_t)prod;
  lo = (uint32_t)s;
  s = (s >> 32) + mid + (prod >> 32);
  mid = (uint32_t)s;
  hi += (uint32_t)(s >> 32);
#endif
}

/* v mod p for a 64-bit v, mu = floor(2^64 / p) */
__device__ __forceinline__ uint32_t reduce64(uint64_t v, uint32_t p, uint64_t mu) {
  uint64_t q = __umul64hi(v, mu);
  uint64_t r = v - q * p;  // < 3p
  while (r >= p) r -= p;
  return (uint32_t)r;
}
