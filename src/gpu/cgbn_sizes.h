/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Selection of the CGBN instance (threads per instance TPI, BITS) for a
   modulus of a given size: the smallest BITS >= bits(n). The ladder (steps
   of 256 bits up to 2048, then 512) and the TPI per size were chosen by measuring the Edwards point addition on an
   RTX 4070 Ti SUPER; sizes whose limb count is not a multiple of TPI are
   avoided (CGBN pads them, which is slow).
   With GPUECM2_ONLY_BITS defined only that size is compiled (development). */
#pragma once
#include <cstdio>
#include <cstdlib>

/* the largest modulus (bits) this build supports */
#if defined(GPUECM2_ONLY_BITS)
#define GPUECM2_MAX_BITS GPUECM2_ONLY_BITS
#else
#define GPUECM2_MAX_BITS 4608
#endif

#define GPUECM2_CGBN_TPI_FOR(B) \
  ((B) <= 512 ? 4 : (B) <= 1792 ? 8 : (B) == 3072 ? 8 : 16)

#if defined(GPUECM2_ONLY_BITS)
#define GPUECM2_CGBN_DISPATCH(BITSEXPR, CALL, ...)                                       \
  do {                                                                                  \
    const size_t bits_ = (BITSEXPR);                                                    \
    if (bits_ <= GPUECM2_ONLY_BITS) {                                                    \
      CALL<GPUECM2_CGBN_TPI_FOR(GPUECM2_ONLY_BITS), GPUECM2_ONLY_BITS>(__VA_ARGS__);    \
      break;                                                                            \
    }                                                                                   \
    fprintf(stderr, "modulus of %zu bits: this build only supports <= %d bits\n", bits_, \
            GPUECM2_ONLY_BITS);                                                         \
    exit(3);                                                                            \
  } while (0)
#else
#define GPUECM2_CGBN_CASE(B, CALL, ...) \
  if (bits_ <= (B)) { CALL<GPUECM2_CGBN_TPI_FOR(B), (B)>(__VA_ARGS__); break; }
#define GPUECM2_CGBN_DISPATCH(BITSEXPR, CALL, ...)                                       \
  do {                                                                                  \
    const size_t bits_ = (BITSEXPR);                                                    \
    GPUECM2_CGBN_CASE(512, CALL, __VA_ARGS__)                                           \
    GPUECM2_CGBN_CASE(768, CALL, __VA_ARGS__)                                           \
    GPUECM2_CGBN_CASE(1024, CALL, __VA_ARGS__)                                          \
    GPUECM2_CGBN_CASE(1280, CALL, __VA_ARGS__)                                          \
    GPUECM2_CGBN_CASE(1536, CALL, __VA_ARGS__)                                          \
    GPUECM2_CGBN_CASE(1792, CALL, __VA_ARGS__)                                          \
    GPUECM2_CGBN_CASE(2048, CALL, __VA_ARGS__)                                          \
    GPUECM2_CGBN_CASE(2560, CALL, __VA_ARGS__)                                          \
    GPUECM2_CGBN_CASE(3072, CALL, __VA_ARGS__)                                          \
    GPUECM2_CGBN_CASE(3584, CALL, __VA_ARGS__)                                          \
    GPUECM2_CGBN_CASE(4096, CALL, __VA_ARGS__)                                          \
    GPUECM2_CGBN_CASE(4608, CALL, __VA_ARGS__)                                          \
    fprintf(stderr, "modulus of %zu bits is too large (max 4608)\n", bits_);           \
    exit(3);                                                                            \
  } while (0)
#endif
