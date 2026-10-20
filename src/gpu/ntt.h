/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Batched number-theoretic transforms over the RNS primes of an RnsContext.
   X holds rows polynomials of length T = 2^logT per prime (RNS layout).
   The forward transform is a decimation-in-frequency NTT whose output is in a
   fixed permuted order; ntt_inverse undoes it exactly but without the 1/T
   factor (and is a decimation-in-time NTT). Cyclic convolution of length T:
   forward both, multiply pointwise, inverse, then divide by T (RnsContext::to_limbs
   does the division). 3 <= logT <= ctx.logTmax(). */
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "gpu/rns.h"

void ntt_forward(const RnsContext &ctx, uint32_t *X, size_t rows, int logT, cudaStream_t s = 0);
void ntt_inverse(const RnsContext &ctx, uint32_t *X, size_t rows, int logT, cudaStream_t s = 0);

/* The pointwise products below followed by ntt_inverse, in one pass less:
   the first pass of the inverse transform forms the products as it loads.
   Z := inverse(X .* Y), rows of length 2^logT; Z may alias X or Y. */
void ntt_inverse_mul(const RnsContext &ctx, const uint32_t *X, const uint32_t *Y, uint32_t *Z, size_t rows, int logT,
                     cudaStream_t s = 0);
/* Y[j] := inverse(X[2j] .* X[2j+1] (- 1 with sub_one)), j < count; Y must not alias X. */
void ntt_inverse_adjacent(const RnsContext &ctx, const uint32_t *X, uint32_t *Y, size_t count, int logT, bool sub_one,
                          cudaStream_t s = 0);
/* Z[2j] := inverse(B[j] .* C[2j+1]), Z[2j+1] := inverse(B[j] .* C[2j]); Z must not alias B or C. */
void ntt_inverse_swap(const RnsContext &ctx, const uint32_t *B, const uint32_t *C, uint32_t *Z, size_t count, int logT,
                      cudaStream_t s = 0);

/* Z := X .* Y * 2^-32 (Montgomery) per prime; Z may alias X or Y. n = rows*T. */
void ntt_pointwise_mul(const RnsContext &ctx, const uint32_t *X, const uint32_t *Y, uint32_t *Z, size_t rows_T,
                       cudaStream_t s = 0);

/* Y[j] = X[2j] .* X[2j+1] (Montgomery), rows of length T, j < count.
   With sub_one, the transform of the constant polynomial 1 is subtracted from
   the product (used to cancel the wrapped x^T term of a monic product). */
void ntt_pointwise_adjacent(const RnsContext &ctx, const uint32_t *X, uint32_t *Y, size_t count, size_t T,
                            bool sub_one = false, cudaStream_t s = 0);
/* Z[2j] = B[j] .* C[2j+1], Z[2j+1] = B[j] .* C[2j] (Montgomery), j < count */
void ntt_pointwise_swap(const RnsContext &ctx, const uint32_t *B, const uint32_t *C, uint32_t *Z, size_t count,
                        size_t T, cudaStream_t s = 0);
