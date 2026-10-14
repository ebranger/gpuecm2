/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Batched polynomial multiplication over Z/nZ. */
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "gpu/rns.h"

/* C_r = A_r * B_r for r < rows. A_r has la coefficients (coefficient i of row r
   is limb-array entry r*la + i, leading dimension lda), likewise B_r (lb) and
   C_r (la + lb - 1 coefficients, entry r*(la+lb-1) + i).
   Requires min(la, lb) <= the max_operand_len the context was made for.
   Products longer than the largest NTT are split into pieces. */
void poly_mul(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t la, const uint32_t *B, size_t ldb,
              size_t lb, uint32_t *C, size_t ldc, size_t rows, cudaStream_t s = 0);
