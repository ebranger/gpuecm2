/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Element-wise operations on limb arrays of residues mod n (layout as in rns.h:
   limb j of coefficient g at A[j*ld + g]). Inputs must be reduced (< n). */
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "gpu/rns.h"

/* C[g] = A[g] + B[g] mod n  (g < count); C may alias A or B */
void limb_add(const RnsContext &ctx, const uint32_t *A, size_t lda, const uint32_t *B, size_t ldb, uint32_t *C,
              size_t ldc, size_t count, cudaStream_t s = 0);
/* C[g] = A[g] - B[g] mod n */
void limb_sub(const RnsContext &ctx, const uint32_t *A, size_t lda, const uint32_t *B, size_t ldb, uint32_t *C,
              size_t ldc, size_t count, cudaStream_t s = 0);
/* C[g] = -A[g] mod n */
void limb_neg(const RnsContext &ctx, const uint32_t *A, size_t lda, uint32_t *C, size_t ldc, size_t count,
              cudaStream_t s = 0);
/* C[g] = A[rev ? count-1-g : g]  (C must not alias A when rev) */
void limb_copy(const RnsContext &ctx, const uint32_t *A, size_t lda, uint32_t *C, size_t ldc, size_t count, bool rev,
               cudaStream_t s = 0);
/* C[g] = small constant c (c < n) */
void limb_set_ui(const RnsContext &ctx, uint32_t *C, size_t ldc, size_t count, uint32_t c, cudaStream_t s = 0);

/* Row-wise forms of the above for rows rows of cnt coefficients: row r of
   an operand starts at coefficient r*stride (as, bs, cs) of it; with rev,
   each row is reversed. */
void limb_add_rows(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t as, const uint32_t *B, size_t ldb,
                   size_t bs, uint32_t *C, size_t ldc, size_t cs, size_t rows, size_t cnt, cudaStream_t s = 0);
void limb_sub_rows(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t as, const uint32_t *B, size_t ldb,
                   size_t bs, uint32_t *C, size_t ldc, size_t cs, size_t rows, size_t cnt, cudaStream_t s = 0);
void limb_neg_rows(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t as, uint32_t *C, size_t ldc, size_t cs,
                   size_t rows, size_t cnt, cudaStream_t s = 0);
void limb_copy_rows(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t as, uint32_t *C, size_t ldc,
                    size_t cs, size_t rows, size_t cnt, bool rev, cudaStream_t s = 0);
void limb_set_ui_rows(const RnsContext &ctx, uint32_t *C, size_t ldc, size_t cs, size_t rows, size_t cnt, uint32_t c,
                      cudaStream_t s = 0);

/* Row-wise gather into a dense array of `rows` rows of len coefficients
   (Out entry r*len + t, leading dimension ldo). Row r of the source starts at
   coefficient r*row_stride of A; data coefficient d (d < data_len) is placed at
   t = offset + d, or offset + data_len - 1 - d if reverse; a coefficient 1 is
   placed at t = lead_pos if lead_pos >= 0; everything else is 0. */
void limb_gather_rows(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t row_stride, size_t data_len,
                      bool reverse, size_t offset, long lead_pos, size_t rows, size_t len, uint32_t *Out, size_t ldo,
                      cudaStream_t s = 0);

/* Inverse of a dense gather: Dst entry r*dst_stride + dst_off + t := Src entry
   r*cnt + t, for r < rows, t < cnt. */
void limb_scatter_rows(const RnsContext &ctx, const uint32_t *Src, size_t lds, size_t rows, size_t cnt, uint32_t *Dst,
                       size_t ldd, size_t dst_stride, size_t dst_off, cudaStream_t s = 0);

/* limb-array allocation helpers (stream-ordered) */
uint32_t *limb_alloc(const RnsContext &ctx, size_t count, cudaStream_t s = 0);
void limb_free(uint32_t *p, cudaStream_t s = 0);
