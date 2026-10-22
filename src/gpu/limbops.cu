/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "gpu/limbops.h"

#include "gpu/arith32.cuh"
#include "gpu/cuda_check.h"

namespace {

unsigned blocks(size_t n) { size_t b = (n + 127) / 128; return (unsigned)(b < 65535 ? b : 65535); }

/* op: 0 add, 1 sub, 2 neg (B unused) */
/* Entry g of a rows x cnt operation: row g / cnt, position g % cnt, at
   coefficient row*stride + position of an operand (stride == cnt: dense). */
__device__ __forceinline__ size_t at(size_t g, size_t cnt, size_t stride) {
  if (stride == cnt) return g;
  const size_t r = g / cnt;
  return r * stride + (g - r * cnt);
}

template <int OP>
__global__ void k_addsub(const uint32_t *__restrict__ A, size_t lda, size_t as, const uint32_t *__restrict__ B,
                         size_t ldb, size_t bs, uint32_t *C, size_t ldc, size_t cs, size_t cnt, size_t count,
                         RnsTables t) {
  const int nl = t.nlimbs;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < count; e += (size_t)gridDim.x * blockDim.x) {
    const size_t ia = at(e, cnt, as), ib = OP == 2 ? 0 : at(e, cnt, bs), ic = at(e, cnt, cs);
    if (OP == 0) {
      /* s = a + b; if s >= n: s -= n.  Compute s and s - n, pick one. */
      uint32_t carry = 0;
      int64_t br = 0;
      bool ge;
      /* first pass: carry of a+b and borrow of (a+b)-n */
      for (int j = 0; j < nl; j++) {
        uint64_t s = (uint64_t)A[j * lda + ia] + B[j * ldb + ib] + carry;
        carry = (uint32_t)(s >> 32);
        int64_t d = (int64_t)(uint32_t)s - t.n[j] + br;
        br = d >> 32;
      }
      ge = carry || br == 0;
      carry = 0;
      br = 0;
      for (int j = 0; j < nl; j++) {
        uint64_t s = (uint64_t)A[j * lda + ia] + B[j * ldb + ib] + carry;
        carry = (uint32_t)(s >> 32);
        int64_t d = (int64_t)(uint32_t)s - (ge ? t.n[j] : 0) + br;
        br = d >> 32;
        C[j * ldc + ic] = (uint32_t)d;
      }
    } else {
      /* d = a - b (a = 0 for neg); if negative: d += n */
      int64_t br = 0;
      for (int j = 0; j < nl; j++) {
        int64_t d = (int64_t)(OP == 1 ? A[j * lda + ia] : 0) - (OP == 1 ? B[j * ldb + ib] : A[j * lda + ia]) + br;
        br = d >> 32;
      }
      const bool neg = br < 0;
      br = 0;
      uint32_t carry = 0;
      for (int j = 0; j < nl; j++) {
        int64_t d = (int64_t)(OP == 1 ? A[j * lda + ia] : 0) - (OP == 1 ? B[j * ldb + ib] : A[j * lda + ia]) + br;
        br = d >> 32;
        uint64_t s = (uint64_t)(uint32_t)d + (neg ? t.n[j] : 0) + carry;
        carry = (uint32_t)(s >> 32);
        C[j * ldc + ic] = (uint32_t)s;
      }
    }
  }
}

/* the same with RNS storage: residues mod b_j = p[PA + j] of integers < 64 n,
   a difference is x + (16 n - y) (exact for y <= 16 n) */
template <int OP>
__global__ void k_addsub_rns(const uint32_t *__restrict__ A, size_t lda, size_t as, const uint32_t *__restrict__ B,
                             size_t ldb, size_t bs, uint32_t *C, size_t ldc, size_t cs, size_t cnt, size_t count,
                             RnsTables t) {
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < count; e += (size_t)gridDim.x * blockDim.x) {
    const size_t ia = at(e, cnt, as), ib = OP == 2 ? 0 : at(e, cnt, bs), ic = at(e, cnt, cs);
    for (int j = 0; j < t.PB; j++) {
      const uint32_t p = t.p[t.PA + j], a = A[j * lda + ia];
      uint32_t r;
      if (OP == 0) r = add_mod(a, B[j * ldb + ib], p);
      else if (OP == 1) r = add_mod(a, sub_mod(t.n16B[j], B[j * ldb + ib], p), p);
      else r = sub_mod(t.n16B[j], a, p);
      C[j * ldc + ic] = r;
    }
  }
}

/* word j of the constant c: c (j = 0) and 0, or c * one mod b_j (RNS storage) */
__device__ __forceinline__ uint32_t const_word(uint32_t c, int j, const RnsTables &t) {
  if (!t.PB) return j == 0 ? c : 0;
  const int q = t.PA + j;
  return (uint32_t)reduce64((uint64_t)c * t.one[q], t.p[q], t.mu[q]);
}

__global__ void k_copy(const uint32_t *__restrict__ A, size_t lda, size_t as, uint32_t *__restrict__ C, size_t ldc,
                       size_t cs, size_t cnt, size_t count, bool rev, int nl) {
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < count; e += (size_t)gridDim.x * blockDim.x) {
    const size_t r = e / cnt, i = e - r * cnt;
    const size_t src = r * as + (rev ? cnt - 1 - i : i), dst = r * cs + i;
    for (int j = 0; j < nl; j++) C[j * ldc + dst] = A[j * lda + src];
  }
}

__global__ void k_set_ui(uint32_t *C, size_t ldc, size_t cs, size_t cnt, size_t count, uint32_t c, RnsTables t) {
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < count; e += (size_t)gridDim.x * blockDim.x) {
    const size_t dst = at(e, cnt, cs);
    for (int j = 0; j < t.cw; j++) C[j * ldc + dst] = const_word(c, j, t);
  }
}

__global__ void k_gather(const uint32_t *__restrict__ A, size_t lda, size_t row_stride, size_t data_len, bool rev,
                         size_t offset, long lead_pos, size_t rows, size_t len, uint32_t *__restrict__ Out, size_t ldo,
                         RnsTables tb) {
  const int nl = tb.cw;
  const size_t total = rows * len;
  for (size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x; g < total; g += (size_t)gridDim.x * blockDim.x) {
    const size_t r = g / len, t = g % len;
    if ((long)t == lead_pos) {
      for (int j = 0; j < nl; j++) Out[j * ldo + g] = const_word(1, j, tb);
    } else if (t >= offset && t < offset + data_len) {
      const size_t d = rev ? offset + data_len - 1 - t : t - offset;
      const size_t src = r * row_stride + d;
      for (int j = 0; j < nl; j++) Out[j * ldo + g] = A[j * lda + src];
    } else {
      for (int j = 0; j < nl; j++) Out[j * ldo + g] = 0;
    }
  }
}

__global__ void k_scatter(const uint32_t *__restrict__ Src, size_t lds, size_t rows, size_t cnt,
                          uint32_t *__restrict__ Dst, size_t ldd, size_t dst_stride, size_t dst_off, int nl) {
  const size_t total = rows * cnt;
  for (size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x; g < total; g += (size_t)gridDim.x * blockDim.x) {
    const size_t d = (g / cnt) * dst_stride + dst_off + g % cnt;
    for (int j = 0; j < nl; j++) Dst[j * ldd + d] = Src[j * lds + g];
  }
}

}  // namespace

void limb_scatter_rows(const RnsContext &ctx, const uint32_t *Src, size_t lds, size_t rows, size_t cnt, uint32_t *Dst,
                       size_t ldd, size_t dst_stride, size_t dst_off, cudaStream_t s) {
  if (!rows || !cnt) return;
  k_scatter<<<blocks(rows * cnt), 128, 0, s>>>(Src, lds, rows, cnt, Dst, ldd, dst_stride, dst_off, ctx.coef_words());
  CUDA_CHECK(cudaGetLastError());
}

void limb_gather_rows(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t row_stride, size_t data_len,
                      bool reverse, size_t offset, long lead_pos, size_t rows, size_t len, uint32_t *Out, size_t ldo,
                      cudaStream_t s) {
  if (!rows || !len) return;
  k_gather<<<blocks(rows * len), 128, 0, s>>>(A, lda, row_stride, data_len, reverse, offset, lead_pos, rows, len, Out,
                                             ldo, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
}

void limb_add_rows(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t as, const uint32_t *B, size_t ldb,
                   size_t bs, uint32_t *C, size_t ldc, size_t cs, size_t rows, size_t cnt, cudaStream_t s) {
  if (!rows || !cnt) return;
  if (ctx.rns_store())
    k_addsub_rns<0><<<blocks(rows * cnt), 128, 0, s>>>(A, lda, as, B, ldb, bs, C, ldc, cs, cnt, rows * cnt,
                                                     ctx.tables());
  else
    k_addsub<0><<<blocks(rows * cnt), 128, 0, s>>>(A, lda, as, B, ldb, bs, C, ldc, cs, cnt, rows * cnt, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
}
void limb_sub_rows(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t as, const uint32_t *B, size_t ldb,
                   size_t bs, uint32_t *C, size_t ldc, size_t cs, size_t rows, size_t cnt, cudaStream_t s) {
  if (!rows || !cnt) return;
  if (ctx.rns_store())
    k_addsub_rns<1><<<blocks(rows * cnt), 128, 0, s>>>(A, lda, as, B, ldb, bs, C, ldc, cs, cnt, rows * cnt,
                                                     ctx.tables());
  else
    k_addsub<1><<<blocks(rows * cnt), 128, 0, s>>>(A, lda, as, B, ldb, bs, C, ldc, cs, cnt, rows * cnt, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
}
void limb_neg_rows(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t as, uint32_t *C, size_t ldc, size_t cs,
                   size_t rows, size_t cnt, cudaStream_t s) {
  if (!rows || !cnt) return;
  if (ctx.rns_store())
    k_addsub_rns<2><<<blocks(rows * cnt), 128, 0, s>>>(A, lda, as, nullptr, 0, 0, C, ldc, cs, cnt, rows * cnt,
                                                     ctx.tables());
  else
    k_addsub<2><<<blocks(rows * cnt), 128, 0, s>>>(A, lda, as, nullptr, 0, 0, C, ldc, cs, cnt, rows * cnt,
                                                   ctx.tables());
  CUDA_CHECK(cudaGetLastError());
}
void limb_copy_rows(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t as, uint32_t *C, size_t ldc,
                    size_t cs, size_t rows, size_t cnt, bool rev, cudaStream_t s) {
  if (!rows || !cnt) return;
  k_copy<<<blocks(rows * cnt), 128, 0, s>>>(A, lda, as, C, ldc, cs, cnt, rows * cnt, rev, ctx.coef_words());
  CUDA_CHECK(cudaGetLastError());
}
void limb_set_ui_rows(const RnsContext &ctx, uint32_t *C, size_t ldc, size_t cs, size_t rows, size_t cnt, uint32_t c,
                      cudaStream_t s) {
  if (!rows || !cnt) return;
  k_set_ui<<<blocks(rows * cnt), 128, 0, s>>>(C, ldc, cs, cnt, rows * cnt, c, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
}

void limb_add(const RnsContext &ctx, const uint32_t *A, size_t lda, const uint32_t *B, size_t ldb, uint32_t *C,
              size_t ldc, size_t count, cudaStream_t s) {
  limb_add_rows(ctx, A, lda, count, B, ldb, count, C, ldc, count, 1, count, s);
}
void limb_sub(const RnsContext &ctx, const uint32_t *A, size_t lda, const uint32_t *B, size_t ldb, uint32_t *C,
              size_t ldc, size_t count, cudaStream_t s) {
  limb_sub_rows(ctx, A, lda, count, B, ldb, count, C, ldc, count, 1, count, s);
}
void limb_neg(const RnsContext &ctx, const uint32_t *A, size_t lda, uint32_t *C, size_t ldc, size_t count,
              cudaStream_t s) {
  limb_neg_rows(ctx, A, lda, count, C, ldc, count, 1, count, s);
}
void limb_copy(const RnsContext &ctx, const uint32_t *A, size_t lda, uint32_t *C, size_t ldc, size_t count, bool rev,
               cudaStream_t s) {
  limb_copy_rows(ctx, A, lda, count, C, ldc, count, 1, count, rev, s);
}
void limb_set_ui(const RnsContext &ctx, uint32_t *C, size_t ldc, size_t count, uint32_t c, cudaStream_t s) {
  limb_set_ui_rows(ctx, C, ldc, count, 1, count, c, s);
}

uint32_t *limb_alloc(const RnsContext &ctx, size_t count, cudaStream_t s) {
  uint32_t *p = nullptr;
  CUDA_CHECK(cudaMallocAsync(&p, (size_t)ctx.coef_words() * (count ? count : 1) * 4, s));
  return p;
}
void limb_free(uint32_t *p, cudaStream_t s) {
  if (p) CUDA_CHECK(cudaFreeAsync(p, s));
}
