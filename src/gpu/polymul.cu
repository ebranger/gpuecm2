/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "gpu/polymul.h"

#include "gpu/arith32.cuh"
#include "gpu/cuda_check.h"
#include "gpu/ntt.h"

namespace {

int ceil_log2_sz(size_t n) {
  int k = 0;
  while (((size_t)1 << k) < n) k++;
  return k;
}

/* Y[(q*rows + r)*K + k] = sum_{i+j=k} XA[r*sa + i] .* XB[r*sb + j] (Montgomery),
   rows of length T, K = sa + sb - 1 */
__global__ void k_piece_products(const uint32_t *__restrict__ XA, const uint32_t *__restrict__ XB,
                                 uint32_t *__restrict__ Y, size_t rows, int sa, int sb, size_t T, RnsTables t) {
  const int q = blockIdx.y, K = sa + sb - 1;
  const uint32_t p = t.p[q], pinv = t.pinv[q];
  const size_t total = rows * K * T;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x) {
    const size_t pos = e % T, rk = e / T, r = rk / K;
    const int k = (int)(rk % K);
    uint32_t acc = 0;
    for (int i = k - sb + 1 > 0 ? k - sb + 1 : 0; i <= k && i < sa; i++) {
      uint32_t a = XA[(((size_t)q * rows + r) * sa + i) * T + pos];
      uint32_t b = XB[(((size_t)q * rows + r) * sb + (k - i)) * T + pos];
      acc = add_mod(acc, mul_mont(a, b, p, pinv), p);
    }
    Y[((size_t)q * rows * K + rk) * T + pos] = acc;
  }
}

/* Z[(q*rows + r)*lc + c] = sum_k Y[((q*rows + r)*K + k)*T + c - k*h], 0 <= c - k*h < T */
__global__ void k_overlap_add(const uint32_t *__restrict__ Y, uint32_t *__restrict__ Z, size_t rows, int K,
                              size_t T, size_t h, size_t lc, RnsTables t) {
  const int q = blockIdx.y;
  const uint32_t p = t.p[q];
  const size_t total = rows * lc;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x) {
    const size_t r = e / lc, c = e % lc;
    const size_t kmax = c / h;  // terms k with k*h <= c < k*h + T: k in [kmax - 1, kmax] (T = 2h)
    uint32_t acc = 0;
    for (size_t k = kmax > 0 ? kmax - 1 : 0; k <= kmax && k < (size_t)K; k++)
      if (c - k * h < T) acc = add_mod(acc, Y[(((size_t)q * rows + r) * K + k) * T + (c - k * h)], p);
    Z[((size_t)q * rows + r) * lc + c] = acc;
  }
}

unsigned grid_x(size_t n) { size_t g = (n + 255) / 256; return (unsigned)(g < 65535 ? g : 65535); }

}  // namespace

void poly_mul(const RnsContext &ctx, const uint32_t *A, size_t lda, size_t la, const uint32_t *B, size_t ldb,
              size_t lb, uint32_t *C, size_t ldc, size_t rows, cudaStream_t s) {
  if (rows == 0 || la == 0 || lb == 0) return;
  const size_t lc = la + lb - 1;
  const int P = ctx.P();
  if (lc <= ((size_t)1 << ctx.logTmax())) {
    const int logT = ceil_log2_sz(lc) < 3 ? 3 : ceil_log2_sz(lc);
    const size_t T = (size_t)1 << logT, sz = (size_t)P * rows * T * 4;
    uint32_t *XA, *XB;
    CUDA_CHECK(cudaMallocAsync(&XA, sz, s));
    CUDA_CHECK(cudaMallocAsync(&XB, sz, s));
    ctx.from_limbs(A, lda, rows, la, XA, T, s);
    ctx.from_limbs(B, ldb, rows, lb, XB, T, s);
    ntt_forward(ctx, XA, rows, logT, s);
    ntt_forward(ctx, XB, rows, logT, s);
    ntt_pointwise_mul(ctx, XA, XB, XA, rows * T, s);
    ntt_inverse(ctx, XA, rows, logT, s);
    ctx.to_limbs(XA, rows, T, 0, lc, C, ldc, logT, 1, s);
    CUDA_CHECK(cudaFreeAsync(XA, s));
    CUDA_CHECK(cudaFreeAsync(XB, s));
    return;
  }
  /* split into pieces of h = Tmax/2 coefficients; piece products have
     < T = 2h coefficients, so the cyclic convolutions do not wrap */
  const int logT = ctx.logTmax();
  const size_t T = (size_t)1 << logT, h = T / 2;
  const int sa = (int)((la + h - 1) / h), sb = (int)((lb + h - 1) / h), K = sa + sb - 1;
  uint32_t *XA, *XB, *Y, *Z;
  CUDA_CHECK(cudaMallocAsync(&XA, (size_t)P * rows * sa * T * 4, s));
  CUDA_CHECK(cudaMallocAsync(&XB, (size_t)P * rows * sb * T * 4, s));
  ctx.from_limbs_pieces(A, lda, rows, la, h, sa, XA, T, s);
  ctx.from_limbs_pieces(B, ldb, rows, lb, h, sb, XB, T, s);
  ntt_forward(ctx, XA, rows * sa, logT, s);
  ntt_forward(ctx, XB, rows * sb, logT, s);
  CUDA_CHECK(cudaMallocAsync(&Y, (size_t)P * rows * K * T * 4, s));
  k_piece_products<<<dim3(grid_x(rows * K * T), P), 256, 0, s>>>(XA, XB, Y, rows, sa, sb, T, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFreeAsync(XA, s));
  CUDA_CHECK(cudaFreeAsync(XB, s));
  ntt_inverse(ctx, Y, rows * K, logT, s);
  CUDA_CHECK(cudaMallocAsync(&Z, (size_t)P * rows * lc * 4, s));
  k_overlap_add<<<dim3(grid_x(rows * lc), P), 256, 0, s>>>(Y, Z, rows, K, T, h, lc, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFreeAsync(Y, s));
  ctx.to_limbs(Z, rows, lc, 0, lc, C, ldc, logT, 1, s);
  CUDA_CHECK(cudaFreeAsync(Z, s));
}
