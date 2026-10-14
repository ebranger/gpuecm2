/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Batched NTTs. Lengths up to 4096 are done by one kernel per batch (kRow),
   with each thread holding 8 elements and doing up to 3 radix-2 levels in
   registers per pass, exchanging data through shared memory between passes.
   Longer transforms use the four-step method T = R*C: R-point transforms down
   the columns (kCol), a twist by omega_T^(l*k1), then C-point transforms
   along the rows (kRow). */
#include "gpu/ntt.h"

#include "gpu/arith32.cuh"
#include "gpu/cuda_check.h"

namespace {

constexpr int NTT_THREADS = 512;
constexpr int TILE = 4096;  // elements per block

template <bool INV>
__device__ __forceinline__ void bfly(uint32_t &u, uint32_t &v, uint32_t w, uint32_t wp, uint32_t p) {
  if (!INV) {
    uint32_t s = add_mod(u, v, p), d = sub_mod(u, v, p);
    u = s;
    v = mul_shoup(d, w, wp, p);
  } else {
    uint32_t t = mul_shoup(v, w, wp, p);
    v = sub_mod(u, t, p);
    u = add_mod(u, t, p);
  }
}

/* A pass does R radix-2 levels (strides sb*2^(R-1) .. sb) of an M-point
   transform. Groups of G = 2^R elements {g + sb*i}; each thread holds 8/G
   groups. Element (u, i) of thread qq is at index elem_index(qq, u, i). */
template <int R>
__device__ __forceinline__ int elem_index(int qq, int u, int i, int sb) {
  constexpr int G = 1 << R, NG = 8 / G;
  int t = qq * NG + u;
  return (t / sb) * G * sb + (t % sb) + sb * i;
}

template <int R, bool INV>
__device__ __forceinline__ void do_pass(uint32_t v[8], int qq, int sb, int M, const uint32_t *sw,
                                        const uint32_t *swp, uint32_t p) {
  constexpr int G = 1 << R, NG = 8 / G;
#pragma unroll
  for (int u = 0; u < NG; u++) {
    const int rr = (qq * NG + u) % sb;
    uint32_t *e = v + u * G;
#pragma unroll
    for (int step = 0; step < R; step++) {
      const int l = INV ? step : R - 1 - step;  // level: half-size s = sb << l
      const int f = M / (2 * (sb << l));
#pragma unroll
      for (int i = 0; i < G; i++) {
        if (i & (1 << l)) continue;
        const int ex = (rr + sb * (i & ((1 << l) - 1))) * f;
        bfly<INV>(e[i], e[i + (1 << l)], sw[ex], swp[ex], p);
      }
    }
  }
}

template <int R>
__device__ __forceinline__ void index_set(int idx[8], int qq, int sb) {
#pragma unroll
  for (int u = 0; u < 8 / (1 << R); u++)
#pragma unroll
    for (int i = 0; i < (1 << R); i++) idx[u * (1 << R) + i] = elem_index<R>(qq, u, i, sb);
}

/* pass schedule: forward = top levels first; the first forward pass (last
   inverse pass) has r0 = logM - 3*(npass-1) levels */
__device__ __forceinline__ void pass_params(int logM, int k, bool inv, int &r, int &sb) {
  const int npass = (logM + 2) / 3, r0 = logM - 3 * (npass - 1);
  const int kk = inv ? npass - 1 - k : k;  // position in forward order
  r = kk == 0 ? r0 : 3;
  const int levels_above = kk == 0 ? 0 : r0 + 3 * (kk - 1);  // levels done before this pass (forward)
  sb = 1 << (logM - levels_above - r);
}

template <bool INV>
__device__ __forceinline__ void run_pass(uint32_t v[8], int r, int qq, int sb, int M, const uint32_t *sw,
                                         const uint32_t *swp, uint32_t p) {
  if (r == 3) do_pass<3, INV>(v, qq, sb, M, sw, swp, p);
  else if (r == 2) do_pass<2, INV>(v, qq, sb, M, sw, swp, p);
  else do_pass<1, INV>(v, qq, sb, M, sw, swp, p);
}

__device__ __forceinline__ void get_index_set(int idx[8], int r, int qq, int sb) {
  if (r == 3) index_set<3>(idx, qq, sb);
  else if (r == 2) index_set<2>(idx, qq, sb);
  else index_set<1>(idx, qq, sb);
}

/* M-point transforms of contiguous rows: X[(q*nrows + row)*M + i]. */
template <bool INV>
__global__ void __launch_bounds__(NTT_THREADS) kRow(uint32_t *__restrict__ X, size_t nrows, int logM, RnsTables t) {
  extern __shared__ uint32_t sm[];
  const int M = 1 << logM, tpr = M / 8, rows_per_block = TILE / M;
  uint32_t *sw = sm, *swp = sm + M / 2, *sh = sm + M;
  const int q = blockIdx.y;
  const uint32_t p = t.p[q];
  const uint32_t *wt = INV ? t.iw : t.w, *wtp = INV ? t.iwp : t.wp;
  const int stride = 4096 / M;
  for (int i = threadIdx.x; i < M / 2; i += NTT_THREADS) {
    sw[i] = wt[(size_t)q * 2048 + i * stride];
    swp[i] = wtp[(size_t)q * 2048 + i * stride];
  }
  const int lrow = threadIdx.x / tpr, qq = threadIdx.x % tpr;
  const size_t row = (size_t)blockIdx.x * rows_per_block + lrow;
  const bool active = row < nrows;
  uint32_t *g = X + ((size_t)q * nrows + row) * M;
  uint32_t *s = sh + lrow * (M + M / 8);
  const int npass = (logM + 2) / 3;
  uint32_t v[8];
  int idx[8], r, sb;
  pass_params(logM, 0, INV, r, sb);
  get_index_set(idx, r, qq, sb);
  if (active)
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = g[idx[i]];
  __syncthreads();
  for (int k = 0; k < npass; k++) {
    if (active) run_pass<INV>(v, r, qq, sb, M, sw, swp, p);
    if (k == npass - 1) break;
#pragma unroll
    for (int i = 0; i < 8; i++) s[idx[i] + (idx[i] >> 3)] = v[i];
    __syncthreads();
    pass_params(logM, k + 1, INV, r, sb);
    get_index_set(idx, r, qq, sb);
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = s[idx[i] + (idx[i] >> 3)];
    __syncthreads();
  }
  if (active)
#pragma unroll
    for (int i = 0; i < 8; i++) g[idx[i]] = v[i];
}

/* R-point transforms down the columns of each T = R*C polynomial, with the
   four-step twist (after the forward pass, before the inverse pass).
   COLS = 4096/R columns per block, R/8 threads per column. */
template <bool INV>
__global__ void __launch_bounds__(NTT_THREADS) kCol(uint32_t *__restrict__ X, size_t rows, int logT, int logR,
                                                     RnsTables t) {
  extern __shared__ uint32_t sm[];
  const int R = 1 << logR, COLS = TILE / R, logC = logT - logR;
  const size_t C = (size_t)1 << logC, T = (size_t)1 << logT;
  uint32_t *sw = sm, *swp = sm + R / 2, *sh = sm + R;
  const int q = blockIdx.y;
  const uint32_t p = t.p[q], pinv = t.pinv[q];
  const uint32_t *wt = INV ? t.iw : t.w, *wtp = INV ? t.iwp : t.wp;
  const int stride = 4096 / R;
  for (int i = threadIdx.x; i < R / 2; i += NTT_THREADS) {
    sw[i] = wt[(size_t)q * 2048 + i * stride];
    swp[i] = wtp[(size_t)q * 2048 + i * stride];
  }
  const size_t blocks_per_row = C / COLS;
  const size_t row = blockIdx.x / blocks_per_row;
  const int lc = threadIdx.x % COLS, qq = threadIdx.x / COLS;
  const size_t l = (blockIdx.x % blocks_per_row) * COLS + lc;
  uint32_t *g = X + ((size_t)q * rows + row) * T + l;
  const uint32_t *TH = (INV ? t.iTH : t.TH) + ((size_t)q << (t.logTmax - 11));
  const uint32_t *TL = (INV ? t.iTL : t.TL) + (size_t)q * 2048;
  const int tw_shift = t.logTmax - logT;  // omega_T = omega_Tmax^(2^tw_shift)
  const int npass = (logR + 2) / 3;

  uint32_t v[8];
  int idx[8], r, sb;
  pass_params(logR, 0, INV, r, sb);
  get_index_set(idx, r, qq, sb);
#pragma unroll
  for (int i = 0; i < 8; i++) v[i] = g[(size_t)idx[i] * C];
  if (INV) {  // untwist
#pragma unroll
    for (int i = 0; i < 8; i++) {
      unsigned k1 = __brev((unsigned)idx[i]) >> (32 - logR);
      size_t E = ((size_t)l * k1) << tw_shift;
      v[i] = mul_mont(v[i], mul_mont(TH[E >> 11], TL[E & 2047], p, pinv), p, pinv);
    }
  }
  __syncthreads();
  for (int k = 0; k < npass; k++) {
    run_pass<INV>(v, r, qq, sb, R, sw, swp, p);
    if (k == npass - 1) break;
#pragma unroll
    for (int i = 0; i < 8; i++) sh[COLS * (idx[i] + (idx[i] >> 3)) + lc] = v[i];
    __syncthreads();
    pass_params(logR, k + 1, INV, r, sb);
    get_index_set(idx, r, qq, sb);
#pragma unroll
    for (int i = 0; i < 8; i++) v[i] = sh[COLS * (idx[i] + (idx[i] >> 3)) + lc];
    __syncthreads();
  }
  if (!INV) {  // twist
#pragma unroll
    for (int i = 0; i < 8; i++) {
      unsigned k1 = __brev((unsigned)idx[i]) >> (32 - logR);
      size_t E = ((size_t)l * k1) << tw_shift;
      v[i] = mul_mont(v[i], mul_mont(TH[E >> 11], TL[E & 2047], p, pinv), p, pinv);
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) g[(size_t)idx[i] * C] = v[i];
}

__global__ void k_pointwise(const uint32_t *__restrict__ X, const uint32_t *__restrict__ Y, uint32_t *__restrict__ Z,
                            size_t n, RnsTables t) {
  const int q = blockIdx.y;
  const uint32_t p = t.p[q], pinv = t.pinv[q];
  const size_t off = (size_t)q * n;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    Z[off + i] = mul_mont(X[off + i], Y[off + i], p, pinv);
}

/* rows of length T per prime; total rows `rows` in X (and Y/Z per mode) */
template <int MODE>  // 0: Y[j] = X[2j]*X[2j+1];  1: Z[2j] = B[j]*C[2j+1], Z[2j+1] = B[j]*C[2j]
__global__ void k_pointwise_pairs(const uint32_t *__restrict__ B, const uint32_t *__restrict__ C,
                                  uint32_t *__restrict__ Z, size_t count, size_t T, bool sub_one, RnsTables t) {
  const int q = blockIdx.y;
  const uint32_t p = t.p[q], pinv = t.pinv[q], ri = sub_one ? t.rinv[q] : 0;
  const size_t total = (MODE == 0 ? count : 2 * count) * T;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x) {
    const size_t row = e / T, pos = e % T;
    if (MODE == 0) {
      const uint32_t *x = B + ((size_t)q * 2 * count + 2 * row) * T + pos;
      Z[((size_t)q * count + row) * T + pos] = sub_mod(mul_mont(x[0], x[T], p, pinv), ri, p);
    } else {
      const size_t j = row / 2, partner = row ^ 1;
      Z[((size_t)q * 2 * count + row) * T + pos] =
          mul_mont(B[((size_t)q * count + j) * T + pos], C[((size_t)q * 2 * count + partner) * T + pos], p, pinv);
    }
  }
}

size_t row_smem(int logM) { return ((size_t)(1 << logM) + TILE + TILE / 8) * 4; }
size_t col_smem(int logR) { return ((size_t)(1 << logR) + TILE + TILE / 8) * 4; }

template <bool INV>
void launch_row(const RnsContext &ctx, uint32_t *X, size_t nrows, int logM, cudaStream_t s) {
  const size_t M = (size_t)1 << logM, rows_per_block = TILE / M;
  dim3 grid((unsigned)((nrows + rows_per_block - 1) / rows_per_block), (unsigned)ctx.P());
  kRow<INV><<<grid, NTT_THREADS, row_smem(logM), s>>>(X, nrows, logM, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
}

template <bool INV>
void launch_col(const RnsContext &ctx, uint32_t *X, size_t rows, int logT, int logR, cudaStream_t s) {
  const size_t C = (size_t)1 << (logT - logR), COLS = TILE >> logR;
  dim3 grid((unsigned)(rows * (C / COLS)), (unsigned)ctx.P());
  kCol<INV><<<grid, NTT_THREADS, col_smem(logR), s>>>(X, rows, logT, logR, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
}

int split_logR(int logT) { return logT - 12 > 3 ? logT - 12 : 3; }

void check_len(const RnsContext &ctx, int logT) {
  if (logT < 3 || logT > ctx.logTmax()) {
    fprintf(stderr, "ntt: unsupported length 2^%d (max 2^%d)\n", logT, ctx.logTmax());
    exit(3);
  }
}

}  // namespace

void ntt_forward(const RnsContext &ctx, uint32_t *X, size_t rows, int logT, cudaStream_t s) {
  check_len(ctx, logT);
  if (logT <= 12) {
    launch_row<false>(ctx, X, rows, logT, s);
  } else {
    const int logR = split_logR(logT);
    launch_col<false>(ctx, X, rows, logT, logR, s);
    launch_row<false>(ctx, X, rows << logR, logT - logR, s);
  }
}

void ntt_inverse(const RnsContext &ctx, uint32_t *X, size_t rows, int logT, cudaStream_t s) {
  check_len(ctx, logT);
  if (logT <= 12) {
    launch_row<true>(ctx, X, rows, logT, s);
  } else {
    const int logR = split_logR(logT);
    launch_row<true>(ctx, X, rows << logR, logT - logR, s);
    launch_col<true>(ctx, X, rows, logT, logR, s);
  }
}

void ntt_pointwise_mul(const RnsContext &ctx, const uint32_t *X, const uint32_t *Y, uint32_t *Z, size_t rows_T,
                       cudaStream_t s) {
  dim3 grid((unsigned)((rows_T + 255) / 256 < 4096 ? (rows_T + 255) / 256 : 4096), (unsigned)ctx.P());
  k_pointwise<<<grid, 256, 0, s>>>(X, Y, Z, rows_T, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
}

void ntt_pointwise_adjacent(const RnsContext &ctx, const uint32_t *X, uint32_t *Y, size_t count, size_t T,
                            bool sub_one, cudaStream_t s) {
  if (!count) return;
  size_t b = (count * T + 255) / 256;
  k_pointwise_pairs<0><<<dim3((unsigned)(b < 4096 ? b : 4096), ctx.P()), 256, 0, s>>>(X, nullptr, Y, count, T,
                                                                                    sub_one, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
}

void ntt_pointwise_swap(const RnsContext &ctx, const uint32_t *B, const uint32_t *C, uint32_t *Z, size_t count,
                        size_t T, cudaStream_t s) {
  if (!count) return;
  size_t b = (2 * count * T + 255) / 256;
  k_pointwise_pairs<1><<<dim3((unsigned)(b < 4096 ? b : 4096), ctx.P()), 256, 0, s>>>(B, C, Z, count, T,
                                                                                    false, ctx.tables());
  CUDA_CHECK(cudaGetLastError());
}
