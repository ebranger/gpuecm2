/* Copyright 2026 Erik Branger.
   The algorithms follow GMP-ECM's stage 2 (stage2.c, ecm_ntt.c, polyeval.c:
   Copyright 2001-2012 Paul Zimmermann, Alexander Kruppa, Dave Newman et al.,
   GNU LGPL version 3 or later), reorganised for batched GPU execution.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "gpu/polyalg.h"

#include <algorithm>

#include "gpu/arith32.cuh"
#include "gpu/cuda_check.h"
#include "gpu/limbops.h"
#include "gpu/ntt.h"
#include "gpu/polymul.h"

namespace {

int clog2(size_t n) {
  int k = 0;
  while (((size_t)1 << k) < n) k++;
  return k;
}

struct Rns {  // a stream-ordered RNS buffer: P x rows x T words
  uint32_t *X = nullptr;
  size_t rows = 0, T = 0;
  int logT = 0;
  cudaStream_t s = 0;
  Rns(const RnsContext &ctx, size_t rows_, int logT_, cudaStream_t s_) : rows(rows_), T((size_t)1 << logT_), logT(logT_), s(s_) {
    CUDA_CHECK(cudaMallocAsync(&X, (size_t)ctx.P() * rows * T * 4, s));
  }
  ~Rns() { if (X) cudaFreeAsync(X, s); }
  Rns(const Rns &) = delete;
  Rns &operator=(const Rns &) = delete;
};

/* RNS row(s) holding a polynomial of len coefficients from limb array A (ld),
   optionally reversed and shifted (see RnsContext::from_limbs_ex), transformed. */
void load_fwd(const RnsContext &ctx, Rns &R, const uint32_t *A, size_t ld, const RowMap &map, size_t rows,
              size_t src_len, size_t piece_len, size_t npieces, size_t offset, bool reverse, int lead_pos = -1) {
  ctx.from_limbs_map(A, ld, map, rows, src_len, piece_len, npieces, R.X, R.T, offset, reverse, R.s);
  if (lead_pos >= 0) ctx.add_const(R.X, rows * npieces, R.T, (size_t)lead_pos, 1, R.s);
  ntt_forward(ctx, R.X, rows * npieces, R.logT, R.s);
}

/* single-row cyclic product Z = X * Y (both transformed, length T), returned
   untransformed in Z (may alias X) */
void mul_inv(const RnsContext &ctx, const Rns &X, const Rns &Y, Rns &Z) {
  ntt_pointwise_mul(ctx, X.X, Y.X, Z.X, X.rows * X.T, Z.s);
  ntt_inverse(ctx, Z.X, Z.rows, Z.logT, Z.s);
}

/* Transient GPU memory allowed per batched step; larger batches are chunked. */
constexpr size_t kChunkBytes = (size_t)768 << 20;

size_t chunk_rows(const RnsContext &ctx, size_t T, size_t rows_per_item) {
  size_t per_item = (size_t)ctx.P() * T * 4 * rows_per_item;
  size_t c = kChunkBytes / per_item;
  return c ? c : 1;
}

__device__ __forceinline__ uint32_t mulmod_b(uint32_t a, uint32_t b, uint32_t p, uint64_t mu) {
  return reduce64((uint64_t)a * b, p, mu);
}

/* Products of pairs of monic degree-M polynomials directly on residues (no
   transform): X holds 2*count rows of M coefficients (left, right child),
   Y gets count rows of 2M coefficients (leading 1 implicit). Exact values. */
template <int M>
__global__ void k_small_monic_pairs(const uint32_t *__restrict__ X, uint32_t *__restrict__ Y, size_t count,
                                    RnsTables t) {
  const int q = blockIdx.y;
  const uint32_t p = t.p[q];
  const uint64_t mu = t.mu[q];
  for (size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x; j < count; j += (size_t)gridDim.x * blockDim.x) {
    const uint32_t *a = X + ((size_t)q * 2 * count + 2 * j) * M, *b = a + M;
    uint32_t c[2 * M];
#pragma unroll
    for (int k = 0; k < 2 * M; k++) c[k] = 0;
#pragma unroll
    for (int i = 0; i <= M; i++)
#pragma unroll
      for (int jj = 0; jj <= M; jj++) {
        if (i + jj >= 2 * M) continue;
        uint32_t ai = i < M ? a[i] : 1, bj = jj < M ? b[jj] : 1;
        c[i + jj] = add_mod(c[i + jj], mulmod_b(ai, bj, p, mu), p);
      }
    uint32_t *y = Y + ((size_t)q * count + j) * 2 * M;
#pragma unroll
    for (int k = 0; k < 2 * M; k++) y[k] = c[k];
  }
}

/* Transposed step for parents of degree 2M with two children of degree M:
   B holds count rows of 2M values, C holds 2*count rows of M child
   coefficients; Z[2j] (left child) = middle product of b_j and child 2j+1,
   Z[2j+1] = middle product of b_j and child 2j. Exact values. */
template <int M>
__global__ void k_small_tmul(const uint32_t *__restrict__ B, const uint32_t *__restrict__ C, uint32_t *__restrict__ Z,
                             size_t count, RnsTables t) {
  const int q = blockIdx.y;
  const uint32_t p = t.p[q];
  const uint64_t mu = t.mu[q];
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < 2 * count; e += (size_t)gridDim.x * blockDim.x) {
    const size_t j = e / 2;
    const uint32_t *b = B + ((size_t)q * count + j) * 2 * M;
    const uint32_t *c = C + ((size_t)q * 2 * count + (e ^ 1)) * M;  // sibling's polynomial
    uint32_t *z = Z + ((size_t)q * 2 * count + e) * M;
#pragma unroll
    for (int i = 0; i < M; i++) {
      uint32_t r = b[i + M];  // leading coefficient 1 of the sibling
#pragma unroll
      for (int jj = 0; jj < M; jj++) r = add_mod(r, mulmod_b(c[jj], b[i + jj], p, mu), p);
      z[i] = r;
    }
  }
}

unsigned gridx(size_t n) { size_t b = (n + 127) / 128; return (unsigned)(b < 65535 ? b : 65535); }

/* Logical operand for exact_product: row r starts at coefficient r*row_stride
   of A; see limb_gather_rows for the meaning of the other fields. */
struct Operand {
  const uint32_t *A;
  size_t ld, row_stride, data_len;
  bool reverse;
  size_t offset;
  long lead_pos;
  size_t len;  // logical length (number of coefficients)
};

/* Fallback for products too long for a single NTT: coefficients lo..lo+cnt-1
   of the exact products a_r * b_r (r < rows) go to Dst entry
   r*dst_stride + dst_off + i, computed with the splitting poly_mul. */
void exact_product(const RnsContext &ctx, const Operand &a, const Operand &b, size_t rows, size_t lo, size_t cnt,
                   uint32_t *Dst, size_t ldd, size_t dst_stride, size_t dst_off, cudaStream_t s) {
  uint32_t *Ea = limb_alloc(ctx, rows * a.len, s), *Eb = limb_alloc(ctx, rows * b.len, s);
  limb_gather_rows(ctx, a.A, a.ld, a.row_stride, a.data_len, a.reverse, a.offset, a.lead_pos, rows, a.len, Ea,
                   rows * a.len, s);
  limb_gather_rows(ctx, b.A, b.ld, b.row_stride, b.data_len, b.reverse, b.offset, b.lead_pos, rows, b.len, Eb,
                   rows * b.len, s);
  const size_t lc = a.len + b.len - 1;
  uint32_t *C = limb_alloc(ctx, rows * lc, s);
  poly_mul(ctx, Ea, rows * a.len, a.len, Eb, rows * b.len, b.len, C, rows * lc, rows, s);
  limb_free(Ea, s);
  limb_free(Eb, s);
  uint32_t *D = limb_alloc(ctx, rows * cnt, s);
  limb_gather_rows(ctx, C + lo, rows * lc, lc, cnt, false, 0, -1, rows, cnt, D, rows * cnt, s);
  limb_free(C, s);
  limb_scatter_rows(ctx, D, rows * cnt, rows, cnt, Dst, ldd, dst_stride, dst_off, s);
  limb_free(D, s);
}

}  // namespace

/* ------------------------------------------------------------------------- */

void tree_build(const RnsContext &ctx, const uint32_t *roots, size_t n, uint32_t *F, ProductTree *tree,
                cudaStream_t s, size_t batch) {
  const int L = clog2(n);
  const size_t B = batch, N = B * n;  // N: entries (and leading dimension) of every array
  if (tree) {
    tree->n = n;
    tree->batch = B;
    tree->L = L;
    tree->level.assign(L, nullptr);
  }
  if (N == 0) return;
  if (L == 0) {  // n == 1: F = x + f_0
    limb_copy(ctx, roots, N, F, N, N, false, s);
    return;
  }
  uint32_t *cur = limb_alloc(ctx, N, s);
  limb_copy(ctx, roots, N, cur, N, N, false, s);
  for (int l = 0; l < L; l++) {
    if (tree) tree->level[l] = cur;
    const size_t m = (size_t)1 << l;
    uint32_t *nxt = (l + 1 == L) ? F : limb_alloc(ctx, N, s);
    /* products of pairs of full nodes (degree m each): length-2m cyclic product,
       the wrapped x^2m term (= 1) is cancelled in the transform domain.
       Pair j of problem c starts at entry c*n + 2mj. */
    const size_t pairs = (n / m) / 2, total = B * pairs;
    const RowMap pmap = RowMap::grouped(2 * m, pairs, n);
    if (pairs && l + 1 > ctx.logTmax()) {
      for (size_t c = 0; c < B; c++) {
        Operand A{cur + c * n, N, 2 * m, m, false, 0, (long)m, m + 1};
        Operand Bo{cur + c * n + m, N, 2 * m, m, false, 0, (long)m, m + 1};
        exact_product(ctx, A, Bo, pairs, 0, 2 * m, nxt + c * n, N, 2 * m, 0, s);
      }
    } else if (pairs && m <= 2) {  // tiny degrees: direct products on residues
      for (size_t p0 = 0, cr = chunk_rows(ctx, 2 * m, 3); p0 < total; p0 += cr) {
        const size_t pr = std::min(cr, total - p0);
        uint32_t *X, *Y;
        CUDA_CHECK(cudaMallocAsync(&X, (size_t)ctx.P() * 2 * pr * m * 4, s));
        CUDA_CHECK(cudaMallocAsync(&Y, (size_t)ctx.P() * pr * 2 * m * 4, s));
        ctx.from_limbs_map(cur, N, pmap.from(p0), pr, 2 * m, m, 2, X, m, 0, false, s);
        if (m == 1) k_small_monic_pairs<1><<<dim3(gridx(pr), ctx.P()), 128, 0, s>>>(X, Y, pr, ctx.tables());
        else k_small_monic_pairs<2><<<dim3(gridx(pr), ctx.P()), 128, 0, s>>>(X, Y, pr, ctx.tables());
        CUDA_CHECK(cudaGetLastError());
        ctx.to_limbs_map(Y, pr, 2 * m, 0, 2 * m, nxt, N, pmap.from(p0), 0, 0, s);
        CUDA_CHECK(cudaFreeAsync(X, s));
        CUDA_CHECK(cudaFreeAsync(Y, s));
      }
    } else if (pairs) {
      const int logT = std::max(3, l + 1);
      const size_t T = (size_t)1 << logT;
      for (size_t p0 = 0, cr = chunk_rows(ctx, T, 3); p0 < total; p0 += cr) {
        const size_t pr = std::min(cr, total - p0);
        Rns X(ctx, 2 * pr, logT, s);
        load_fwd(ctx, X, cur, N, pmap.from(p0), pr, 2 * m, m, 2, 0, false, (int)m);
        Rns Y(ctx, pr, logT, s);
        ntt_pointwise_adjacent(ctx, X.X, Y.X, pr, T, T == 2 * m, s);
        ntt_inverse(ctx, Y.X, pr, logT, s);
        ctx.to_limbs_map(Y.X, pr, T, 0, 2 * m, nxt, N, pmap.from(p0), logT, 1, s);
      }
    }
    /* leftover (one per problem, at entry c*n + base): a full node times the
       partial node, or a single node copied up */
    const size_t base = pairs * 2 * m;
    if (base < n) {
      const size_t k = n - base;
      const RowMap cmap = RowMap::dense(n);
      if (k > m && clog2(k + 1) > ctx.logTmax()) {
        const size_t r = k - m;
        for (size_t c = 0; c < B; c++) {
          Operand A{cur + c * n + base, N, 0, m, false, 0, (long)m, m + 1};
          Operand Bo{cur + c * n + base + m, N, 0, r, false, 0, (long)r, r + 1};
          exact_product(ctx, A, Bo, 1, 0, k, nxt + c * n + base, N, 0, 0, s);
        }
      } else if (k > m) {
        const size_t r = k - m;
        const int logT = std::max(3, clog2(k + 1));
        Rns XA(ctx, B, logT, s), XB(ctx, B, logT, s);
        load_fwd(ctx, XA, cur + base, N, cmap, B, m, m, 1, 0, false, (int)m);
        load_fwd(ctx, XB, cur + base + m, N, cmap, B, r, r, 1, 0, false, (int)r);
        mul_inv(ctx, XA, XB, XA);
        ctx.to_limbs_map(XA.X, B, XA.T, 0, k, nxt + base, N, cmap, logT, 1, s);
      } else {
        limb_copy_rows(ctx, cur + base, N, n, nxt + base, N, n, B, k, false, s);
      }
    }
    if (!tree) limb_free(cur, s);
    cur = nxt;
  }
}

void tree_free(ProductTree &tree, cudaStream_t s) {
  for (uint32_t *p : tree.level) limb_free(p, s);
  tree.level.clear();
}

/* ------------------------------------------------------------------------- */

void poly_invert(const RnsContext &ctx, const uint32_t *F, size_t n, uint32_t *I, cudaStream_t s, size_t batch) {
  const size_t B = batch;
  if (n == 0 || B == 0) return;
  /* R = rev(F) as a power series: R[0] = 1, R[t] = F[n-t] (rows of n+1) */
  const size_t nr = n + 1, ldR = B * nr;
  uint32_t *R = limb_alloc(ctx, ldR, s);
  limb_set_ui_rows(ctx, R, ldR, nr, B, 1, 1, s);
  limb_copy_rows(ctx, F, B * n, n, R + 1, ldR, nr, B, n, true, s);
  const size_t cap = (size_t)1 << clog2(n), ecap = std::max<size_t>(1, cap / 2);
  const size_t ldI = B * cap, ldE = B * ecap;
  uint32_t *Ic = limb_alloc(ctx, ldI, s), *E = limb_alloc(ctx, ldE, s);
  limb_set_ui_rows(ctx, Ic, ldI, cap, B, 1, 1, s);  // I_1 = 1
  for (size_t k = 1; k < n; k *= 2) {
    /* Newton: R * I_k = 1 + x^k E (mod x^2k);  I_2k = I_k - x^k (I_k E mod x^k) */
    const int logT = std::max(3, clog2(2 * k));
    const size_t lr = std::min(2 * k, n + 1);
    if (logT > ctx.logTmax()) {
      for (size_t c = 0; c < B; c++) {
        Operand Rop{R + c * nr, ldR, 0, lr, false, 0, -1, lr}, Iop{Ic + c * cap, ldI, 0, k, false, 0, -1, k};
        exact_product(ctx, Rop, Iop, 1, k, k, E + c * ecap, ldE, 0, 0, s);
        Operand Eop{E + c * ecap, ldE, 0, k, false, 0, -1, k};
        exact_product(ctx, Iop, Eop, 1, 0, k, Ic + c * cap + k, ldI, 0, 0, s);
      }
      limb_neg_rows(ctx, Ic + k, ldI, cap, Ic + k, ldI, cap, B, k, s);
      continue;
    }
    Rns XR(ctx, B, logT, s), XI(ctx, B, logT, s);
    load_fwd(ctx, XR, R, ldR, RowMap::dense(nr), B, lr, lr, 1, 0, false);
    load_fwd(ctx, XI, Ic, ldI, RowMap::dense(cap), B, k, k, 1, 0, false);
    mul_inv(ctx, XR, XI, XR);  // wraps only below index k
    ctx.to_limbs_map(XR.X, B, XR.T, k, k, E, ldE, RowMap::dense(ecap), logT, 1, s);
    Rns XE(ctx, B, logT, s);
    load_fwd(ctx, XE, E, ldE, RowMap::dense(ecap), B, k, k, 1, 0, false);
    mul_inv(ctx, XE, XI, XE);  // degree < 2k - 1: no wrap
    ctx.to_limbs_map(XE.X, B, XE.T, 0, k, Ic + k, ldI, RowMap::dense(cap), logT, 1, s);
    limb_neg_rows(ctx, Ic + k, ldI, cap, Ic + k, ldI, cap, B, k, s);
  }
  limb_copy_rows(ctx, Ic, ldI, cap, I, B * n, n, B, n, false, s);
  limb_free(R, s);
  limb_free(Ic, s);
  limb_free(E, s);
}

/* ------------------------------------------------------------------------- */

MulModF::MulModF(const RnsContext &ctx, const uint32_t *F, const uint32_t *I, size_t n, cudaStream_t s, size_t batch)
    : ctx_(ctx), n_(n), batch_(batch), s_(s) {
  logT2_ = std::max(3, clog2(2 * n - 1));
  wrap_ = n >= 8 && (n & (n - 1)) == 0;
  logTq_ = wrap_ ? clog2(n) : logT2_;
  if (n < 2 || batch == 0) return;
  if (logT2_ > ctx.logTmax()) {  // products too long for one NTT: keep F and I, use exact products
    F_ = F;
    I_ = I;
    return;
  }
  const size_t P = ctx.P(), B = batch, T2 = (size_t)1 << logT2_, Tq = (size_t)1 << logTq_;
  CUDA_CHECK(cudaMallocAsync(&I_hat_, P * B * T2 * 4, s));
  CUDA_CHECK(cudaMallocAsync(&F_hat_, P * B * Tq * 4, s));
  /* I mod x^(n-1), transformed at length T2 */
  ctx.from_limbs_map(I, B * n, RowMap::dense(n), B, n - 1, n - 1, 1, I_hat_, T2, 0, false, s);
  ntt_forward(ctx, I_hat_, B, logT2_, s);
  /* F with its leading 1 (at x^n, which wraps to x^0 for the length-n product) */
  ctx.from_limbs_map(F, B * n, RowMap::dense(n), B, n, n, 1, F_hat_, Tq, 0, false, s);
  ctx.add_const(F_hat_, B, Tq, wrap_ ? 0 : n, 1, s);
  ntt_forward(ctx, F_hat_, B, logTq_, s);
}

MulModF::~MulModF() {
  /* stream-ordered: work queued on s_ that reads these may still be pending */
  if (I_hat_) cudaFreeAsync(I_hat_, s_);
  if (F_hat_) cudaFreeAsync(F_hat_, s_);
}

void MulModF::mulmod(const uint32_t *G, uint32_t *H, cudaStream_t s) {
  const RnsContext &ctx = ctx_;
  const size_t n = n_, B = batch_, ld = B * n, la = 2 * n - 1, ldA = B * la;
  if (n == 0 || B == 0) return;
  /* a = G * H, 2n - 1 coefficients */
  uint32_t *A = limb_alloc(ctx, ldA, s);
  if (F_) {  // fallback for long products
    const size_t lq = n - 1, ldQ = B * lq;
    uint32_t *Qr = limb_alloc(ctx, ldQ, s), *C = limb_alloc(ctx, ld, s);
    for (size_t c = 0; c < B; c++) {
      Operand Gop{G + c * n, ld, 0, n, false, 0, -1, n}, Hop{H + c * n, ld, 0, n, false, 0, -1, n};
      exact_product(ctx, Gop, Hop, 1, 0, la, A + c * la, ldA, 0, 0, s);
      Operand Aop{A + c * la + n, ldA, 0, lq, true, 0, -1, lq}, Iop{I_ + c * n, ld, 0, lq, false, 0, -1, lq};
      exact_product(ctx, Aop, Iop, 1, 0, lq, Qr + c * lq, ldQ, 0, 0, s);
      Operand Qop{Qr + c * lq, ldQ, 0, lq, true, 0, -1, lq}, Fop{F_ + c * n, ld, 0, n, false, 0, (long)n, n + 1};
      exact_product(ctx, Qop, Fop, 1, 0, n, C + c * n, ld, 0, 0, s);
    }
    limb_sub_rows(ctx, A, ldA, la, C, ld, n, H, ld, n, B, n, s);
    limb_free(A, s);
    limb_free(Qr, s);
    limb_free(C, s);
    return;
  }
  const RowMap dn = RowMap::dense(n), da = RowMap::dense(la);
  {
    Rns XG(ctx, B, logT2_, s), XH(ctx, B, logT2_, s);
    load_fwd(ctx, XG, G, ld, dn, B, n, n, 1, 0, false);
    load_fwd(ctx, XH, H, ld, dn, B, n, n, 1, 0, false);
    mul_inv(ctx, XG, XH, XG);
    ctx.to_limbs_map(XG.X, B, XG.T, 0, la, A, ldA, da, logT2_, 1, s);
  }
  if (n == 1) {  // F = x + f: G*H is a constant
    limb_copy(ctx, A, ldA, H, ld, B, false, s);
    limb_free(A, s);
    return;
  }
  /* quotient: rev(q) = rev(a_top) * I mod x^(n-1), a_top = a[n .. 2n-2] */
  const size_t lq = n - 1, ldQ = B * lq;
  uint32_t *Qr = limb_alloc(ctx, ldQ, s);
  {
    Rns XQ(ctx, B, logT2_, s);
    load_fwd(ctx, XQ, A + n, ldA, da, B, lq, lq, 1, 0, true);
    ntt_pointwise_mul(ctx, XQ.X, I_hat_, XQ.X, B * XQ.T, s);
    ntt_inverse(ctx, XQ.X, B, logT2_, s);
    ctx.to_limbs_map(XQ.X, B, XQ.T, 0, lq, Qr, ldQ, RowMap::dense(lq), logT2_, 1, s);
  }
  /* C = q * (F + x^n), cyclic of length n (wrap) or full; H = a_low - (qF)_low */
  uint32_t *C = limb_alloc(ctx, ld, s);
  {
    Rns XC(ctx, B, logTq_, s);
    load_fwd(ctx, XC, Qr, ldQ, RowMap::dense(lq), B, lq, lq, 1, 0, true);  // q = rev(Qr)
    ntt_pointwise_mul(ctx, XC.X, F_hat_, XC.X, B * XC.T, s);
    ntt_inverse(ctx, XC.X, B, logTq_, s);
    ctx.to_limbs_map(XC.X, B, XC.T, 0, n, C, ld, dn, logTq_, 1, s);
  }
  limb_sub_rows(ctx, A, ldA, la, C, ld, n, H, ld, n, B, n, s);
  if (wrap_) limb_add_rows(ctx, H, ld, n, A + n, ldA, la, H, ld, n, B, n - 1, s);  // + a[i+n] for i < n-1
  limb_free(A, s);
  limb_free(Qr, s);
  limb_free(C, s);
}

/* ------------------------------------------------------------------------- */

void poly_eval_tree(const RnsContext &ctx, const ProductTree &tree, const uint32_t *I, const uint32_t *H,
                    uint32_t *values, cudaStream_t s) {
  const size_t n = tree.n, B = tree.batch, N = B * n;
  if (N == 0) return;
  if (n == 1) {  // deg H < 1
    limb_copy(ctx, H, N, values, N, N, false, s);
    return;
  }
  const RowMap dn = RowMap::dense(n);
  /* b = rev(high(H * Q)), Q = rev(I) = Quo(x^(2n-1), F): coefficients n-1 .. 2n-2 */
  uint32_t *b = limb_alloc(ctx, N, s);
  uint32_t *top = limb_alloc(ctx, N, s);
  if (clog2(2 * n - 1) > ctx.logTmax()) {
    for (size_t c = 0; c < B; c++) {
      Operand Hop{H + c * n, N, 0, n, false, 0, -1, n}, Qop{I + c * n, N, 0, n, true, 0, -1, n};
      exact_product(ctx, Hop, Qop, 1, n - 1, n, top + c * n, N, 0, 0, s);
    }
  } else {
    const int logT = std::max(3, clog2(2 * n - 1));
    Rns XH(ctx, B, logT, s), XQ(ctx, B, logT, s);
    load_fwd(ctx, XH, H, N, dn, B, n, n, 1, 0, false);
    load_fwd(ctx, XQ, I, N, dn, B, n, n, 1, 0, true);
    mul_inv(ctx, XH, XQ, XH);
    ctx.to_limbs_map(XH.X, B, XH.T, n - 1, n, top, N, dn, logT, 1, s);
  }
  limb_copy_rows(ctx, top, N, n, b, N, n, B, n, true, s);
  limb_free(top, s);
  /* transposed descent: a node with input b (k values) and children c_L (deg l),
     c_R (deg m) passes to the left child the middle product of b and c_R and to
     the right child the middle product of b and c_L (with explicit leading ones):
     r_i = sum_j c_j b_(i+j), computed as coefficient i + deg(c) of rev(c) * b */
  for (int l = tree.L - 1; l >= 0; l--) {
    const size_t m = (size_t)1 << l;
    const uint32_t *ch = tree.level[l];
    uint32_t *bn = limb_alloc(ctx, N, s);
    const size_t pc = n / (2 * m), total = B * pc;
    const RowMap pmap = RowMap::grouped(2 * m, pc, n);  // parent j of problem c at c*n + 2mj
    const RowMap kmap = RowMap::grouped(m, 2 * pc, n);  // its children (rows 2j, 2j+1)
    if (pc && l + 1 > ctx.logTmax()) {
      for (size_t c = 0; c < B; c++) {
        Operand Bop{b + c * n, N, 2 * m, 2 * m, false, 0, -1, 2 * m};
        Operand CR{ch + c * n + m, N, 2 * m, m, true, 1, 0, m + 1}, CL{ch + c * n, N, 2 * m, m, true, 1, 0, m + 1};
        exact_product(ctx, Bop, CR, pc, m, m, bn + c * n, N, 2 * m, 0, s);  // left children
        exact_product(ctx, Bop, CL, pc, m, m, bn + c * n, N, 2 * m, m, s);  // right children
      }
    } else if (pc && m <= 2) {  // tiny degrees: direct middle products on residues
      for (size_t p0 = 0, cr = chunk_rows(ctx, 2 * m, 5); p0 < total; p0 += cr) {
        const size_t pr = std::min(cr, total - p0);
        uint32_t *XB, *XC, *Z;
        CUDA_CHECK(cudaMallocAsync(&XB, (size_t)ctx.P() * pr * 2 * m * 4, s));
        CUDA_CHECK(cudaMallocAsync(&XC, (size_t)ctx.P() * 2 * pr * m * 4, s));
        CUDA_CHECK(cudaMallocAsync(&Z, (size_t)ctx.P() * 2 * pr * m * 4, s));
        ctx.from_limbs_map(b, N, pmap.from(p0), pr, 2 * m, 2 * m, 1, XB, 2 * m, 0, false, s);
        ctx.from_limbs_map(ch, N, pmap.from(p0), pr, 2 * m, m, 2, XC, m, 0, false, s);
        if (m == 1) k_small_tmul<1><<<dim3(gridx(2 * pr), ctx.P()), 128, 0, s>>>(XB, XC, Z, pr, ctx.tables());
        else k_small_tmul<2><<<dim3(gridx(2 * pr), ctx.P()), 128, 0, s>>>(XB, XC, Z, pr, ctx.tables());
        CUDA_CHECK(cudaGetLastError());
        ctx.to_limbs_map(Z, 2 * pr, m, 0, m, bn, N, kmap.from(2 * p0), 0, 0, s);
        CUDA_CHECK(cudaFreeAsync(XB, s));
        CUDA_CHECK(cudaFreeAsync(XC, s));
        CUDA_CHECK(cudaFreeAsync(Z, s));
      }
    } else if (pc) {
      const int logT = std::max(3, l + 1);
      const size_t T = (size_t)1 << logT;
      for (size_t p0 = 0, cr = chunk_rows(ctx, T, 5); p0 < total; p0 += cr) {
        const size_t pr = std::min(cr, total - p0);
        Rns XB(ctx, pr, logT, s), XC(ctx, 2 * pr, logT, s), Z(ctx, 2 * pr, logT, s);
        load_fwd(ctx, XB, b, N, pmap.from(p0), pr, 2 * m, 2 * m, 1, 0, false);
        load_fwd(ctx, XC, ch, N, pmap.from(p0), pr, 2 * m, m, 2, 1, true, 0);
        ntt_pointwise_swap(ctx, XB.X, XC.X, Z.X, pr, T, s);
        ntt_inverse(ctx, Z.X, 2 * pr, logT, s);
        ctx.to_limbs_map(Z.X, 2 * pr, T, m, m, bn, N, kmap.from(2 * p0), logT, 1, s);
      }
    }
    const size_t base = pc * 2 * m;
    if (base < n) {
      const size_t k = n - base;
      if (k > m && clog2(k) > ctx.logTmax()) {
        const size_t r = k - m;
        for (size_t c = 0; c < B; c++) {
          const size_t o = c * n + base;
          Operand Bop{b + o, N, 0, k, false, 0, -1, k};
          Operand CR{ch + o + m, N, 0, r, true, 1, 0, r + 1}, CL{ch + o, N, 0, m, true, 1, 0, m + 1};
          exact_product(ctx, Bop, CR, 1, r, m, bn + o, N, 0, 0, s);
          exact_product(ctx, Bop, CL, 1, m, r, bn + o + m, N, 0, 0, s);
        }
      } else if (k > m) {
        const size_t r = k - m;
        const int logT = std::max(3, clog2(k));
        Rns XB(ctx, B, logT, s), XL(ctx, B, logT, s), XR(ctx, B, logT, s), Z(ctx, B, logT, s);
        load_fwd(ctx, XB, b + base, N, dn, B, k, k, 1, 0, false);
        load_fwd(ctx, XL, ch + base, N, dn, B, m, m, 1, 1, true, 0);
        load_fwd(ctx, XR, ch + base + m, N, dn, B, r, r, 1, 1, true, 0);
        mul_inv(ctx, XB, XR, Z);
        ctx.to_limbs_map(Z.X, B, Z.T, r, m, bn + base, N, dn, logT, 1, s);
        mul_inv(ctx, XB, XL, Z);
        ctx.to_limbs_map(Z.X, B, Z.T, m, r, bn + base + m, N, dn, logT, 1, s);
      } else {
        limb_copy_rows(ctx, b + base, N, n, bn + base, N, n, B, k, false, s);
      }
    }
    limb_free(b, s);
    b = bn;
  }
  limb_copy(ctx, b, N, values, N, N, false, s);
  limb_free(b, s);
}
