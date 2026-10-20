/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "gpu/rns.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>

#include "gpu/arith32.cuh"
#include "gpu/cuda_check.h"
#include "host/mpz_class_lite.h"

/* ------------------------------------------------------------------------- */
/* host helpers                                                               */

namespace {

uint64_t powmod64(uint64_t a, uint64_t e, uint64_t p) {
  uint64_t r = 1;
  a %= p;
  while (e) {
    if (e & 1) r = r * a % p;
    a = a * a % p;
    e >>= 1;
  }
  return r;
}

bool is_prime32(uint64_t n) {  // deterministic for n < 3215031751
  if (n < 2) return false;
  for (uint64_t a : {2u, 3u, 5u, 7u}) if (n % a == 0) return n == a;
  uint64_t d = n - 1;
  int s = 0;
  while (!(d & 1)) { d >>= 1; s++; }
  for (uint64_t a : {2u, 3u, 5u, 7u}) {
    uint64_t x = powmod64(a, d, n);
    if (x == 1 || x == n - 1) continue;
    bool comp = true;
    for (int i = 1; i < s && comp; i++) { x = x * x % n; if (x == n - 1) comp = false; }
    if (comp) return false;
  }
  return true;
}

uint32_t shoup(uint32_t w, uint32_t p) { return (uint32_t)(((uint64_t)w << 32) / p); }
uint32_t to_mont(uint64_t x, uint32_t p) { return (uint32_t)((x << 32) % p); }

}  // namespace

template <class T>
const T *RnsContext::upload(const std::vector<T> &v) {
  T *d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  allocs_.push_back(d);
  return d;
}

RnsContext::~RnsContext() {
  for (void *p : allocs_) cudaFree(p);
}

bool RnsContext::init(const mpz_t n, size_t max_operand_len, int want_logT, std::string *err) {
  const int nbits = (int)mpz_sizeinbase(n, 2);
  const int nl = (nbits + 31) / 32;
  int lglen = 0;
  while (((size_t)1 << lglen) < max_operand_len) lglen++;
  /* exact CRT needs M > 2 * max_operand_len * n^2 */
  const int need_bits = 2 * nbits + lglen + 2;

  Z M;
  int k;
  for (k = want_logT; k >= 12; k--) {
    primes_.clear();
    mpz_set_ui(M, 1);
    for (uint64_t c = ((1ULL << 31) - 1) >> k; c >= 1; c--) {
      uint64_t p = (c << k) + 1;
      if (p < (1ULL << 28)) break;
      if (!is_prime32(p)) continue;
      primes_.push_back((uint32_t)p);
      mpz_mul_ui(M, M, (unsigned long)p);
      if ((int)mpz_sizeinbase(M, 2) >= need_bits) break;
    }
    if ((int)mpz_sizeinbase(M, 2) >= need_bits) break;
  }
  if (k < 12) {
    *err = "not enough NTT primes for a modulus of this size";
    return false;
  }
  const int P = (int)primes_.size();
  const size_t Tmax = (size_t)1 << k;
  t_.P = P; t_.nlimbs = nl; t_.logTmax = k;

  std::vector<uint32_t> pinv(P), k32(P), k64(P), v(P), rinv(P), pow32((size_t)P * nl), Mi((size_t)P * nl);
  std::vector<uint64_t> mu(P);
  std::vector<uint32_t> crt((size_t)(k + 1) * 2 * P), crtp(crt.size());
  std::vector<uint32_t> w((size_t)P * 2048), wp(w.size()), iw(w.size()), iwp(w.size());
  std::vector<uint32_t> TL((size_t)P * 2048), iTL(TL.size()), TH((size_t)P * (Tmax / 2048)), iTH(TH.size());
  Z Mq, t, nmod;
  for (int q = 0; q < P; q++) {
    const uint64_t p = primes_[q];
    uint32_t x = 1;
    for (int i = 0; i < 5; i++) x *= 2 - (uint32_t)p * x;  // p^-1 mod 2^32
    pinv[q] = 0u - x;
    k32[q] = (uint32_t)((1ULL << 32) % p);
    k64[q] = (uint32_t)((uint64_t)k32[q] * k32[q] % p);
    v[q] = (uint32_t)((1ULL << 60) / p);
    rinv[q] = (uint32_t)powmod64(k32[q], p - 2, p);
    mu[q] = UINT64_MAX / p;  // == floor(2^64 / p) for odd p
    uint64_t pw = 1;
    for (int j = 0; j < nl; j++) { pow32[(size_t)q * nl + j] = (uint32_t)pw; pw = pw * k32[q] % p; }

    mpz_divexact_ui(Mq, M, (unsigned long)p);
    mpz_mod(t, Mq, n);
    std::vector<uint32_t> limbs(nl, 0);
    size_t cnt;
    mpz_export(limbs.data(), &cnt, -1, 4, 0, 0, t);
    for (int j = 0; j < nl; j++) Mi[(size_t)q * nl + j] = limbs[j];
    uint64_t inv = powmod64(mpz_fdiv_ui(Mq, (unsigned long)p), p - 2, p);  // (M/p)^-1 mod p
    for (int lt = 0; lt <= k; lt++)
      for (int m = 0; m < 2; m++) {
        uint64_t c = inv * powmod64(powmod64(2, lt, p), p - 2, p) % p;  // / 2^lt
        if (m) c = (c << 32) % p;                                        // * 2^32
        size_t idx = ((size_t)lt * 2 + m) * P + q;
        crt[idx] = (uint32_t)c;
        crtp[idx] = shoup((uint32_t)c, (uint32_t)p);
      }

    /* roots of unity */
    uint64_t g = 3;
    while (powmod64(g, (p - 1) / 2, p) != p - 1) g++;
    const uint64_t om = powmod64(g, (p - 1) >> k, p), iom = powmod64(om, p - 2, p);  // order 2^k
    const uint64_t o4096 = powmod64(om, Tmax / 4096, p), io4096 = powmod64(iom, Tmax / 4096, p);
    uint64_t a = 1, b = 1;
    for (int i = 0; i < 2048; i++) {
      size_t idx = (size_t)q * 2048 + i;
      w[idx] = (uint32_t)a; wp[idx] = shoup((uint32_t)a, (uint32_t)p);
      iw[idx] = (uint32_t)b; iwp[idx] = shoup((uint32_t)b, (uint32_t)p);
      a = a * o4096 % p; b = b * io4096 % p;
    }
    a = 1; b = 1;
    for (int i = 0; i < 2048; i++) {
      TL[(size_t)q * 2048 + i] = to_mont(a, (uint32_t)p);
      iTL[(size_t)q * 2048 + i] = to_mont(b, (uint32_t)p);
      a = a * om % p; b = b * iom % p;
    }
    const uint64_t s = powmod64(om, 2048, p), is = powmod64(iom, 2048, p);
    a = 1; b = 1;
    for (size_t h = 0; h < Tmax / 2048; h++) {
      TH[(size_t)q * (Tmax / 2048) + h] = to_mont(a, (uint32_t)p);
      iTH[(size_t)q * (Tmax / 2048) + h] = to_mont(b, (uint32_t)p);
      a = a * s % p; b = b * is % p;
    }
  }
  std::vector<uint32_t> nl_n(nl, 0), nl_nmM(nl, 0);
  size_t cnt;
  mpz_export(nl_n.data(), &cnt, -1, 4, 0, 0, n);
  mpz_mod(nmod, M, n);
  mpz_sub(t, n, nmod);
  mpz_export(nl_nmM.data(), &cnt, -1, 4, 0, 0, t);
  t_.ntop = (double)nl_n[nl - 1] + (nl >= 2 ? (double)nl_n[nl - 2] / 4294967296.0 : 0.0) +
            (nl >= 3 ? (double)nl_n[nl - 3] / 18446744073709551616.0 : 0.0);

  t_.p = upload(primes_); t_.pinv = upload(pinv); t_.k32 = upload(k32); t_.k64 = upload(k64);
  t_.v = upload(v); t_.rinv = upload(rinv); t_.mu = upload(mu); t_.pow32 = upload(pow32); t_.Mi = upload(Mi);
  t_.n = upload(nl_n); t_.nmM = upload(nl_nmM);
  t_.crt = upload(crt); t_.crtp = upload(crtp);
  t_.w = upload(w); t_.wp = upload(wp); t_.iw = upload(iw); t_.iwp = upload(iwp);
  t_.TL = upload(TL); t_.TH = upload(TH); t_.iTL = upload(iTL); t_.iTH = upload(iTH);
  return true;
}

/* ------------------------------------------------------------------------- */
/* limbs -> RNS                                                               */

namespace {

/* Both conversions are matrix products over the coefficients (columns):
     to RNS:    X[q][c] = sum_j pow32[q][j] * limb_j(c)  (mod p_q)
     to limbs:  S[j][c] = sum_q Mi[q][j] * y_q(c)        (96-bit column sums)
   computed with tiles of rows x coefficients: each of the 256 threads of a
   block computes TM rows x TN coefficients (rows 16 apart, coefficients 16
   apart), the operands staged in shared memory GB_K at a time, so a
   multiply-add needs well under one load. TM (rows per thread) is chosen per
   modulus to waste little on the last row tile. */
constexpr int GB_K = 16, GB_THREADS = 256;
constexpr int CONV_THREADS = 256;

template <int TM, int TN>
__device__ __forceinline__ void tile_mac(uint32_t (&lo)[TM][TN], uint32_t (&mid)[TM][TN], uint32_t (&hi)[TM][TN],
                                         const uint32_t *As, int lda, const uint32_t *Bs, int ldb, int k, int tm,
                                         int tn) {
  uint32_t a[TM], b[TN];
#pragma unroll
  for (int i = 0; i < TM; i++) a[i] = As[k * lda + tm + 16 * i];
#pragma unroll
  for (int i = 0; i < TN; i++) b[i] = Bs[k * ldb + tn + 16 * i];
#pragma unroll
  for (int i = 0; i < TM; i++)
#pragma unroll
    for (int j = 0; j < TN; j++) mac96(lo[i][j], mid[i][j], hi[i][j], a[i], b[j]);
}

/* limbs -> RNS: rows are the primes, k runs over the limbs */
template <int TM>
__global__ void __launch_bounds__(GB_THREADS) k_to_rns(const uint32_t *__restrict__ A, size_t ld, RowMap src_map,
                                                       size_t src_len, size_t piece_len, size_t npieces, size_t K,
                                                       uint32_t *__restrict__ X, size_t xrows, size_t T,
                                                       size_t dst_offset, bool reverse, RnsTables t) {
  constexpr int TN = 4, BM = 16 * TM, BN = 16 * TN;
  __shared__ uint32_t As[GB_K * (BM + 1)];
  __shared__ uint32_t Bs[GB_K * BN];
  __shared__ size_t src_sh[BN], dst_sh[BN];
  const int tid = threadIdx.x, tm = tid / 16, tn = tid % 16;
  const size_t n0 = (size_t)blockIdx.x * BN;
  const int m0 = blockIdx.y * BM, P = t.P, nl = t.nlimbs;
  if (tid < BN) {
    const size_t g = n0 + tid;
    size_t src = SIZE_MAX, dst = SIZE_MAX;
    if (g < K) {
      const size_t rp = g / piece_len, pos = g - rp * piece_len;  // output row (row-piece), position in piece
      const size_t r = rp / npieces, in_row = (rp - r * npieces) * piece_len + pos;
      if (in_row < src_len) src = src_map(r) + in_row;
      dst = rp * T + dst_offset + (reverse ? piece_len - 1 - pos : pos);
    }
    src_sh[tid] = src;
    dst_sh[tid] = dst;
  }
  uint32_t lo[TM][TN] = {}, mid[TM][TN] = {}, hi[TM][TN] = {};
  for (int k0 = 0; k0 < nl; k0 += GB_K) {
    __syncthreads();
    for (int e = tid; e < GB_K * BM; e += GB_THREADS) {  // As[k][m] = pow32[m0+m][k0+k]
      const int m = e / GB_K, k = e % GB_K;
      As[k * (BM + 1) + m] = (m0 + m < P && k0 + k < nl) ? __ldg(t.pow32 + (size_t)(m0 + m) * nl + k0 + k) : 0;
    }
    for (int e = tid; e < GB_K * BN; e += GB_THREADS) {  // Bs[k][n] = limb k0+k of coefficient n0+n
      const int k = e / BN, n = e % BN;
      const size_t src = src_sh[n];
      Bs[k * BN + n] = (src != SIZE_MAX && k0 + k < nl) ? A[(size_t)(k0 + k) * ld + src] : 0;
    }
    __syncthreads();
#pragma unroll 4
    for (int k = 0; k < GB_K; k++) tile_mac<TM, TN>(lo, mid, hi, As, BM + 1, Bs, BN, k, tm, tn);
  }
#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int q = m0 + tm + 16 * i;
    if (q >= P) continue;
    const uint32_t p = t.p[q], k64 = t.k64[q], k32 = t.k32[q];
    const uint64_t mu = t.mu[q];
#pragma unroll
    for (int j = 0; j < TN; j++) {
      const size_t dst = dst_sh[tn + 16 * j];
      if (dst == SIZE_MAX) continue;
      const uint64_t vv = (uint64_t)hi[i][j] * k64 + (uint64_t)mid[i][j] * k32 + lo[i][j];
      X[(size_t)q * xrows * T + dst] = reduce64(vv, p, mu);
    }
  }
}

/* RNS -> limbs, one dot product per thread (for small and very large moduli,
   where it beats the tiled kernel below): exact CRT  x = sum y_q (M/p_q) - alpha M,
   y_q = x_q * (M/p_q)^-1 * scale mod p_q,  alpha = floor(sum y_q/p_q + 1/4) */
template <int CB>
__global__ void __launch_bounds__(CONV_THREADS) k_to_limbs_dot(const uint32_t *__restrict__ X, size_t rows, size_t T,
                                                           size_t lo_i, size_t cnt, size_t K,
                                                           uint32_t *__restrict__ Out, size_t ld, RowMap dst_map,
                                                           const uint32_t *__restrict__ cs,
                                                           const uint32_t *__restrict__ csp, RnsTables t) {
  extern __shared__ uint32_t sh[];
  const int P = t.P, nl = t.nlimbs, W = nl + 3;
  uint32_t *ys = sh;                     // [P][CB]
  uint32_t *c0 = ys + (size_t)P * CB;    // [W][CB] column sums (96-bit), later the result
  uint32_t *c1 = c0 + (size_t)W * CB;
  uint32_t *c2 = c1 + (size_t)W * CB;
  __shared__ uint32_t alpha_sh[CB];
  const size_t g0 = (size_t)blockIdx.x * CB;

  for (int idx = threadIdx.x; idx < P * CB; idx += CONV_THREADS) {
    int q = idx / CB, c = idx % CB;
    size_t g = g0 + c;
    uint32_t y = 0;
    if (g < K) {
      size_t row = g / cnt, i = lo_i + g % cnt;
      y = mul_shoup(X[((size_t)q * rows + row) * T + i], cs[q], csp[q], t.p[q]);
    }
    ys[idx] = y;
  }
  __syncthreads();

  const int c = threadIdx.x % CB, grp = threadIdx.x / CB, ngrp = CONV_THREADS / CB;
  for (int j = grp; j < nl; j += ngrp) {
    uint32_t a = 0, b = 0, d = 0;
    for (int q = 0; q < P; q++) mac96(a, b, d, ys[q * CB + c], __ldg(t.Mi + (size_t)q * nl + j));
    c0[j * CB + c] = a; c1[j * CB + c] = b; c2[j * CB + c] = d;
  }
  if (threadIdx.x < CB) {
    uint32_t a = 0, b = 0, d = 0;
    for (int q = 0; q < P; q++) mac96(a, b, d, ys[q * CB + threadIdx.x], t.v[q]);
    /* sum is (d:b:a) / 2^60; alpha = floor(sum + 1/4) */
    uint64_t top = ((uint64_t)d << 32 | b) + (1u << 26);  // + 2^58 / 2^32
    alpha_sh[threadIdx.x] = (uint32_t)(top >> 28);        // / 2^(60-32)
  }
  __syncthreads();
  if (threadIdx.x >= CB) return;
  const size_t g = g0 + c;
  if (g >= K) return;

  /* carry-propagate: limb j = c0[j] + c1[j-1] + c2[j-2] + carry; result into c0 */
  uint64_t carry = 0;
  uint32_t prev1 = 0, prev2 = 0;  // c1[j-1], c2[j-2]
  uint32_t pc2 = 0;               // c2[j-1]
  for (int j = 0; j < W; j++) {
    uint32_t a = j < nl ? c0[j * CB + c] : 0;
    uint32_t b1 = j < nl ? c1[j * CB + c] : 0;
    uint32_t b2 = j < nl ? c2[j * CB + c] : 0;
    uint64_t s = (uint64_t)a + prev1 + prev2 + carry;
    c0[j * CB + c] = (uint32_t)s;
    carry = s >> 32;
    prev1 = b1;
    prev2 = pc2;
    pc2 = b2;
  }
  /* += alpha * (n - M mod n) */
  const uint32_t al = alpha_sh[c];
  carry = 0;
  for (int j = 0; j < W; j++) {
    uint64_t s = (uint64_t)c0[j * CB + c] + (j < nl ? (uint64_t)al * t.nmM[j] : 0) + carry;
    c0[j * CB + c] = (uint32_t)s;
    carry = s >> 32;
  }
  /* reduce mod n: quotient estimate q <= floor(v/n), subtract q*n, fix up */
  double tt = 0;
  for (int j = W - 1; j >= nl - 2 && j >= 0; j--) tt = tt * 4294967296.0 + (double)c0[j * CB + c];
  if (nl >= 2) tt /= 4294967296.0;  // scale: v / 2^(32*(nl-1))
  double qd = floor(tt / t.ntop) - 1.0;
  if (qd < 0) qd = 0;
  const uint64_t qq = (uint64_t)qd;
  for (int h = 0; h < 2; h++) {
    const uint32_t qh = (uint32_t)(qq >> (32 * h));
    if (!qh) continue;
    uint64_t cr = 0;
    int64_t br = 0;
    for (int j = 0; j + h < W; j++) {
      uint64_t prod = (j < nl ? (uint64_t)qh * t.n[j] : 0) + cr;
      cr = prod >> 32;
      int64_t sd = (int64_t)c0[(j + h) * CB + c] - (uint32_t)prod + br;
      c0[(j + h) * CB + c] = (uint32_t)sd;
      br = sd >> 32;
    }
  }
  for (int it = 0; it < 4; it++) {  // at most a few subtractions of n remain
    int64_t br = 0;
    for (int j = 0; j < W; j++) {  // compare by trial subtraction into c1
      int64_t sd = (int64_t)c0[j * CB + c] - (j < nl ? t.n[j] : 0) + br;
      c1[j * CB + c] = (uint32_t)sd;
      br = sd >> 32;
    }
    if (br < 0) break;
    for (int j = 0; j < W; j++) c0[j * CB + c] = c1[j * CB + c];
  }
  const size_t row = g / cnt, o = dst_map(row) + (g - row * cnt);
  for (int j = 0; j < nl; j++) Out[(size_t)j * ld + o] = c0[j * CB + c];
}

template <int CB>
size_t to_limbs_dot_smem(int P, int nl) { return ((size_t)P * CB + 3 * (size_t)(nl + 3) * CB) * 4; }

/* RNS -> limbs: exact CRT  x = sum y_q (M/p_q) - alpha M,
   y_q = x_q * (M/p_q)^-1 * scale mod p_q,  alpha = floor(sum y_q/p_q + 1/4).
   A block handles TL_N coefficients, the output limbs a tile of 16 TM at a
   time: the column sums of the tile (k over the primes) go to shared memory,
   then one thread per coefficient adds them into the result, propagating the
   carries (and alpha (n - M mod n)) from tile to tile in registers; at the end
   it reduces mod n. Shared memory: the result and one tile of sums. */
constexpr int TL_N = 32;
size_t to_limbs_smem(int nl, int TM) {
  const size_t BM = 16 * TM;
  return ((size_t)nl * TL_N + 3 * BM * TL_N + (size_t)GB_K * (BM + 1) + (size_t)GB_K * TL_N) * 4 +
         TL_N * sizeof(size_t);
}

template <int TM>
__global__ void __launch_bounds__(GB_THREADS) k_to_limbs(const uint32_t *__restrict__ X, size_t rows, size_t T,
                                                         size_t lo_i, size_t cnt, size_t K,
                                                         uint32_t *__restrict__ Out, size_t ld, RowMap dst_map,
                                                         const uint32_t *__restrict__ cs,
                                                         const uint32_t *__restrict__ csp, RnsTables t) {
  constexpr int TN = 2, BM = 16 * TM, BN = TL_N;
  extern __shared__ uint32_t sh[];
  const int P = t.P, nl = t.nlimbs, W = nl + 3;
  uint32_t *res = sh;                                                // [nl][BN] result limbs
  uint32_t *s0 = res + (size_t)nl * BN, *s1 = s0 + BM * BN, *s2 = s1 + BM * BN;  // [BM][BN] tile sums
  uint32_t *As = s2 + BM * BN, *Bs = As + GB_K * (BM + 1);
  size_t *col = (size_t *)(Bs + GB_K * BN);  // offset of coefficient n in a row of X, or SIZE_MAX
  const int tid = threadIdx.x, tm = tid / 16, tn = tid % 16;
  const size_t n0 = (size_t)blockIdx.x * BN;
  if (tid < BN) {
    const size_t g = n0 + tid;
    col[tid] = g < K ? (g / cnt) * T + lo_i + g % cnt : SIZE_MAX;
  }
  const size_t qstride = rows * T;
  /* carry state of coefficient tid (threads tid < BN): limb j gets
     c0[j] + c1[j-1] + c2[j-2] + carry, then + alpha (n - M mod n)_j + carry2 */
  uint64_t carry = 0, carry2 = 0;
  uint32_t prev1 = 0, prev2 = 0, pc2 = 0, al = 0;
  if (tid < BN && col[tid] != SIZE_MAX) {  /* alpha = floor(sum_q y_q v_q / 2^60 + 1/4) */
    const size_t o = col[tid];
    uint32_t a = 0, b = 0, d = 0;
    for (int q = 0; q < P; q++)
      mac96(a, b, d, mul_shoup(X[(size_t)q * qstride + o], cs[q], csp[q], t.p[q]), t.v[q]);
    const uint64_t top = ((uint64_t)d << 32 | b) + (1u << 26);  // + 2^58 / 2^32
    al = (uint32_t)(top >> 28);                                // / 2^(60-32)
  }
  for (int m0 = 0; m0 < nl; m0 += BM) {
    uint32_t lo[TM][TN] = {}, mid[TM][TN] = {}, hi[TM][TN] = {};
    for (int k0 = 0; k0 < P; k0 += GB_K) {
      __syncthreads();
      for (int e = tid; e < GB_K * BM; e += GB_THREADS) {  // As[k][m] = Mi[k0+k][m0+m]
        const int k = e / BM, m = e % BM;
        As[k * (BM + 1) + m] = (k0 + k < P && m0 + m < nl) ? __ldg(t.Mi + (size_t)(k0 + k) * nl + m0 + m) : 0;
      }
      for (int e = tid; e < GB_K * BN; e += GB_THREADS) {  // Bs[k][n] = y_{k0+k} of coefficient n0+n
        const int k = e / BN, n = e % BN, q = k0 + k;
        const size_t o = col[n];
        Bs[k * BN + n] = (q < P && o != SIZE_MAX) ? mul_shoup(X[(size_t)q * qstride + o], cs[q], csp[q], t.p[q]) : 0;
      }
      __syncthreads();
#pragma unroll 4
      for (int k = 0; k < GB_K; k++) tile_mac<TM, TN>(lo, mid, hi, As, BM + 1, Bs, BN, k, tm, tn);
    }
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
      for (int jj = 0; jj < TN; jj++) {
        const int jl = tm + 16 * i, n = tn + 16 * jj;
        s0[jl * BN + n] = lo[i][jj];
        s1[jl * BN + n] = mid[i][jj];
        s2[jl * BN + n] = hi[i][jj];
      }
    __syncthreads();
    if (tid < BN) {
      const int jend = m0 + BM < nl ? m0 + BM : nl;
      for (int j = m0; j < jend; j++) {
        const int jl = j - m0;
        const uint64_t v1 = (uint64_t)s0[jl * BN + tid] + prev1 + prev2 + carry;
        carry = v1 >> 32;
        const uint64_t v2 = (uint64_t)(uint32_t)v1 + (uint64_t)al * t.nmM[j] + carry2;
        carry2 = v2 >> 32;
        res[j * BN + tid] = (uint32_t)v2;
        prev1 = s1[jl * BN + tid];
        prev2 = pc2;
        pc2 = s2[jl * BN + tid];
      }
    }
  }
  if (tid >= BN || col[tid] == SIZE_MAX) return;
  const int c = tid;
  /* the top three limbs (j = nl .. nl+2) in registers */
  uint32_t r[3];
  for (int j = nl; j < W; j++) {
    const uint64_t v1 = (uint64_t)prev1 + prev2 + carry;
    carry = v1 >> 32;
    const uint64_t v2 = (uint64_t)(uint32_t)v1 + carry2;
    carry2 = v2 >> 32;
    r[j - nl] = (uint32_t)v2;
    prev1 = 0;
    prev2 = pc2;
    pc2 = 0;
  }
  auto limb = [&](int j) -> uint32_t { return j < nl ? res[j * BN + c] : r[j - nl]; };
  auto set = [&](int j, uint32_t v) {
    if (j < nl) res[j * BN + c] = v;
    else r[j - nl] = v;
  };
  /* reduce mod n: quotient estimate q <= floor(v/n), subtract q*n, fix up */
  double tt = 0;
  for (int j = W - 1; j >= nl - 2 && j >= 0; j--) tt = tt * 4294967296.0 + (double)limb(j);
  if (nl >= 2) tt /= 4294967296.0;  // scale: v / 2^(32*(nl-1))
  double qd = floor(tt / t.ntop) - 1.0;
  if (qd < 0) qd = 0;
  const uint64_t qq = (uint64_t)qd;
  for (int h = 0; h < 2; h++) {  // v -= qq * n, one 32-bit digit of qq at a time
    const uint32_t qh = (uint32_t)(qq >> (32 * h));
    if (!qh) continue;
    uint64_t cr = 0;
    int64_t br = 0;
    for (int j = 0; j + h < W; j++) {
      const uint64_t prod = (j < nl ? (uint64_t)qh * t.n[j] : 0) + cr;
      cr = prod >> 32;
      const int64_t sd = (int64_t)limb(j + h) - (uint32_t)prod + br;
      set(j + h, (uint32_t)sd);
      br = sd >> 32;
    }
  }
  for (int it = 0; it < 4; it++) {  // at most a few subtractions of n remain: v >= n ?
    int64_t br = 0;
    for (int j = 0; j < W; j++) br = ((int64_t)limb(j) - (j < nl ? t.n[j] : 0) + br) >> 32;
    if (br < 0) break;
    br = 0;
    for (int j = 0; j < W; j++) {
      const int64_t sd = (int64_t)limb(j) - (j < nl ? t.n[j] : 0) + br;
      set(j, (uint32_t)sd);
      br = sd >> 32;
    }
  }
  const size_t g = n0 + c, row = g / cnt, oo = dst_map(row) + (g - row * cnt);
  for (int j = 0; j < nl; j++) Out[(size_t)j * ld + oo] = res[j * BN + c];
}

/* RNS -> limbs: exact CRT  x = sum y_q (M/p_q) - alpha M,
   y_q = x_q * (M/p_q)^-1 * scale mod p_q,  alpha = floor(sum y_q/p_q + 1/4).
   As k_to_limbs below, but the column sums of all limbs are kept in shared
   memory and the carries propagated at the end (faster while that fits twice
   in an SM: up to 3552 bits). */
size_t to_limbs_cs_smem(int nl, int TM) {
  return ((size_t)3 * nl * TL_N + (size_t)GB_K * (16 * TM + 1) + (size_t)GB_K * TL_N) * 4 + TL_N * sizeof(size_t);
}

template <int TM>
__global__ void __launch_bounds__(GB_THREADS) k_to_limbs_cs(const uint32_t *__restrict__ X, size_t rows, size_t T,
                                                         size_t lo_i, size_t cnt, size_t K,
                                                         uint32_t *__restrict__ Out, size_t ld, RowMap dst_map,
                                                         const uint32_t *__restrict__ cs,
                                                         const uint32_t *__restrict__ csp, RnsTables t) {
  constexpr int TN = 2, BM = 16 * TM, BN = TL_N;
  extern __shared__ uint32_t sh[];
  const int P = t.P, nl = t.nlimbs, W = nl + 3;
  uint32_t *c0 = sh, *c1 = c0 + (size_t)nl * BN, *c2 = c1 + (size_t)nl * BN;  // [nl][BN] column sums
  uint32_t *As = c2 + (size_t)nl * BN, *Bs = As + GB_K * (BM + 1);
  size_t *col = (size_t *)(Bs + GB_K * BN);  // offset of coefficient n in a row of X, or SIZE_MAX
  const int tid = threadIdx.x, tm = tid / 16, tn = tid % 16;
  const size_t n0 = (size_t)blockIdx.x * BN;
  if (tid < BN) {
    const size_t g = n0 + tid;
    col[tid] = g < K ? (g / cnt) * T + lo_i + g % cnt : SIZE_MAX;
  }
  const size_t qstride = rows * T;
  for (int m0 = 0; m0 < nl; m0 += BM) {
    uint32_t lo[TM][TN] = {}, mid[TM][TN] = {}, hi[TM][TN] = {};
    for (int k0 = 0; k0 < P; k0 += GB_K) {
      __syncthreads();
      for (int e = tid; e < GB_K * BM; e += GB_THREADS) {  // As[k][m] = Mi[k0+k][m0+m]
        const int k = e / BM, m = e % BM;
        As[k * (BM + 1) + m] = (k0 + k < P && m0 + m < nl) ? __ldg(t.Mi + (size_t)(k0 + k) * nl + m0 + m) : 0;
      }
      for (int e = tid; e < GB_K * BN; e += GB_THREADS) {  // Bs[k][n] = y_{k0+k} of coefficient n0+n
        const int k = e / BN, n = e % BN, q = k0 + k;
        const size_t o = col[n];
        Bs[k * BN + n] = (q < P && o != SIZE_MAX) ? mul_shoup(X[(size_t)q * qstride + o], cs[q], csp[q], t.p[q]) : 0;
      }
      __syncthreads();
#pragma unroll 4
      for (int k = 0; k < GB_K; k++) tile_mac<TM, TN>(lo, mid, hi, As, BM + 1, Bs, BN, k, tm, tn);
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
      const int j = m0 + tm + 16 * i;
      if (j >= nl) continue;
#pragma unroll
      for (int jj = 0; jj < TN; jj++) {
        const int n = tn + 16 * jj;
        c0[j * BN + n] = lo[i][jj];
        c1[j * BN + n] = mid[i][jj];
        c2[j * BN + n] = hi[i][jj];
      }
    }
  }
  __syncthreads();
  if (tid >= BN || col[tid] == SIZE_MAX) return;
  const int c = tid;
  const size_t o = col[c];
  /* alpha = floor(sum_q y_q v_q / 2^60 + 1/4) */
  uint32_t al;
  {
    uint32_t a = 0, b = 0, d = 0;
    for (int q = 0; q < P; q++)
      mac96(a, b, d, mul_shoup(X[(size_t)q * qstride + o], cs[q], csp[q], t.p[q]), t.v[q]);
    const uint64_t top = ((uint64_t)d << 32 | b) + (1u << 26);  // + 2^58 / 2^32
    al = (uint32_t)(top >> 28);                                // / 2^(60-32)
  }
  /* the result, W limbs: limbs < nl in c0, the top three in registers */
  uint32_t r[3] = {0, 0, 0};
  auto limb = [&](int j) -> uint32_t { return j < nl ? c0[j * BN + c] : r[j - nl]; };
  auto set = [&](int j, uint32_t v) {
    if (j < nl) c0[j * BN + c] = v;
    else r[j - nl] = v;
  };
  /* limb j = c0[j] + c1[j-1] + c2[j-2] (carry-propagated) + alpha (n - M mod n)_j */
  {
    uint64_t carry = 0, carry2 = 0;
    uint32_t prev1 = 0, prev2 = 0, pc2 = 0;
    for (int j = 0; j < W; j++) {
      const uint32_t a = j < nl ? c0[j * BN + c] : 0;
      const uint32_t b1 = j < nl ? c1[j * BN + c] : 0;
      const uint32_t b2 = j < nl ? c2[j * BN + c] : 0;
      const uint64_t s1 = (uint64_t)a + prev1 + prev2 + carry;
      carry = s1 >> 32;
      const uint64_t s2 = (uint64_t)(uint32_t)s1 + (j < nl ? (uint64_t)al * t.nmM[j] : 0) + carry2;
      carry2 = s2 >> 32;
      set(j, (uint32_t)s2);
      prev1 = b1;
      prev2 = pc2;
      pc2 = b2;
    }
  }
  /* reduce mod n: quotient estimate q <= floor(v/n), subtract q*n, fix up */
  double tt = 0;
  for (int j = W - 1; j >= nl - 2 && j >= 0; j--) tt = tt * 4294967296.0 + (double)limb(j);
  if (nl >= 2) tt /= 4294967296.0;  // scale: v / 2^(32*(nl-1))
  double qd = floor(tt / t.ntop) - 1.0;
  if (qd < 0) qd = 0;
  const uint64_t qq = (uint64_t)qd;
  for (int h = 0; h < 2; h++) {  // v -= qq * n, one 32-bit digit of qq at a time
    const uint32_t qh = (uint32_t)(qq >> (32 * h));
    if (!qh) continue;
    uint64_t cr = 0;
    int64_t br = 0;
    for (int j = 0; j + h < W; j++) {
      const uint64_t prod = (j < nl ? (uint64_t)qh * t.n[j] : 0) + cr;
      cr = prod >> 32;
      const int64_t sd = (int64_t)limb(j + h) - (uint32_t)prod + br;
      set(j + h, (uint32_t)sd);
      br = sd >> 32;
    }
  }
  for (int it = 0; it < 4; it++) {  // at most a few subtractions of n remain: v >= n ?
    int64_t br = 0;
    for (int j = 0; j < W; j++) br = ((int64_t)limb(j) - (j < nl ? t.n[j] : 0) + br) >> 32;
    if (br < 0) break;
    br = 0;
    for (int j = 0; j < W; j++) {
      const int64_t sd = (int64_t)limb(j) - (j < nl ? t.n[j] : 0) + br;
      set(j, (uint32_t)sd);
      br = sd >> 32;
    }
  }
  const size_t g = n0 + c, row = g / cnt, oo = dst_map(row) + (g - row * cnt);
  for (int j = 0; j < nl; j++) Out[(size_t)j * ld + oo] = c0[j * BN + c];
}

/* X[(q*xrows + r)*T + i] for the positions i outside [off, off + len): 0, or
   c (mod p) at i == lead */
__global__ void k_pad(uint32_t *__restrict__ X, size_t xrows, size_t T, size_t off, size_t len, long lead, int c,
                      RnsTables t) {
  const int q = blockIdx.y;
  const uint32_t p = t.p[q];
  const uint32_t cc = c >= 0 ? (uint32_t)c % p : p - (uint32_t)(-c) % p;
  const size_t pad = T - len, total = xrows * pad;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x) {
    const size_t r = e / pad, k = e - r * pad, i = k < off ? k : k + len;
    X[((size_t)q * xrows + r) * T + i] = (long)i == lead ? cc : 0;
  }
}

/* rows per thread (TM = 2, 3 or 4) for a matrix product with `rows` rows:
   little padding of the last tile, more rows per thread (fewer loads) */
int choose_tm(int rows) {
  int best = 4;
  double best_cost = 1e30;
  for (int tm = 4; tm >= 2; tm--) {
    const int bm = 16 * tm, padded = (rows + bm - 1) / bm * bm;
    const double cost = padded * (1.0 + 1.0 / tm);
    if (cost < best_cost - 1e-9) { best_cost = cost; best = tm; }
  }
  return best;
}

}  // namespace

void RnsContext::from_limbs_map(const uint32_t *A, size_t ld, const RowMap &src, size_t rows, size_t src_len,
                                size_t piece_len, size_t npieces, uint32_t *X, size_t T, size_t dst_offset,
                                bool reverse, cudaStream_t s, long lead_pos, int lead_c) const {
  const size_t xrows = rows * npieces, K = xrows * piece_len;
  /* the positions outside [dst_offset, dst_offset + piece_len) are written
     here (zeros, and the leading constant if it falls there), the others by
     k_to_rns: every word of X is written once */
  const bool lead_in_data = lead_pos >= (long)dst_offset && lead_pos < (long)(dst_offset + piece_len);
  if (piece_len < T && xrows) {
    const size_t pad = T - piece_len, total = xrows * pad, b = (total + 255) / 256;
    k_pad<<<dim3((unsigned)(b < 8192 ? b : 8192), t_.P), 256, 0, s>>>(X, xrows, T, dst_offset, piece_len,
                                                                      lead_in_data ? -1 : lead_pos, lead_c, t_);
    CUDA_CHECK(cudaGetLastError());
  }
  if (K == 0) return;
  const int tm = choose_tm(t_.P);
  const dim3 grid((unsigned)((K + 63) / 64), (unsigned)((t_.P + 16 * tm - 1) / (16 * tm)));
#define GPUECM2_TO_RNS(TM)                                                                                          \
  k_to_rns<TM><<<grid, GB_THREADS, 0, s>>>(A, ld, src, src_len, piece_len, npieces, K, X, xrows, T, dst_offset,   \
                                           reverse, t_)
  if (tm == 4) GPUECM2_TO_RNS(4);
  else if (tm == 3) GPUECM2_TO_RNS(3);
  else GPUECM2_TO_RNS(2);
#undef GPUECM2_TO_RNS
  CUDA_CHECK(cudaGetLastError());
  if (lead_in_data) add_const(X, xrows, T, (size_t)lead_pos, lead_c, s);  // after the data is written
}

void RnsContext::to_limbs_map(const uint32_t *X, size_t rows, size_t T, size_t lo, size_t cnt, uint32_t *Out,
                              size_t ld, const RowMap &dst, int logT_scale, int mont, cudaStream_t s) const {
  const size_t K = rows * cnt;
  if (K == 0) return;
  const uint32_t *cs = t_.crt + ((size_t)logT_scale * 2 + (mont ? 1 : 0)) * t_.P;
  const uint32_t *csp = t_.crtp + ((size_t)logT_scale * 2 + (mont ? 1 : 0)) * t_.P;
  const int nl = t_.nlimbs, P = t_.P;
  /* 32 or more limbs: tiled matrix products, keeping the column sums of all
     limbs while they leave room for two blocks per SM (up to 3552 bits),
     above that propagating the carries tile by tile; fewer limbs: one dot
     product per thread (faster there; bench/bench_conv). GPUECM2_TO_LIMBS =
     cs / tiled / dot overrides, for benchmarks. */
  const int tm = choose_tm(nl);
  const size_t smem_cs = to_limbs_cs_smem(nl, tm), smem = to_limbs_smem(nl, tm);
  int kind = nl < 32 ? 0 : smem_cs <= 48 * 1024 ? 1 : 2;  // 0 dot, 1 column sums kept, 2 per tile
  if (const char *e = getenv("GPUECM2_TO_LIMBS"))
    kind = !strcmp(e, "dot") ? 0 : !strcmp(e, "cs") && smem_cs <= 64 * 1024 ? 1 : 2;
  if (kind == 1) {
    const unsigned grid = (unsigned)((K + TL_N - 1) / TL_N);
#define GPUECM2_TO_LIMBS_CS(TM)                                                                                     \
  do {                                                                                                              \
    if (smem_cs > 48 * 1024)                                                                                        \
      CUDA_CHECK(cudaFuncSetAttribute(k_to_limbs_cs<TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_cs)); \
    k_to_limbs_cs<TM><<<grid, GB_THREADS, smem_cs, s>>>(X, rows, T, lo, cnt, K, Out, ld, dst, cs, csp, t_);          \
  } while (0)
    if (tm == 4) GPUECM2_TO_LIMBS_CS(4);
    else if (tm == 3) GPUECM2_TO_LIMBS_CS(3);
    else GPUECM2_TO_LIMBS_CS(2);
#undef GPUECM2_TO_LIMBS_CS
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  const bool tiled = kind == 2;
  if (!tiled) {
    const size_t limit = 46 * 1024;
    if (to_limbs_dot_smem<32>(P, nl) <= limit)
      k_to_limbs_dot<32><<<(unsigned)((K + 31) / 32), CONV_THREADS, to_limbs_dot_smem<32>(P, nl), s>>>(
          X, rows, T, lo, cnt, K, Out, ld, dst, cs, csp, t_);
    else if (to_limbs_dot_smem<16>(P, nl) <= limit)
      k_to_limbs_dot<16><<<(unsigned)((K + 15) / 16), CONV_THREADS, to_limbs_dot_smem<16>(P, nl), s>>>(
          X, rows, T, lo, cnt, K, Out, ld, dst, cs, csp, t_);
    else
      k_to_limbs_dot<8><<<(unsigned)((K + 7) / 8), CONV_THREADS, to_limbs_dot_smem<8>(P, nl), s>>>(
          X, rows, T, lo, cnt, K, Out, ld, dst, cs, csp, t_);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  const unsigned grid = (unsigned)((K + TL_N - 1) / TL_N);
#define GPUECM2_TO_LIMBS(TM)                                                                                        \
  do {                                                                                                              \
    if (smem > 48 * 1024)                                                                                           \
      CUDA_CHECK(cudaFuncSetAttribute(k_to_limbs<TM>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));     \
    k_to_limbs<TM><<<grid, GB_THREADS, smem, s>>>(X, rows, T, lo, cnt, K, Out, ld, dst, cs, csp, t_);              \
  } while (0)
  if (tm == 4) GPUECM2_TO_LIMBS(4);
  else if (tm == 3) GPUECM2_TO_LIMBS(3);
  else GPUECM2_TO_LIMBS(2);
#undef GPUECM2_TO_LIMBS
  CUDA_CHECK(cudaGetLastError());
}

namespace {
__global__ void k_add_const(uint32_t *X, size_t rows, size_t T, size_t pos, int c, RnsTables t) {
  const int q = blockIdx.y;
  const uint32_t p = t.p[q];
  const uint32_t cc = c >= 0 ? (uint32_t)c % p : p - (uint32_t)(-c) % p;
  for (size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += (size_t)gridDim.x * blockDim.x) {
    uint32_t *x = X + ((size_t)q * rows + r) * T + pos;
    uint32_t v = *x + cc;  // both < 2^31
    *x = v >= p ? v - p : v;
  }
}
}  // namespace

void RnsContext::add_const(uint32_t *X, size_t rows, size_t T, size_t pos, int c, cudaStream_t s) const {
  if (!rows || !c) return;
  size_t b = (rows + 127) / 128;
  k_add_const<<<dim3((unsigned)(b < 65535 ? b : 65535), t_.P), 128, 0, s>>>(X, rows, T, pos, c, t_);
  CUDA_CHECK(cudaGetLastError());
}
