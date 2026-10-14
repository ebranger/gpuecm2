/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   The arithmetic is done on the twisted Edwards model of the Montgomery curve
   B v^2 = u^3 + A u^2 + u: a = (A+2)/B, d = (A-2)/B, (x, y) = (u/v, (u-1)/(u+1)),
   inversely u = (1+y)/(1-y) = (Z+Y)/(Z-Y). A root is GMP-ECM's Weierstrass
   x = (3u + A)/(3B). The point (u, 1) is (u(u+1) : u-1 : u+1 : u(u-1)). */
#include "gpu/roots.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu/cuda_check.h"
#include "gpu/ecc.cuh"

namespace {

constexpr int BLOCK = 256;
/* k_step occupancy: with 4-5 limbs per thread it needs ~126 registers (2
   blocks per SM); limiting it to 80 (3 blocks) is ~4% faster despite small
   spills (measured at 1024 bits). Other sizes are left to the compiler. */
constexpr int step_min_blocks(uint32_t tpi, uint32_t bits) {
  return (bits / 32 / tpi == 4 || bits / 32 / tpi == 5) ? 3 : 1;
}

/* device constants: n (plain), the rest in Montgomery form */
enum { C_N = 0, C_A, C_D, C_AM, C_C3, C_ONE, C_PX, C_PY, C_PZ, C_PT, C_COUNT };

template <uint32_t TPI, uint32_t BITS>
struct K {
  typedef EcEnv<TPI, BITS> Env;
  typedef typename Env::mem_t mem_t;
  typedef typename Env::bn_t bn_t;
  typedef typename Env::context_t ctx_t;
  typedef typename Env::Pt Pt;
};

template <uint32_t BITS>
__device__ __forceinline__ cgbn_mem_t<BITS> *mp(const cgbn_mem_t<BITS> *p) {
  return const_cast<cgbn_mem_t<BITS> *>(p);
}

template <uint32_t TPI, uint32_t BITS>
__device__ __forceinline__ void load_pt(EcEnv<TPI, BITS> &E, typename EcEnv<TPI, BITS>::Pt &P, const cgbn_mem_t<BITS> *X,
                                        const cgbn_mem_t<BITS> *Y, const cgbn_mem_t<BITS> *Z,
                                        const cgbn_mem_t<BITS> *T, size_t i) {
  cgbn_load(E.env, P.X, mp(&X[i]));
  cgbn_load(E.env, P.Y, mp(&Y[i]));
  cgbn_load(E.env, P.Z, mp(&Z[i]));
  cgbn_load(E.env, P.T, mp(&T[i]));
}

template <uint32_t TPI, uint32_t BITS>
__device__ __forceinline__ void store_pt(EcEnv<TPI, BITS> &E, const typename EcEnv<TPI, BITS>::Pt &P,
                                         cgbn_mem_t<BITS> *X, cgbn_mem_t<BITS> *Y, cgbn_mem_t<BITS> *Z,
                                         cgbn_mem_t<BITS> *T, size_t i) {
  cgbn_store(E.env, &X[i], P.X);
  cgbn_store(E.env, &Y[i], P.Y);
  cgbn_store(E.env, &Z[i], P.Z);
  cgbn_store(E.env, &T[i], P.T);
}

constexpr int WIN = 6, WTAB = (1 << WIN) - 1;  // fixed-base window and table entries per window

/* The doubling chain of the table is a single sequential computation, so it
   wants the lowest latency per operation rather than throughput: more threads
   per number than elsewhere (measured on an RTX 4070 Ti SUPER), but never a
   padded configuration (limb count not a multiple of TPI). */
constexpr uint32_t chain_tpi(uint32_t bits) {
  return bits % 1024 == 0 ? 32 : bits % 512 == 0 ? 16 : 8;
}

/* table[t*WTAB + 0] = 2^(WIN t) P for t < W (sequential; one instance per
   curve, i.e. per block: curve c's constants at cur + c*C_COUNT, its table
   at entry c*ntab) */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_pow2_chain(const cgbn_mem_t<BITS> *cur, uint32_t np0, int W, size_t ntab, cgbn_mem_t<BITS> *TX,
                             cgbn_mem_t<BITS> *TY, cgbn_mem_t<BITS> *TZ, cgbn_mem_t<BITS> *TT) {
  typedef K<TPI, BITS> T;
  cur += (size_t)blockIdx.x * C_COUNT;
  TX += blockIdx.x * ntab; TY += blockIdx.x * ntab; TZ += blockIdx.x * ntab; TT += blockIdx.x * ntab;
  typename T::ctx_t ctx(cgbn_no_checks);
  typename T::Env E(ctx, &cur[C_N], np0);
  typename T::bn_t a;
  cgbn_load(E.env, a, mp(&cur[C_A]));
  typename T::Pt Q;
  load_pt(E, Q, &cur[C_PX], &cur[C_PY], &cur[C_PZ], &cur[C_PT], 0);
  for (int t = 0; t < W; t++) {
    store_pt(E, Q, TX, TY, TZ, TT, (size_t)t * WTAB);
    for (int i = 0; i < WIN; i++) E.pdbl(Q, Q, a);
  }
}

/* round r of the table: table[t*WTAB + j-1] = (j - h) Q_t + h Q_t for
   j = h+1 .. min(2h, WTAB), h = 2^r, all windows t of all curves (one
   instance per entry) */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_table_round(const cgbn_mem_t<BITS> *cur, uint32_t np0, int W, int h, size_t ntab, size_t ncurves,
                              cgbn_mem_t<BITS> *TX, cgbn_mem_t<BITS> *TY, cgbn_mem_t<BITS> *TZ,
                              cgbn_mem_t<BITS> *TT) {
  typedef K<TPI, BITS> T;
  const int jn = (2 * h < WTAB ? 2 * h : WTAB) - h;  // entries per window in this round
  const size_t inst0 = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / TPI, per = (size_t)W * jn;
  if (inst0 >= per * ncurves) return;
  const size_t cv = inst0 / per, inst = inst0 - cv * per;
  cur += cv * C_COUNT;
  const int t = (int)(inst / jn), j = h + 1 + (int)(inst % jn);
  typename T::ctx_t ctx(cgbn_no_checks);
  typename T::Env E(ctx, &cur[C_N], np0);
  typename T::bn_t a, d;
  cgbn_load(E.env, a, mp(&cur[C_A]));
  cgbn_load(E.env, d, mp(&cur[C_D]));
  typename T::Pt P, Q;
  const size_t base = cv * ntab + (size_t)t * WTAB;
  load_pt(E, P, TX, TY, TZ, TT, base + (j - h) - 1);
  load_pt(E, Q, TX, TY, TZ, TT, base + h - 1);
  E.template padd<false>(P, P, Q, a, d);
  store_pt(E, P, TX, TY, TZ, TT, base + j - 1);
}

/* X, Y, T *= INV (= 1/Z): projective -> affine, Z = 1 implied */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_affine(const cgbn_mem_t<BITS> *cur, uint32_t np0, size_t count, cgbn_mem_t<BITS> *TX,
                         cgbn_mem_t<BITS> *TY, cgbn_mem_t<BITS> *TT, const cgbn_mem_t<BITS> *INV) {
  typedef K<TPI, BITS> T;
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (i >= count) return;
  typename T::ctx_t ctx(cgbn_no_checks);
  typename T::Env E(ctx, &cur[C_N], np0);
  typename T::bn_t v, inv;
  cgbn_load(E.env, inv, mp(&INV[i]));
  cgbn_mem_t<BITS> *arr[3] = {TX, TY, TT};
  for (auto *p : arr) {
    cgbn_load(E.env, v, &p[i]);
    E.mul(v, v, inv);
    cgbn_store(E.env, &p[i], v);
  }
}

/* F[c*count + i] = scal_i * P_c with curve c's fixed-base table (affine
   entries, at c*ntab): one mixed addition per nonzero window digit, no
   doublings */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_scalar_mul_tab(const cgbn_mem_t<BITS> *cur, uint32_t np0, const uint32_t *scal, int swords,
                                 const int32_t *nbits, size_t count, size_t ncurves, size_t ntab,
                                 const cgbn_mem_t<BITS> *TX, const cgbn_mem_t<BITS> *TY, const cgbn_mem_t<BITS> *TT,
                                 cgbn_mem_t<BITS> *FX, cgbn_mem_t<BITS> *FY, cgbn_mem_t<BITS> *FZ,
                                 cgbn_mem_t<BITS> *FT) {
  typedef K<TPI, BITS> T;
  const size_t out = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (out >= count * ncurves) return;
  const size_t cv = out / count, inst = out - cv * count;
  cur += cv * C_COUNT;
  TX += cv * ntab; TY += cv * ntab; TT += cv * ntab;
  typename T::ctx_t ctx(cgbn_no_checks);
  typename T::Env E(ctx, &cur[C_N], np0);
  typename T::bn_t a, d, one;
  cgbn_load(E.env, a, mp(&cur[C_A]));
  cgbn_load(E.env, d, mp(&cur[C_D]));
  cgbn_load(E.env, one, mp(&cur[C_ONE]));
  typename T::Pt R, Q;
  E.set_neutral(R, one);
  const int32_t nb = nbits[inst];
  const int bits = nb < 0 ? -nb : nb;
  const uint32_t *s = scal + inst * swords;
  for (int t = 0; t * WIN < bits; t++) {
    const int b0 = t * WIN, w0 = b0 >> 5, sh = b0 & 31;
    uint64_t two = s[w0] | (w0 + 1 < swords ? (uint64_t)s[w0 + 1] << 32 : 0);
    const int dg = (int)((two >> sh) & WTAB);
    if (!dg) continue;
    const size_t e = (size_t)t * WTAB + dg - 1;
    cgbn_load(E.env, Q.X, mp(&TX[e]));
    cgbn_load(E.env, Q.Y, mp(&TY[e]));
    cgbn_load(E.env, Q.T, mp(&TT[e]));
    E.template padd<true>(R, R, Q, a, d);
  }
  if (nb < 0) E.pneg(R);
  store_pt(E, R, FX, FY, FZ, FT, out);
}

/* Step n of all progressions of all curves. Each progression is a table of
   S+1 points (index (c*nprog + p)*(S+1) + i for curve c); one group of TPI
   threads per addition entry i += entry i+1 (i < S), reading the tables
   before the step (R*) and writing those after it (W*; entry S is constant
   and present in both). Before the step, entry 0 (the current value) is
   output to slot sl = slot[n*nprog + p] if that is >= 0 (or, without a slot
   table, sl = base[p] + n*inc if that is < count), as Z+Y (OS) and Z-Y
   (OD), so that u = OS/OD; slot sl is position sl % blen of output block
   sl / blen, which is kept in a ring of nring blocks; a block holds blen
   values of each curve, curve c's at c*blen. */
template <uint32_t TPI, uint32_t BITS>
__global__ void __launch_bounds__(BLOCK, step_min_blocks(TPI, BITS)) k_step(const cgbn_mem_t<BITS> *cur, uint32_t np0, const cgbn_mem_t<BITS> *RX,
                       const cgbn_mem_t<BITS> *RY, const cgbn_mem_t<BITS> *RZ, const cgbn_mem_t<BITS> *RT,
                       cgbn_mem_t<BITS> *WX, cgbn_mem_t<BITS> *WY, cgbn_mem_t<BITS> *WZ, cgbn_mem_t<BITS> *WT, int S,
                       size_t nprog, size_t ncurves, size_t n, const int32_t *slot, const int64_t *sbase,
                       int64_t sinc, size_t count, size_t blen, size_t nring, cgbn_mem_t<BITS> *OS,
                       cgbn_mem_t<BITS> *OD) {
  typedef K<TPI, BITS> T;
  const size_t e = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (e >= nprog * S * ncurves) return;
  const size_t cp = e / S;  // c*nprog + p
  const int i = (int)(e - cp * S);
  const size_t cv = cp / nprog, p = cp - cv * nprog;
  const size_t base = cp * (S + 1) + i;
  cur += cv * C_COUNT;
  typename T::ctx_t ctx(cgbn_no_checks);
  typename T::Env E(ctx, &cur[C_N], np0);
  typename T::bn_t a, d;
  cgbn_load(E.env, a, mp(&cur[C_A]));
  cgbn_load(E.env, d, mp(&cur[C_D]));
  typename T::Pt A, B;
  load_pt(E, A, RX, RY, RZ, RT, base);
  load_pt(E, B, RX, RY, RZ, RT, base + 1);
  if (i == 0) {
    int64_t sl = slot ? (int64_t)slot[n * nprog + p] : sbase[p] + (int64_t)n * sinc;
    if (!slot && sl >= (int64_t)count) sl = -1;
    if (sl >= 0) {
      const size_t blk = (size_t)sl / blen, o = ((blk % nring) * ncurves + cv) * blen + ((size_t)sl - blk * blen);
      typename T::bn_t t;
      E.add(t, A.Z, A.Y);
      cgbn_store(E.env, &OS[o], t);
      E.sub(t, A.Z, A.Y);
      cgbn_store(E.env, &OD[o], t);
    }
  }
  E.template padd<false>(A, A, B, a, d);
  store_pt(E, A, WX, WY, WZ, WT, base);
}

/* prefix products over chunks of `chunk` values (zero values count as 1) */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_chunk_prefix(const cgbn_mem_t<BITS> *cur, uint32_t np0, const cgbn_mem_t<BITS> *V, size_t count,
                               size_t chunk, cgbn_mem_t<BITS> *PRE, cgbn_mem_t<BITS> *TOT) {
  typedef K<TPI, BITS> T;
  const size_t c = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (c * chunk >= count) return;
  typename T::ctx_t ctx(cgbn_no_checks);
  typename T::Env E(ctx, &cur[C_N], np0);
  typename T::bn_t acc, z, one;
  cgbn_load(E.env, one, mp(&cur[C_ONE]));
  cgbn_set(E.env, acc, one);
  const size_t end = (c + 1) * chunk < count ? (c + 1) * chunk : count;
  for (size_t i = c * chunk; i < end; i++) {
    cgbn_load(E.env, z, mp(&V[i]));
    if (E.is_zero(z)) cgbn_set(E.env, z, one);
    E.mul(acc, acc, z);
    cgbn_store(E.env, &PRE[i], acc);
  }
  cgbn_store(E.env, &TOT[c], acc);
}

/* INV[i] = 1/V[i] from the chunk prefixes and the inverse of each chunk total */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_chunk_back(const cgbn_mem_t<BITS> *cur, uint32_t np0, const cgbn_mem_t<BITS> *V, size_t count,
                             size_t chunk, const cgbn_mem_t<BITS> *PRE, const cgbn_mem_t<BITS> *INVTOT,
                             cgbn_mem_t<BITS> *INV) {
  typedef K<TPI, BITS> T;
  const size_t c = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (c * chunk >= count) return;
  typename T::ctx_t ctx(cgbn_no_checks);
  typename T::Env E(ctx, &cur[C_N], np0);
  typename T::bn_t I, z, one, prev, r;
  cgbn_load(E.env, one, mp(&cur[C_ONE]));
  cgbn_load(E.env, I, mp(&INVTOT[c]));
  const size_t start = c * chunk, end = (c + 1) * chunk < count ? (c + 1) * chunk : count;
  for (size_t i = end; i-- > start;) {
    if (i > start) cgbn_load(E.env, prev, mp(&PRE[i - 1]));
    else cgbn_set(E.env, prev, one);
    E.mul(r, I, prev);
    cgbn_store(E.env, &INV[i], r);
    cgbn_load(E.env, z, mp(&V[i]));
    if (E.is_zero(z)) cgbn_set(E.env, z, one);
    E.mul(I, I, z);
  }
}

/* out limbs (32-bit, limb-major, ld) := x = (3u + A) / (3B) with u = OS/OD,
   canonical, or 0 if OD == 0 (the neutral element). Inputs: blen values per
   curve (curve c's at c*blen) of which the first len are used; output entry
   c*blen + i. */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_extract(const cgbn_mem_t<BITS> *cur, uint32_t np0, const cgbn_mem_t<BITS> *OS,
                          const cgbn_mem_t<BITS> *OD, const cgbn_mem_t<BITS> *INV, size_t blen, size_t len,
                          size_t ncurves, uint32_t *out, size_t ld, int nl) {
  typedef K<TPI, BITS> T;
  const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (i >= blen * ncurves || i % blen >= len) return;
  cur += (i / blen) * C_COUNT;
  typename T::ctx_t ctx(cgbn_no_checks);
  typename T::Env E(ctx, &cur[C_N], np0);
  typename T::bn_t x, u, v;
  cgbn_load(E.env, v, mp(&OD[i]));
  if (E.is_zero(v)) {
    cgbn_set_ui32(E.env, x, 0);
  } else {
    cgbn_load(E.env, u, mp(&OS[i]));
    cgbn_load(E.env, v, mp(&INV[i]));
    E.mul(u, u, v);
    E.add(x, u, u);
    E.add(x, x, u);
    cgbn_load(E.env, v, mp(&cur[C_AM]));
    E.add(x, x, v);
    cgbn_load(E.env, v, mp(&cur[C_C3]));
    E.mul(x, x, v);
    E.from_mont(x, x);
  }
  const bool lead = (threadIdx.x % TPI) == 0;
  for (int j = 0; j < nl; j++) {
    uint32_t w = cgbn_extract_bits_ui32(E.env, x, 32 * j, 32);
    if (lead) out[(size_t)j * ld + i] = w;
  }
}

/* test kernel: io[0..3] = P, io[4..7] = Q (projective), io[8..10] = Q affine
   (X, Y, T); outputs io[11..14] = P + Q, io[15..18] = 2P, io[19..22] = P + Q
   with the mixed addition */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_test_ops(const cgbn_mem_t<BITS> *cur, uint32_t np0, cgbn_mem_t<BITS> *io) {
  typedef K<TPI, BITS> T;
  typename T::ctx_t ctx(cgbn_no_checks);
  typename T::Env E(ctx, &cur[C_N], np0);
  typename T::bn_t a, d;
  cgbn_load(E.env, a, mp(&cur[C_A]));
  cgbn_load(E.env, d, mp(&cur[C_D]));
  typename T::Pt P, Q, Qa, R;
  load_pt(E, P, &io[0], &io[1], &io[2], &io[3], 0);
  load_pt(E, Q, &io[4], &io[5], &io[6], &io[7], 0);
  cgbn_load(E.env, Qa.X, &io[8]);
  cgbn_load(E.env, Qa.Y, &io[9]);
  cgbn_load(E.env, Qa.T, &io[10]);
  E.template padd<false>(R, P, Q, a, d);
  store_pt(E, R, &io[11], &io[12], &io[13], &io[14], 0);
  E.pdbl(R, P, a);
  store_pt(E, R, &io[15], &io[16], &io[17], &io[18], 0);
  R = P;
  E.template padd<true>(R, R, Qa, a, d);
  store_pt(E, R, &io[19], &io[20], &io[21], &io[22], 0);
}

/* --------------------------------------------------------------------- host */

template <uint32_t BITS>
void to_mem(cgbn_mem_t<BITS> &m, const mpz_t v) {
  for (auto &w : m._limbs) w = 0;
  size_t cnt;
  mpz_export(m._limbs, &cnt, -1, 4, 0, 0, v);
}
template <uint32_t BITS>
void from_mem(mpz_t v, const cgbn_mem_t<BITS> &m) {
  mpz_import(v, BITS / 32, -1, 4, 0, 0, m._limbs);
}

uint32_t np0_of(const mpz_t n) {
  uint32_t n0 = (uint32_t)mpz_get_ui(n), x = 1;
  for (int i = 0; i < 5; i++) x *= 2 - n0 * x;
  return 0u - x;
}

/* transient buffers come from the stream-ordered pool (cheap once it has grown) */
template <class T>
T *dmalloc(size_t n, cudaStream_t s) {
  T *p;
  CUDA_CHECK(cudaMallocAsync(&p, sizeof(T) * (n ? n : 1), s));
  return p;
}
void dfree(void *p, cudaStream_t s) {
  if (p) CUDA_CHECK(cudaFreeAsync(p, s));
}

unsigned grid_for(size_t instances, uint32_t tpi) {
  size_t threads = instances * tpi;
  return (unsigned)((threads + BLOCK - 1) / BLOCK);
}

/* Host side of the Edwards model of a curve. Returns ROOTS_FACTOR (factor
   set) if 3B or u+1 is not invertible. */
struct EdwardsModel {
  Z n, a, d, A, c3;          // a, d: Edwards; A: Montgomery; c3 = 1/(3B)
  Z X, Y, Zc, T;             // the point (u, 1)
  int init(const MontgomeryCurve &C, mpz_t factor) {
    mpz_set(n, C.n);
    Z B, t, inv;
    mpz_add(B, C.u, C.A); mpz_mul(B, B, C.u); mpz_add_ui(B, B, 1); mpz_mul(B, B, C.u); mpz_mod(B, B, n);  // B = u^3 + A u^2 + u
    mpz_mul_ui(t, B, 3);
    if (!mpz_invert(c3, t, n)) { mpz_gcd(factor, t, n); return ROOTS_FACTOR; }
    mpz_mul_ui(inv, c3, 3); mpz_mod(inv, inv, n);                          // 1/B
    mpz_add_ui(a, C.A, 2); mpz_mul(a, a, inv); mpz_mod(a, a, n);
    mpz_sub_ui(d, C.A, 2); mpz_mul(d, d, inv); mpz_mod(d, d, n);
    mpz_mod(A, C.A, n);
    mpz_add_ui(Zc, C.u, 1); mpz_mod(Zc, Zc, n);
    mpz_gcd(t, Zc, n);
    if (mpz_cmp_ui(t, 1) != 0) { mpz_set(factor, t); return ROOTS_FACTOR; }
    mpz_mul(X, C.u, Zc); mpz_mod(X, X, n);
    mpz_sub_ui(Y, C.u, 1); mpz_mod(Y, Y, n);
    mpz_mul(T, C.u, Y); mpz_mod(T, T, n);
    return ROOTS_OK;
  }
};

/* the constants for the device, Montgomery form (R = 2^BITS) except n;
   C_COUNT values per curve */
template <uint32_t BITS>
cgbn_mem_t<BITS> *upload_consts(const EdwardsModel *Ms, size_t count, cudaStream_t s) {
  Z R, t;
  mpz_set_ui(R, 1);
  mpz_mul_2exp(R, R, BITS);
  mpz_mod(R, R, Ms[0].n);
  std::vector<cgbn_mem_t<BITS>> h(C_COUNT * count);
  for (size_t c = 0; c < count; c++) {
    const EdwardsModel &M = Ms[c];
    cgbn_mem_t<BITS> *hc = h.data() + c * C_COUNT;
    to_mem(hc[C_N], M.n.get());
    const Z *vals[C_COUNT] = {nullptr, &M.a, &M.d, &M.A, &M.c3, nullptr, &M.X, &M.Y, &M.Zc, &M.T};
    for (int i = 1; i < C_COUNT; i++) {
      if (vals[i]) mpz_mul(t, vals[i]->get(), R);
      else mpz_set(t, R);  // one
      mpz_mod(t, t, M.n);
      to_mem(hc[i], t.get());
    }
  }
  cgbn_mem_t<BITS> *d = dmalloc<cgbn_mem_t<BITS>>(h.size(), s);
  CUDA_CHECK(cudaMemcpyAsync(d, h.data(), sizeof(cgbn_mem_t<BITS>) * h.size(), cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return d;
}

/* INV[i] = 1/V[i] (Montgomery form in and out; V[i] == 0 is treated as 1).
   Levels of chunked prefix products on the GPU until at most HOST_INV values
   remain, which are inverted on the host. The first chunk size adapts to the
   count (long chunks for throughput when there is enough parallelism, short
   ones otherwise); later levels use short chunks to keep the sequential
   chains short. Returns ROOTS_FACTOR (gcd in factor) if some V[i] is not
   invertible. */
template <uint32_t TPI, uint32_t BITS>
int batch_inverse(const cgbn_mem_t<BITS> *cur, uint32_t np0, const mpz_t n, const cgbn_mem_t<BITS> *V, size_t count,
                  cgbn_mem_t<BITS> *INV, mpz_t factor, cudaStream_t s) {
  typedef cgbn_mem_t<BITS> mem_t;
  constexpr size_t HOST_INV = 32, TARGET_INSTANCES = 16384;
  struct Level { const mem_t *V; mem_t *INV; size_t count, chunk; mem_t *PRE, *TOT; };
  std::vector<Level> lv;
  std::vector<void *> tmp;
  const mem_t *v = V;
  mem_t *inv = INV;
  size_t cnt = count;
  while (cnt > HOST_INV) {
    const size_t chunk = lv.empty() ? std::min<size_t>(64, std::max<size_t>(8, cnt / TARGET_INSTANCES)) : 8;
    const size_t nch = (cnt + chunk - 1) / chunk;
    Level L{v, inv, cnt, chunk, dmalloc<mem_t>(cnt, s), dmalloc<mem_t>(nch, s)};
    tmp.push_back(L.PRE);
    tmp.push_back(L.TOT);
    k_chunk_prefix<TPI, BITS><<<grid_for(nch, TPI), BLOCK, 0, s>>>(cur, np0, L.V, cnt, chunk, L.PRE, L.TOT);
    CUDA_CHECK(cudaGetLastError());
    lv.push_back(L);
    v = L.TOT;
    inv = dmalloc<mem_t>(nch, s);  // the inverses of this level's totals
    tmp.push_back(inv);
    cnt = nch;
  }
  std::vector<mem_t> ht(cnt);
  CUDA_CHECK(cudaMemcpyAsync(ht.data(), v, sizeof(mem_t) * cnt, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  int ret = ROOTS_OK;
  Z R2, x, xi, one;
  mpz_set_ui(R2, 1);
  mpz_mul_2exp(R2, R2, 2 * BITS);
  mpz_mod(R2, R2, n);
  for (size_t i = 0; i < cnt; i++) {
    from_mem(x, ht[i]);  // Montgomery form T*R
    if (mpz_sgn(x.get()) == 0) {  // zero counts as one: 1 -> R
      mpz_set_ui(one, 1);
      mpz_mul_2exp(one, one, BITS);
      mpz_mod(x, one, n);
    }
    if (!mpz_invert(xi, x, n)) {
      mpz_gcd(factor, x, n);
      ret = ROOTS_FACTOR;
      break;
    }
    mpz_mul(xi, xi, R2);  // (T R)^-1 R^2 = T^-1 R
    mpz_mod(xi, xi, n);
    to_mem(ht[i], xi.get());
  }
  if (ret == ROOTS_OK) {
    CUDA_CHECK(cudaMemcpyAsync(inv, ht.data(), sizeof(mem_t) * cnt, cudaMemcpyHostToDevice, s));
    for (size_t l = lv.size(); l-- > 0;) {
      const Level &L = lv[l];
      const size_t nch = (L.count + L.chunk - 1) / L.chunk;
      const mem_t *invtot = l + 1 < lv.size() ? lv[l + 1].INV : inv;
      k_chunk_back<TPI, BITS><<<grid_for(nch, TPI), BLOCK, 0, s>>>(cur, np0, L.V, L.count, L.chunk, L.PRE, invtot,
                                                                  L.INV);
      CUDA_CHECK(cudaGetLastError());
    }
  }
  for (void *p : tmp) dfree(p, s);
  CUDA_CHECK(cudaStreamSynchronize(s));
  return ret;
}

}  // namespace

/* -------------------------------------------------------- RootGenerator */

struct RootGenerator::Impl {
  virtual ~Impl() {}
  virtual int start(const RootPlanGPU &plan) = 0;
  virtual int next_block(const RnsContext &ctx, uint32_t *out, mpz_t factor) = 0;
};

namespace {

template <uint32_t TPI, uint32_t BITS>
struct ImplT : RootGenerator::Impl {
  typedef cgbn_mem_t<BITS> mem_t;
  Z n;
  uint32_t np0 = 0;
  cudaStream_t s = 0;
  size_t nc = 1;    // curves
  size_t ntab = 0;  // table entries per curve
  mem_t *cur = nullptr, *TX = nullptr, *TY = nullptr, *TT = nullptr;
  size_t maxbits = 0;

  ~ImplT() override {
    free_plan();
    for (void *p : std::initializer_list<void *>{cur, TX, TY, TT}) dfree(p, s);
  }

  int init(const MontgomeryCurve *C, size_t count, size_t maxbits_, mpz_t factor, size_t *failed, cudaStream_t s_) {
    s = s_;
    nc = count;
    maxbits = std::max<size_t>(maxbits_, 1);
    mpz_set(n, C[0].n);
    np0 = np0_of(n);
    std::vector<EdwardsModel> M(count);
    for (size_t c = 0; c < count; c++)
      if (M[c].init(C[c], factor) != ROOTS_OK) {
        *failed = c;
        return ROOTS_FACTOR;
      }
    cur = upload_consts<BITS>(M.data(), count, s);
    /* fixed-base tables of j 2^(WIN t) P, normalised to affine */
    const int W = (int)((maxbits + WIN - 1) / WIN);
    ntab = (size_t)W * WTAB;
    const size_t all = ntab * nc;
    TX = dmalloc<mem_t>(all, s); TY = dmalloc<mem_t>(all, s); TT = dmalloc<mem_t>(all, s);
    mem_t *TZ = dmalloc<mem_t>(all, s), *TI = dmalloc<mem_t>(all, s);
    k_pow2_chain<chain_tpi(BITS), BITS><<<(unsigned)nc, chain_tpi(BITS), 0, s>>>(cur, np0, W, ntab, TX, TY, TZ, TT);
    for (int h = 1; h < WTAB; h *= 2) {
      const int jn = std::min(2 * h, WTAB) - h;
      k_table_round<TPI, BITS><<<grid_for((size_t)W * jn * nc, TPI), BLOCK, 0, s>>>(cur, np0, W, h, ntab, nc, TX, TY,
                                                                                   TZ, TT);
      CUDA_CHECK(cudaGetLastError());
    }
    int ret = batch_inverse<TPI, BITS>(cur, np0, n, TZ, all, TI, factor, s);
    if (ret == ROOTS_OK) {
      k_affine<TPI, BITS><<<grid_for(all, TPI), BLOCK, 0, s>>>(cur, np0, all, TX, TY, TT, TI);
      CUDA_CHECK(cudaGetLastError());
    }
    dfree(TZ, s);
    dfree(TI, s);
    return ret;
  }

  /* the plan being run */
  const RootPlanGPU *pl = nullptr;
  size_t next = 0, step = 0;
  mem_t *FX = nullptr, *FY = nullptr, *FZ = nullptr, *FT = nullptr, *GX = nullptr, *GY = nullptr, *GZ = nullptr,
        *GT = nullptr, *OS = nullptr, *OD = nullptr, *INV = nullptr;

  void free_plan() {
    for (void *p : std::initializer_list<void *>{FX, FY, FZ, FT, GX, GY, GZ, GT, OS, OD, INV}) dfree(p, s);
    FX = FY = FZ = FT = GX = GY = GZ = GT = OS = OD = INV = nullptr;
    pl = nullptr;
  }

  int start(const RootPlanGPU &plan) override {
    free_plan();
    next = step = 0;
    if (plan.count == 0) return ROOTS_OK;
    if (plan.scalar_bits > maxbits) {
      fprintf(stderr, "RootGenerator: plan needs %zu-bit scalars, table has %zu\n", plan.scalar_bits, maxbits);
      return ROOTS_ERROR;
    }
    pl = &plan;
    const size_t nfd = plan.nprog * (plan.S + 1), all = nfd * nc;
    FX = dmalloc<mem_t>(all, s); FY = dmalloc<mem_t>(all, s); FZ = dmalloc<mem_t>(all, s); FT = dmalloc<mem_t>(all, s);
    k_scalar_mul_tab<TPI, BITS><<<grid_for(all, TPI), BLOCK, 0, s>>>(cur, np0, plan.scal, plan.swords, plan.nbits,
                                                                    nfd, nc, ntab, TX, TY, TT, FX, FY, FZ, FT);
    CUDA_CHECK(cudaGetLastError());
    /* the steps alternate between two copies of the tables */
    GX = dmalloc<mem_t>(all, s); GY = dmalloc<mem_t>(all, s); GZ = dmalloc<mem_t>(all, s); GT = dmalloc<mem_t>(all, s);
    for (auto pr : {std::make_pair(FX, GX), std::make_pair(FY, GY), std::make_pair(FZ, GZ), std::make_pair(FT, GT)})
      CUDA_CHECK(cudaMemcpyAsync(pr.second, pr.first, sizeof(mem_t) * all, cudaMemcpyDeviceToDevice, s));
    OS = dmalloc<mem_t>(plan.nring * plan.blen * nc, s);
    OD = dmalloc<mem_t>(plan.nring * plan.blen * nc, s);
    INV = dmalloc<mem_t>(plan.blen * nc, s);
    return ROOTS_OK;
  }

  int next_block(const RnsContext &ctx, uint32_t *out, mpz_t factor) override {
    if (!pl || next >= pl->nblocks) return ROOTS_ERROR;
    const RootPlanGPU &P = *pl;
    const size_t b = next++;
    const unsigned nblk = grid_for(P.nprog * P.S * nc, TPI);
    for (; step <= P.last_step[b] && step < P.steps; step++) {
      if (step & 1)
        k_step<TPI, BITS><<<nblk, BLOCK, 0, s>>>(cur, np0, GX, GY, GZ, GT, FX, FY, FZ, FT, P.S, P.nprog, nc, step,
                                                 P.slot, P.base, P.inc, P.count, P.blen, P.nring, OS, OD);
      else
        k_step<TPI, BITS><<<nblk, BLOCK, 0, s>>>(cur, np0, FX, FY, FZ, FT, GX, GY, GZ, GT, P.S, P.nprog, nc, step,
                                                 P.slot, P.base, P.inc, P.count, P.blen, P.nring, OS, OD);
      CUDA_CHECK(cudaGetLastError());
    }
    /* x = (3 OS/OD + A) / (3B) for this block of every curve; in a short last
       block the unused values are zeroed (zeros are skipped by the inversion) */
    const size_t len = std::min(P.blen, P.count - b * P.blen), off = (b % P.nring) * P.blen * nc;
    if (len < P.blen)
      CUDA_CHECK(cudaMemset2DAsync(OD + off + len, P.blen * sizeof(mem_t), 0, (P.blen - len) * sizeof(mem_t), nc, s));
    int ret = batch_inverse<TPI, BITS>(cur, np0, n, OD + off, P.blen * nc, INV, factor, s);
    if (ret == ROOTS_OK) {
      k_extract<TPI, BITS><<<grid_for(P.blen * nc, TPI), BLOCK, 0, s>>>(cur, np0, OS + off, OD + off, INV, P.blen,
                                                                       len, nc, out, P.blen * nc, ctx.nlimbs());
      CUDA_CHECK(cudaGetLastError());
    }
    if (next == P.nblocks || ret != ROOTS_OK) free_plan();
    return ret;
  }
};

template <uint32_t TPI, uint32_t BITS>
void make_impl(RootGenerator::Impl **impl, int *ret, const MontgomeryCurve *C, size_t count, size_t maxbits,
               mpz_t factor, size_t *failed, cudaStream_t s) {
  ImplT<TPI, BITS> *p = new ImplT<TPI, BITS>();
  *ret = p->init(C, count, maxbits, factor, failed, s);
  *impl = p;
}

/* ------------------------------------------------------------------ tests */

/* Weierstrass affine (GMP-ECM's model) <-> Edwards projective, on the host */
struct HostMaps {
  Z n, A, B, iB, A3;  // A3 = A/3
  void init(const MontgomeryCurve &C) {
    mpz_set(n, C.n);
    mpz_set(A, C.A);
    mpz_add(B, C.u, C.A); mpz_mul(B, B, C.u); mpz_add_ui(B, B, 1); mpz_mul(B, B, C.u); mpz_mod(B, B, n);
    mpz_invert(iB, B, n);
    Z three;
    mpz_set_ui(three, 3);
    mpz_invert(A3, three, n);
    mpz_mul(A3, A3, A); mpz_mod(A3, A3, n);
  }
  /* returns false if not representable */
  bool to_ed(const AffinePoint &P, Z *out /* X, Y, Z, T */) {
    if (P.inf) {
      mpz_set_ui(out[0], 0); mpz_set_ui(out[1], 1); mpz_set_ui(out[2], 1); mpz_set_ui(out[3], 0);
      return true;
    }
    Z u, v, t;
    mpz_mul(u, P.x, B); mpz_sub(u, u, A3); mpz_mod(u, u, n);  // u = B x - A/3
    mpz_mul(v, P.y, B); mpz_mod(v, v, n);
    if (mpz_sgn(v.get()) == 0) {
      if (mpz_sgn(u.get()) != 0) return false;  // other 2-torsion: Edwards infinity
      mpz_set_ui(out[0], 0); mpz_sub_ui(out[1], n, 1); mpz_set_ui(out[2], 1); mpz_set_ui(out[3], 0);
      return true;
    }
    /* x = u/v, y = (u-1)/(u+1): (u(u+1) : v(u-1) : v(u+1) : u(u-1)) */
    Z up, um;
    mpz_add_ui(up, u, 1); mpz_mod(up, up, n);
    if (mpz_sgn(up.get()) == 0) return false;
    mpz_sub_ui(um, u, 1); mpz_mod(um, um, n);
    mpz_mul(out[0], u, up); mpz_mod(out[0], out[0], n);
    mpz_mul(out[1], v, um); mpz_mod(out[1], out[1], n);
    mpz_mul(out[2], v, up); mpz_mod(out[2], out[2], n);
    mpz_mul(out[3], u, um); mpz_mod(out[3], out[3], n);
    return true;
  }
  bool from_ed(const Z *in, AffinePoint &P) {
    Z iz, x, y, t;
    if (!mpz_invert(iz, in[2], n)) return false;
    mpz_mul(x, in[0], iz); mpz_mod(x, x, n);
    mpz_mul(y, in[1], iz); mpz_mod(y, y, n);
    mpz_mul(t, in[3], iz); mpz_mod(t, t, n);
    Z xy;
    mpz_mul(xy, x, y); mpz_mod(xy, xy, n);
    if (mpz_cmp(xy, t) != 0) return false;  // inconsistent T
    P.inf = false;
    if (mpz_sgn(x.get()) == 0) {
      if (mpz_cmp_ui(y, 1) == 0) { P.inf = true; return true; }
      /* (0, -1): Montgomery (0, 0) */
      mpz_set(P.x, A3); mpz_mul(P.x, P.x, iB); mpz_mod(P.x, P.x, n);
      mpz_set_ui(P.y, 0);
      return true;
    }
    Z u, v, den;
    mpz_ui_sub(den, 1, y); mpz_mod(den, den, n);
    if (!mpz_invert(den, den, n)) return false;
    mpz_add_ui(u, y, 1); mpz_mul(u, u, den); mpz_mod(u, u, n);  // u = (1+y)/(1-y)
    if (!mpz_invert(v, x, n)) return false;
    mpz_mul(v, v, u); mpz_mod(v, v, n);                          // v = u/x
    mpz_add(P.x, u, A3); mpz_mul(P.x, P.x, iB); mpz_mod(P.x, P.x, n);
    mpz_mul(P.y, v, iB); mpz_mod(P.y, P.y, n);
    return true;
  }
};

template <uint32_t TPI, uint32_t BITS>
void run_test_ops(bool *ok, const MontgomeryCurve &C, const AffinePoint &P, const AffinePoint &Q, AffinePoint &R,
                  AffinePoint &D, AffinePoint &Mx) {
  typedef cgbn_mem_t<BITS> mem_t;
  *ok = false;
  Z fac;
  EdwardsModel M;
  if (M.init(C, fac) != ROOTS_OK) return;
  HostMaps H;
  H.init(C);
  Z pe[4], qe[4], qa[3], RR, t;
  if (!H.to_ed(P, pe) || !H.to_ed(Q, qe)) return;
  Z iz;
  if (!mpz_invert(iz, qe[2], C.n)) return;
  for (int i : {0, 1, 3}) { mpz_mul(qa[i == 3 ? 2 : i], qe[i], iz); mpz_mod(qa[i == 3 ? 2 : i], qa[i == 3 ? 2 : i], C.n); }
  mpz_set_ui(RR, 1); mpz_mul_2exp(RR, RR, BITS); mpz_mod(RR, RR, C.n);
  std::vector<mem_t> h(23);
  auto put = [&](int i, const Z &v) { mpz_mul(t, v, RR); mpz_mod(t, t, C.n); to_mem(h[i], t.get()); };
  for (int i = 0; i < 4; i++) { put(i, pe[i]); put(4 + i, qe[i]); }
  for (int i = 0; i < 3; i++) put(8 + i, qa[i]);
  mem_t *cur = upload_consts<BITS>(&M, 1, 0);
  mem_t *d = dmalloc<mem_t>(23, 0);
  CUDA_CHECK(cudaMemcpy(d, h.data(), sizeof(mem_t) * 23, cudaMemcpyHostToDevice));
  k_test_ops<TPI, BITS><<<1, TPI>>>(cur, np0_of(C.n), d);
  CUDA_CHECK(cudaMemcpy(h.data(), d, sizeof(mem_t) * 23, cudaMemcpyDeviceToHost));
  dfree(d, 0);
  dfree(cur, 0);
  Z iR;
  mpz_invert(iR, RR, C.n);
  auto get = [&](int o, AffinePoint &out) {
    Z c[4];
    for (int i = 0; i < 4; i++) { from_mem(c[i], h[o + i]); mpz_mul(c[i], c[i], iR); mpz_mod(c[i], c[i], C.n); }
    return H.from_ed(c, out);
  };
  *ok = get(11, R) && get(15, D) && get(19, Mx);
}

template <uint32_t TPI, uint32_t BITS>
void step_capacity(size_t *out) {
  int dev = 0, sms = 0, blocks = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, k_step<TPI, BITS>, BLOCK, 0));
  *out = (size_t)sms * std::max(blocks, 1) * (BLOCK / TPI);
}

}  // namespace

#include "gpu/cgbn_sizes.h"

size_t root_step_capacity(size_t nbits) {
  size_t c = 0;
  GPUECM2_CGBN_DISPATCH(nbits, step_capacity, &c);
  return c;
}

RootGenerator::~RootGenerator() { delete impl_; }

int RootGenerator::init(const MontgomeryCurve *C, size_t count, size_t maxbits, mpz_t factor, cudaStream_t s) {
  delete impl_;
  impl_ = nullptr;
  count_ = count;
  failed_ = SIZE_MAX;
  int ret = ROOTS_ERROR;
  if (count == 0) return ret;
  GPUECM2_CGBN_DISPATCH(mpz_sizeinbase(C[0].n.get(), 2), make_impl, &impl_, &ret, C, count, maxbits, factor, &failed_,
                        s);
  return ret;
}

int RootGenerator::start(const RootPlanGPU &plan) {
  return impl_ ? impl_->start(plan) : ROOTS_ERROR;
}

int RootGenerator::next_block(const RnsContext &ctx, uint32_t *out, mpz_t factor) {
  return impl_ ? impl_->next_block(ctx, out, factor) : ROOTS_ERROR;
}

int RootGenerator::compute(const RnsContext &ctx, const ProgressionPlan &plan, uint32_t *out, mpz_t factor) {
  if (plan.count == 0) return ROOTS_OK;
  if (count_ != 1) return ROOTS_ERROR;
  RootPlanGPU gp(plan, plan.count);
  int r = start(gp);
  if (r == ROOTS_OK) r = next_block(ctx, out, factor);
  CUDA_CHECK(cudaDeviceSynchronize());  // gp is freed on return
  return r;
}

RootPlanGPU::RootPlanGPU(const ProgressionPlan &pl, size_t block_len) {
  count = pl.count;
  if (count == 0) return;
  S = pl.S;
  nprog = pl.nprog;
  steps = pl.steps;
  blen = std::max<size_t>(1, std::min(block_len, count));
  nblocks = (count + blen - 1) / blen;
  scalar_bits = plan_scalar_bits(pl);
  /* which steps feed which blocks: block b can be finished after step
     last_step[b]; the ring must hold every block that receives outputs
     while some earlier block is still unfinished */
  last_step.assign(nblocks, 0);
  nring = 1;
  if (!pl.slot.empty()) {
    std::vector<size_t> max_block(steps, 0);
    for (size_t n = 0; n < steps; n++)
      for (size_t q = 0; q < nprog; q++) {
        const int32_t sl = pl.slot[n * nprog + q];
        if (sl < 0) continue;
        const size_t b = (size_t)sl / blen;
        last_step[b] = std::max(last_step[b], n);
        max_block[n] = std::max(max_block[n], b);
      }
    for (size_t b = 1; b < nblocks; b++) last_step[b] = std::max(last_step[b], last_step[b - 1]);
    size_t hi = 0, n = 0;
    for (size_t b = 0; b < nblocks; b++) {
      for (; n <= last_step[b] && n < steps; n++) hi = std::max(hi, max_block[n]);
      nring = std::max(nring, hi - b + 1);
    }
  } else {
    /* affine form: progression q outputs index base[q] + n*inc for n < nout[q] */
    inc = pl.inc;
    std::vector<size_t> nout(nprog, 0);
    for (size_t q = 0; q < nprog; q++)
      if ((uint64_t)pl.base[q] < count) nout[q] = (count - (size_t)pl.base[q] + (size_t)inc - 1) / (size_t)inc;
    for (size_t b = 0; b < nblocks; b++) {
      /* the last step with an output in blocks 0..b, and the furthest block
         reached by the steps up to it */
      const size_t top = std::min(count, (b + 1) * blen) - 1;
      size_t last = 0;
      for (size_t q = 0; q < nprog; q++)
        if (nout[q] && (size_t)pl.base[q] <= top) last = std::max(last, (top - (size_t)pl.base[q]) / (size_t)inc);
      last_step[b] = last;
      size_t hi = 0;
      for (size_t q = 0; q < nprog; q++)
        if (nout[q]) hi = std::max(hi, ((size_t)pl.base[q] + std::min(last, nout[q] - 1) * (size_t)inc) / blen);
      nring = std::max(nring, hi - std::min(hi, b) + 1);
    }
  }
  nring = std::min(nring, nblocks);
  /* scalars: magnitudes and signed bit lengths */
  const size_t nfd = nprog * (S + 1);
  swords = (int)((scalar_bits + 31) / 32);
  std::vector<uint32_t> hs((size_t)swords * nfd, 0);
  std::vector<int32_t> hb(nfd);
  for (size_t i = 0; i < nfd; i++) {
    const mpz_srcptr c = pl.coeffs[i].get();
    size_t cnt;
    mpz_export(&hs[i * swords], &cnt, -1, 4, 0, 0, c);
    const int bits = mpz_sgn(c) ? (int)mpz_sizeinbase(c, 2) : 0;
    hb[i] = mpz_sgn(c) < 0 ? -bits : bits;
  }
  CUDA_CHECK(cudaMalloc(&scal, hs.size() * 4));
  CUDA_CHECK(cudaMalloc(&nbits, hb.size() * 4));
  CUDA_CHECK(cudaMemcpy(scal, hs.data(), hs.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(nbits, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
  if (!pl.slot.empty()) {
    CUDA_CHECK(cudaMalloc(&slot, pl.slot.size() * 4));
    CUDA_CHECK(cudaMemcpy(slot, pl.slot.data(), pl.slot.size() * 4, cudaMemcpyHostToDevice));
  } else {
    CUDA_CHECK(cudaMalloc(&base, pl.base.size() * 8));
    CUDA_CHECK(cudaMemcpy(base, pl.base.data(), pl.base.size() * 8, cudaMemcpyHostToDevice));
  }
}

RootPlanGPU::~RootPlanGPU() {
  for (void *p : std::initializer_list<void *>{scal, nbits, (void *)slot, (void *)base})
    if (p) cudaFree(p);
}

size_t plan_scalar_bits(const ProgressionPlan &pl) {
  size_t maxbits = 1;
  for (const Z &c : pl.coeffs) maxbits = std::max(maxbits, mpz_sizeinbase(c.get(), 2));
  return maxbits;
}

int compute_roots(const RnsContext &ctx, const MontgomeryCurve &C, const ProgressionPlan &plan, uint32_t *out,
                  mpz_t factor, cudaStream_t s) {
  if (plan.count == 0) return ROOTS_OK;
  RootGenerator g;
  int r = g.init(C, plan_scalar_bits(plan), factor, s);
  return r != ROOTS_OK ? r : g.compute(ctx, plan, out, factor);
}

bool ec_test_ops(const MontgomeryCurve &C, const AffinePoint &P, const AffinePoint &Q, AffinePoint &R,
                 AffinePoint &D, AffinePoint &M) {
  bool ok = false;
  GPUECM2_CGBN_DISPATCH(mpz_sizeinbase(C.n.get(), 2), run_test_ops, &ok, C, P, Q, R, D, M);
  return ok;
}
