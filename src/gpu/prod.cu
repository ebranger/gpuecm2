/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "gpu/prod.h"

#include <vector>

#include "gpu/cuda_check.h"
#include "gpu/cgbn_sizes.h"
#include "gpu/ecc.cuh"
#include "host/mpz_class_lite.h"

namespace {

constexpr int BLOCK = 256;
constexpr size_t CHUNK = 64;

/* limb-major 32-bit limbs -> one cgbn_mem_t per value */
template <uint32_t BITS>
__global__ void k_to_mem(const uint32_t *__restrict__ V, size_t ld, size_t count, int nl,
                         cgbn_mem_t<BITS> *__restrict__ M) {
  const size_t total = count * (BITS / 32);
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < total; e += (size_t)gridDim.x * blockDim.x) {
    const size_t i = e / (BITS / 32);
    const int j = (int)(e % (BITS / 32));
    M[i]._limbs[j] = j < nl ? V[(size_t)j * ld + i] : 0;
  }
}

/* TOT[c] = product of chunk c (Montgomery products of plain residues: the
   result carries a power of R^-1, irrelevant for the gcd). V holds groups of
   len values; each group is cut into nch chunks of CHUNK values. */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_chunk_prod(const cgbn_mem_t<BITS> *nm, uint32_t np0, const cgbn_mem_t<BITS> *V, size_t len,
                             size_t nch, size_t total, cgbn_mem_t<BITS> *TOT) {
  typedef EcEnv<TPI, BITS> Env;
  const size_t c = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (c >= total) return;
  typename Env::context_t ctx(cgbn_no_checks);
  Env E(ctx, nm, np0);
  typename Env::bn_t acc, v;
  const size_t grp = c / nch, lc = c - grp * nch;
  const size_t start = grp * len + lc * CHUNK, end = grp * len + ((lc + 1) * CHUNK < len ? (lc + 1) * CHUNK : len);
  cgbn_load(E.env, acc, const_cast<cgbn_mem_t<BITS> *>(&V[start]));
  for (size_t i = start + 1; i < end; i++) {
    cgbn_load(E.env, v, const_cast<cgbn_mem_t<BITS> *>(&V[i]));
    E.mul(acc, acc, v);
  }
  cgbn_store(E.env, &TOT[c], acc);
}

unsigned grid_for(size_t instances, uint32_t tpi) { return (unsigned)((instances * tpi + BLOCK - 1) / BLOCK); }

template <uint32_t TPI, uint32_t BITS>
void run(const RnsContext &ctx, const mpz_t n, const uint32_t *V, size_t count, size_t batch, mpz_ptr *g,
         cudaStream_t s) {
  typedef cgbn_mem_t<BITS> mem_t;
  mem_t hn, *dn, *M, *T1, *T2;
  for (auto &w : hn._limbs) w = 0;
  size_t cnt;
  mpz_export(hn._limbs, &cnt, -1, 4, 0, 0, n);
  uint32_t x = 1, n0 = hn._limbs[0];
  for (int i = 0; i < 5; i++) x *= 2 - n0 * x;
  const uint32_t np0 = 0u - x;
  /* per problem: count values -> c1 chunk products -> c2 chunk products */
  const size_t B = batch, c1 = (count + CHUNK - 1) / CHUNK, c2 = (c1 + CHUNK - 1) / CHUNK;
  CUDA_CHECK(cudaMallocAsync(&dn, sizeof(mem_t), s));
  CUDA_CHECK(cudaMallocAsync(&M, sizeof(mem_t) * B * count, s));
  CUDA_CHECK(cudaMallocAsync(&T1, sizeof(mem_t) * B * c1, s));
  CUDA_CHECK(cudaMallocAsync(&T2, sizeof(mem_t) * B * c2, s));
  CUDA_CHECK(cudaMemcpyAsync(dn, &hn, sizeof(mem_t), cudaMemcpyHostToDevice, s));
  size_t total = B * count * (BITS / 32), b = (total + 255) / 256;
  k_to_mem<BITS><<<(unsigned)(b < 65535 ? b : 65535), 256, 0, s>>>(V, B * count, B * count, ctx.nlimbs(), M);
  k_chunk_prod<TPI, BITS><<<grid_for(B * c1, TPI), BLOCK, 0, s>>>(dn, np0, M, count, c1, B * c1, T1);
  k_chunk_prod<TPI, BITS><<<grid_for(B * c2, TPI), BLOCK, 0, s>>>(dn, np0, T1, c1, c2, B * c2, T2);
  CUDA_CHECK(cudaGetLastError());
  std::vector<mem_t> h(B * c2);
  CUDA_CHECK(cudaMemcpyAsync(h.data(), T2, sizeof(mem_t) * B * c2, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  Z p, v;
  for (size_t c = 0; c < B; c++) {
    mpz_set_ui(p, 1);
    for (size_t i = 0; i < c2; i++) {
      mpz_import(v, BITS / 32, -1, 4, 0, 0, h[c * c2 + i]._limbs);
      mpz_mul(p, p, v);
      mpz_mod(p, p, n);
    }
    mpz_gcd(g[c], p, n);
  }
  for (void *q : {(void *)dn, (void *)M, (void *)T1, (void *)T2}) cudaFreeAsync(q, s);
}

/* chunk Horner: OUT[c] = sum_{i in chunk c} M[i] x^(i - c*CHUNK), canonical
   (x given in Montgomery form, so mont_mul(v, xm) = v x) */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_horner(const cgbn_mem_t<BITS> *nm, uint32_t np0, const cgbn_mem_t<BITS> *M, size_t len,
                         const cgbn_mem_t<BITS> *xm, cgbn_mem_t<BITS> *OUT) {
  typedef EcEnv<TPI, BITS> Env;
  const size_t c = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (c * CHUNK >= len) return;
  typename Env::context_t ctx(cgbn_no_checks);
  Env E(ctx, nm, np0);
  typename Env::bn_t v, x, m;
  cgbn_load(E.env, x, const_cast<cgbn_mem_t<BITS> *>(xm));
  cgbn_set_ui32(E.env, v, 0);
  const size_t start = c * CHUNK, end = (c + 1) * CHUNK < len ? (c + 1) * CHUNK : len;
  for (size_t i = end; i-- > start;) {
    E.mul(v, v, x);
    cgbn_load(E.env, m, const_cast<cgbn_mem_t<BITS> *>(&M[i]));
    E.add(v, v, m);
  }
  cgbn_store(E.env, &OUT[c], v);
}

/* chunk products: OUT[t*nch + c] = prod_{i in chunk c} (M[i] + R[t]) R^-len(c)
   (Montgomery products of canonical values) */
template <uint32_t TPI, uint32_t BITS>
__global__ void k_rootprod(const cgbn_mem_t<BITS> *nm, uint32_t np0, const cgbn_mem_t<BITS> *M, size_t len,
                           const cgbn_mem_t<BITS> *R, int nr, cgbn_mem_t<BITS> *OUT) {
  typedef EcEnv<TPI, BITS> Env;
  const size_t nch = (len + CHUNK - 1) / CHUNK;
  const size_t inst = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (inst >= nch * nr) return;
  const size_t c = inst % nch, t = inst / nch;
  typename Env::context_t ctx(cgbn_no_checks);
  Env E(ctx, nm, np0);
  typename Env::bn_t acc, r, m;
  cgbn_load(E.env, r, const_cast<cgbn_mem_t<BITS> *>(&R[t]));
  cgbn_set_ui32(E.env, acc, 1);
  const size_t start = c * CHUNK, end = (c + 1) * CHUNK < len ? (c + 1) * CHUNK : len;
  for (size_t i = start; i < end; i++) {
    cgbn_load(E.env, m, const_cast<cgbn_mem_t<BITS> *>(&M[i]));
    E.add(m, m, r);
    E.mul(acc, acc, m);
  }
  cgbn_store(E.env, &OUT[inst], acc);
}

template <uint32_t BITS>
void to_mem_host(cgbn_mem_t<BITS> &m, const mpz_t v) {
  for (auto &w : m._limbs) w = 0;
  size_t cnt;
  mpz_export(m._limbs, &cnt, -1, 4, 0, 0, v);
}

/* limb array -> device cgbn_mem_t array (caller frees with cudaFreeAsync) */
template <uint32_t BITS>
cgbn_mem_t<BITS> *limbs_to_mem(const RnsContext &ctx, const uint32_t *V, size_t ld, size_t len, cudaStream_t s) {
  cgbn_mem_t<BITS> *M;
  CUDA_CHECK(cudaMallocAsync(&M, sizeof(cgbn_mem_t<BITS>) * (len ? len : 1), s));
  size_t total = len * (BITS / 32), b = (total + 255) / 256;
  k_to_mem<BITS><<<(unsigned)(b < 65535 ? b : 65535), 256, 0, s>>>(V, ld, len, ctx.nlimbs(), M);
  CUDA_CHECK(cudaGetLastError());
  return M;
}

uint32_t np0_of(const mpz_t n) {
  uint32_t n0 = (uint32_t)mpz_get_ui(n), x = 1;
  for (int i = 0; i < 5; i++) x *= 2 - n0 * x;
  return 0u - x;
}

template <uint32_t TPI, uint32_t BITS>
void run_poly_value(const RnsContext &ctx, const mpz_t n, const uint32_t *coef, size_t ld, size_t len, bool monic,
                    const mpz_t x, mpz_t out, cudaStream_t s) {
  typedef cgbn_mem_t<BITS> mem_t;
  const size_t nch = (len + CHUNK - 1) / CHUNK;
  std::vector<mem_t> h(2);
  Z t, y;
  mpz_mul_2exp(t, x, BITS);
  mpz_mod(t, t, n);  // x R
  to_mem_host(h[0], n);
  to_mem_host(h[1], t.get());
  mem_t *dc, *OUT;
  CUDA_CHECK(cudaMallocAsync(&dc, sizeof(mem_t) * 2, s));
  CUDA_CHECK(cudaMallocAsync(&OUT, sizeof(mem_t) * (nch ? nch : 1), s));
  CUDA_CHECK(cudaMemcpyAsync(dc, h.data(), sizeof(mem_t) * 2, cudaMemcpyHostToDevice, s));
  mem_t *M = limbs_to_mem<BITS>(ctx, coef, ld, len, s);
  if (nch) k_horner<TPI, BITS><<<grid_for(nch, TPI), BLOCK, 0, s>>>(dc, np0_of(n), M, len, dc + 1, OUT);
  CUDA_CHECK(cudaGetLastError());
  std::vector<mem_t> ho(nch);
  CUDA_CHECK(cudaMemcpyAsync(ho.data(), OUT, sizeof(mem_t) * nch, cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  for (void *q : {(void *)dc, (void *)OUT, (void *)M}) cudaFreeAsync(q, s);
  /* combine: sum_c OUT[c] y^c with y = x^CHUNK, then + x^len if monic */
  Z v;
  mpz_powm_ui(y, x, CHUNK, n);
  mpz_set_ui(out, 0);
  for (size_t c = nch; c-- > 0;) {
    mpz_import(v, BITS / 32, -1, 4, 0, 0, ho[c]._limbs);
    mpz_mul(out, out, y);
    mpz_add(out, out, v);
    mpz_mod(out, out, n);
  }
  if (monic) {
    mpz_powm_ui(t, x, len, n);
    mpz_add(out, out, t);
    mpz_mod(out, out, n);
  }
}

template <uint32_t TPI, uint32_t BITS>
void run_roots_value(const RnsContext &ctx, const mpz_t n, const uint32_t *roots, size_t ld, size_t len,
                     const mpz_srcptr *r, int nr, mpz_ptr *out, cudaStream_t s) {
  typedef cgbn_mem_t<BITS> mem_t;
  const size_t nch = (len + CHUNK - 1) / CHUNK;
  std::vector<mem_t> h(1 + nr);
  to_mem_host(h[0], n);
  for (int t = 0; t < nr; t++) to_mem_host(h[1 + t], r[t]);
  mem_t *dc, *OUT;
  CUDA_CHECK(cudaMallocAsync(&dc, sizeof(mem_t) * h.size(), s));
  CUDA_CHECK(cudaMallocAsync(&OUT, sizeof(mem_t) * (nch * nr ? nch * nr : 1), s));
  CUDA_CHECK(cudaMemcpyAsync(dc, h.data(), sizeof(mem_t) * h.size(), cudaMemcpyHostToDevice, s));
  mem_t *M = limbs_to_mem<BITS>(ctx, roots, ld, len, s);
  if (nch) k_rootprod<TPI, BITS><<<grid_for(nch * nr, TPI), BLOCK, 0, s>>>(dc, np0_of(n), M, len, dc + 1, nr, OUT);
  CUDA_CHECK(cudaGetLastError());
  std::vector<mem_t> ho(nch * nr);
  CUDA_CHECK(cudaMemcpyAsync(ho.data(), OUT, sizeof(mem_t) * ho.size(), cudaMemcpyDeviceToHost, s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  for (void *q : {(void *)dc, (void *)OUT, (void *)M}) cudaFreeAsync(q, s);
  /* each chunk carries R^-(chunk length): multiply back by R^len overall */
  Z v, Rl;
  mpz_set_ui(Rl, 1);
  mpz_mul_2exp(Rl, Rl, BITS);
  mpz_powm_ui(Rl, Rl, len, n);
  for (int t = 0; t < nr; t++) {
    mpz_set(out[t], Rl);
    for (size_t c = 0; c < nch; c++) {
      mpz_import(v, BITS / 32, -1, 4, 0, 0, ho[t * nch + c]._limbs);
      mpz_mul(out[t], out[t], v);
      mpz_mod(out[t], out[t], n);
    }
  }
}

}  // namespace

void poly_value(const RnsContext &ctx, const mpz_t n, const uint32_t *coef, size_t ld, size_t len, bool monic,
                const mpz_t x, mpz_t out, cudaStream_t s) {
  GPUECM2_CGBN_DISPATCH(mpz_sizeinbase(n, 2), run_poly_value, ctx, n, coef, ld, len, monic, x, out, s);
}

void roots_value(const RnsContext &ctx, const mpz_t n, const uint32_t *roots, size_t ld, size_t len,
                 const mpz_srcptr *r, int nr, mpz_ptr *out, cudaStream_t s) {
  GPUECM2_CGBN_DISPATCH(mpz_sizeinbase(n, 2), run_roots_value, ctx, n, roots, ld, len, r, nr, out, s);
}

void product_gcd(const RnsContext &ctx, const mpz_t n, const uint32_t *V, size_t count, mpz_t g, cudaStream_t s) {
  mpz_ptr gp = g;
  product_gcd_batch(ctx, n, V, count, 1, &gp, s);
}

void product_gcd_batch(const RnsContext &ctx, const mpz_t n, const uint32_t *V, size_t count, size_t batch,
                       mpz_ptr *g, cudaStream_t s) {
  if (batch == 0) return;
  if (count == 0) {
    for (size_t c = 0; c < batch; c++) mpz_set(g[c], n);
    return;
  }
  GPUECM2_CGBN_DISPATCH(mpz_sizeinbase(n, 2), run, ctx, n, V, count, batch, g, s);
}
