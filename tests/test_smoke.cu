/* Smoke test: GPU present (compute capability >= 7.5), GMP links, CGBN compiles
   and its Montgomery multiplication agrees with GMP. */
#include <cstdio>
#include <cstring>
#include <gmp.h>
#include <cuda_runtime.h>
#include "cgbn/cgbn.h"
#include "host/version.h"

#define BITS 1024
#define TPI 8
typedef struct { cgbn_mem_t<BITS> a, b, m, r; } inst_t;
typedef cgbn_context_t<TPI> ctx_t;
typedef cgbn_env_t<ctx_t, BITS> env_t;

__global__ void k_mul(cgbn_error_report_t *rep, inst_t *in, int count) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
  if (i >= count) return;
  ctx_t ctx(cgbn_report_monitor, rep, i);
  env_t env(ctx.env<env_t>());
  env_t::cgbn_t a, b, m, r;
  cgbn_load(env, a, &in[i].a); cgbn_load(env, b, &in[i].b); cgbn_load(env, m, &in[i].m);
  uint32_t np0 = cgbn_bn2mont(env, a, a, m);
  cgbn_bn2mont(env, b, b, m);
  cgbn_mont_mul(env, r, a, b, m, np0);
  cgbn_mont2bn(env, r, r, m, np0);
  cgbn_store(env, &in[i].r, r);
}

int main() {
  int dev = 0; cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) { printf("FAIL: no CUDA device\n"); return 1; }
  printf("gpuecm2 %s on %s (cc %d.%d, %d SMs, %.1f GB)\n", gpuecm2_version(), p.name, p.major, p.minor,
         p.multiProcessorCount, p.totalGlobalMem / 1e9);
  if (p.major * 10 + p.minor < 75) { printf("FAIL: compute capability 7.5 or newer required\n"); return 1; }

  const int count = 64;
  inst_t h[count]; memset(h, 0, sizeof h);
  gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs, 1);
  mpz_t a[count], b[count], m, t; mpz_inits(m, t, NULL);
  mpz_urandomb(m, rs, BITS - 6); mpz_setbit(m, 0);
  size_t c;
  for (int i = 0; i < count; i++) {
    mpz_init(a[i]); mpz_init(b[i]); mpz_urandomm(a[i], rs, m); mpz_urandomm(b[i], rs, m);
    mpz_export(h[i].a._limbs, &c, -1, 4, 0, 0, a[i]); mpz_export(h[i].b._limbs, &c, -1, 4, 0, 0, b[i]);
    mpz_export(h[i].m._limbs, &c, -1, 4, 0, 0, m);
  }
  inst_t *d; cudaMalloc(&d, sizeof h); cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice);
  cgbn_error_report_t *rep; cgbn_error_report_alloc(&rep);
  k_mul<<<(count * TPI + 127) / 128, 128>>>(rep, d, count);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess || cgbn_error_report_check(rep)) { printf("FAIL: kernel error %s\n", cudaGetErrorString(err)); return 1; }
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < count; i++) {
    mpz_mul(t, a[i], b[i]); mpz_mod(t, t, m);
    mpz_t r; mpz_init(r); mpz_import(r, BITS / 32, -1, 4, 0, 0, h[i].r._limbs);
    if (mpz_cmp(r, t) != 0) bad++;
    mpz_clear(r);
  }
  printf("%s: CGBN mont_mul vs GMP, %d/%d correct\n", bad ? "FAIL" : "OK", count - bad, count);
  return bad != 0;
}
