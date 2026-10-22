/* Benchmark of a complete stage 2 (random curve and point) with GMP-ECM's
   parameters. N is an RSA challenge number (tests/data/rsa_numbers.txt), a
   product of them ("RSA-2048*RSA-1024"), a bit size (random RSA-like N), or
   "@file": the N of the first line of a GMP-ECM resume file.
   Usage: bench_stage2 <N> <B1> [B2] [-power n | -dickson n] [-k n] [-maxmem MB] [-reps r] [-untimed 1] [-treecache 1] [-rns 1]
                       [-batch b] [-spin 1]
   (-untimed: no synchronisation between phases; only the total is meaningful;
   -batch: b curves per run, times per curve; 0: automatic batch size) */
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gmp.h>

#include "api/stage2.h"
#include "gpu/cuda_check.h"
#include "host/params.h"

#ifndef GPUECM2_TEST_DATA
#define GPUECM2_TEST_DATA "tests/data"
#endif

static bool rsa_number(const std::string &name, mpz_t out) {
  std::ifstream in(GPUECM2_TEST_DATA "/rsa_numbers.txt");
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string nm, val;
    ss >> nm >> val;
    if (nm == name) return mpz_set_str(out, val.c_str(), 10) == 0;
  }
  return false;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: bench_stage2 <RSA-xxx[*RSA-yyy] | bits> B1 [B2] [-power n|-dickson n] [-k n] [-reps r]\n");
    return 1;
  }
  gmp_randstate_t rs;
  gmp_randinit_default(rs);
  gmp_randseed_ui(rs, 1);
  MontgomeryCurve E;
  const std::string nspec = argv[1];
  if (nspec[0] == '@') {
    std::ifstream in(nspec.substr(1));
    std::string line;
    std::getline(in, line);
    const size_t a = line.find("N="), b = line.find(';', a);
    if (a == std::string::npos || mpz_set_str(E.n.get(), line.substr(a + 2, b - a - 2).c_str(), 10) != 0) {
      fprintf(stderr, "no N in %s\n", nspec.c_str() + 1);
      return 1;
    }
  } else if (nspec.compare(0, 4, "RSA-") == 0) {
    mpz_set_ui(E.n.get(), 1);
    size_t pos = 0;
    while (pos <= nspec.size()) {
      size_t star = nspec.find('*', pos);
      if (star == std::string::npos) star = nspec.size();
      Z f;
      if (!rsa_number(nspec.substr(pos, star - pos), f.get())) {
        fprintf(stderr, "unknown RSA number %s\n", nspec.substr(pos, star - pos).c_str());
        return 1;
      }
      mpz_mul(E.n.get(), E.n.get(), f.get());
      pos = star + 1;
    }
  } else {
    const int bits = atoi(argv[1]);
    Z p, q;
    mpz_urandomb(p.get(), rs, bits / 2); mpz_setbit(p.get(), bits / 2 - 1); mpz_nextprime(p.get(), p.get());
    mpz_urandomb(q.get(), rs, bits - bits / 2); mpz_setbit(q.get(), bits - bits / 2 - 1); mpz_nextprime(q.get(), q.get());
    mpz_mul(E.n.get(), p.get(), q.get());
  }
  Stage2Request req;
  req.B1 = atof(argv[2]);
  int reps = 3, argi = 3;
  long batch = 1;
  bool spin = false, treecache = false;
  bool timed = true;
  if (argc > 3 && argv[3][0] != '-') { mpz_set_d(req.B2.get(), atof(argv[3])); argi = 4; }
  for (; argi + 1 < argc; argi += 2) {
    if (!strcmp(argv[argi], "-power")) req.S = atoi(argv[argi + 1]);
    else if (!strcmp(argv[argi], "-dickson")) req.S = -atoi(argv[argi + 1]);
    else if (!strcmp(argv[argi], "-k")) req.k = strtoul(argv[argi + 1], nullptr, 10);
    else if (!strcmp(argv[argi], "-reps")) reps = atoi(argv[argi + 1]);
    else if (!strcmp(argv[argi], "-untimed")) timed = !atoi(argv[argi + 1]);
    else if (!strcmp(argv[argi], "-batch")) batch = atol(argv[argi + 1]);
    else if (!strcmp(argv[argi], "-maxmem")) req.maxmem = atof(argv[argi + 1]) * 1048576.;
    else if (!strcmp(argv[argi], "-spin")) spin = atoi(argv[argi + 1]) != 0;
    else if (!strcmp(argv[argi], "-treecache")) treecache = atoi(argv[argi + 1]) != 0;
    else if (!strcmp(argv[argi], "-rns")) stage2_set_rns_store(atoi(argv[argi + 1]) != 0);
    else { fprintf(stderr, "unknown option %s\n", argv[argi]); return 1; }
  }
  stage2_cpu_wait(spin);
  Stage2Params sp;
  std::string err;
  /* as gpuecm2 does: GMP-ECM's parameters, with more blocks if dF does not fit in the GPU memory */
  if (!stage2_params_for_gpu(sp, req, E.n.get(), stage2_gpu_budget(), nullptr, &err)) {
    printf("%s\n", err.c_str());
    return 1;
  }
  gmp_printf("%zu-bit N, B1=%.0f, B2=%Zd: dF=%lu, k=%lu, S=%d, GPU estimate %.0f MB\n", mpz_sizeinbase(E.n.get(), 2),
             req.B1, sp.B2.get(), sp.dF, sp.k, sp.S, stage2_gpu_bytes(E.n.get(), sp) / 1048576.);
  if (batch <= 0) batch = (long)stage2_auto_batch(E.n.get(), sp, stage2_gpu_budget());
  Stage2Engine eng(E.n.get(), sp);
  eng.set_tree_cache(treecache);
  if (!eng.ok()) { printf("%s\n", eng.error().c_str()); return 1; }
  Stage2Times best;
  best.total = 1e30;
  std::vector<MontgomeryCurve> C(batch);
  for (int rep = 0; rep <= reps; rep++) {  // rep 0 is a warm-up
    for (MontgomeryCurve &c : C) {
      mpz_set(c.n.get(), E.n.get());
      mpz_urandomm(c.A.get(), rs, E.n.get());
      mpz_urandomm(c.u.get(), rs, E.n.get());
    }
    std::vector<Stage2Outcome> o = eng.run_batch(C.data(), C.size(), timed);
    for (size_t i = 0; i < o.size(); i++)
      if (o[i].status != 0) printf("  run %d curve %zu: status %d\n", rep, i, o[i].status);
    if (rep > 0 && o[0].t.total < best.total) best = o[0].t;
  }
  const Stage2Times &t = best;
  const double per = 1e3 / (double)batch;
  if (batch > 1) printf("  batch of %ld curves, times per curve:\n", batch);
  printf("  rootsF %.1f  treeF %.1f  invF %.1f  rootsG %.1f  treesG %.1f  mulmod %.1f  eval %.1f  prod %.1f ms\n",
         t.rootsF * per, t.treeF * per, t.invF * per, t.rootsG * per, t.treesG * per, t.mulmod * per, t.eval * per,
         t.prod * per);
  printf("  total %.2f ms (best of %d)\n", t.total * per, reps);
  if (getenv("BENCH_MEM")) {  // peak of the stream-ordered pool (all allocations of stage 2 go there)
    cudaMemPool_t pool;
    uint64_t used_hi = 0, res_hi = 0;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess &&
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_hi) == cudaSuccess &&
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &res_hi) == cudaSuccess)
      printf("  GPU memory pool peak %.0f MB (reserved %.0f MB), estimate %.0f MB (%ld curves)\n",
             used_hi / 1048576., res_hi / 1048576., stage2_gpu_bytes(E.n.get(), sp, batch) / 1048576., batch);
  }
  return 0;
}
