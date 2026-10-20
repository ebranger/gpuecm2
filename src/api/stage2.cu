/* Copyright 2026 Erik Branger.
   The structure follows GMP-ECM's stage2() (stage2.c, Copyright 2001-2012
   Paul Zimmermann, Alexander Kruppa et al., GNU LGPL v3+).
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "api/stage2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "gpu/cuda_check.h"
#include "gpu/limbops.h"
#include "gpu/polyalg.h"
#include "gpu/prod.h"

namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int clog2(size_t n) { int k = 0; while (((size_t)1 << k) < n) k++; return k; }
/* Number of progressions per curve for a root plan of `count` roots f(v) P
   with v up to vmax, for `batch` curves sharing the GPU, given that `cap`
   additions run concurrently. Each step does batch*nprog*S additions, best a
   whole number m of waves (nprog = m cap/(S batch)); more progressions mean
   fewer steps (each costing some launch overhead, taken as a tenth of a
   wave) but more initial scalar multiplications (nprog*(S+1) scalars of
   about S*log2(vmax) bits per curve, one addition per 6 bits). Minimises the
   estimated total in units of additions. */
size_t nprog_target(int S_, size_t count, double vmax, size_t cap, size_t batch) {
  const double S = std::max(1, std::abs(S_)), C = (double)std::max<size_t>(cap, 64), B = (double)batch;
  const double w = S * std::log2(std::max(vmax, 2.0)) / 6 + 1;
  double best = 1e300;
  size_t target = 1;
  for (int m = 1; m <= 8; m++) {
    const double np = std::max(1.0, std::floor(m * C / (S * B)));
    const double steps = std::ceil((double)count / np);
    const double waves = std::ceil(B * np * S / C);
    const double cost = B * np * (S + 1) * w + steps * (waves * C + 0.1 * C);
    if (cost < best) { best = cost; target = (size_t)np; }
  }
  return target;
}
size_t cgbn_bits(size_t bits) {  // see gpu/cgbn_sizes.h
  size_t cgbn = 512;
  while (cgbn < bits) cgbn += cgbn < 2048 ? 256 : 512;
  return cgbn;
}
}  // namespace

void stage2_cpu_wait(bool spin) {
  cudaSetDeviceFlags(spin ? cudaDeviceScheduleSpin : cudaDeviceScheduleBlockingSync);
}

size_t stage2_gpu_bytes(const mpz_t n, const Stage2Params &sp, size_t batch) {
  const size_t bits = mpz_sizeinbase(n, 2), nl4 = 4 * ((bits + 31) / 32), dF = sp.dF, B = batch;
  const size_t P = (2 * bits + clog2(2 * dF + 2) + 2) / 30 + 2;
  const size_t cgbn = cgbn_bits(bits);
  const size_t L = clog2(dF), T2 = (size_t)1 << clog2(2 * dF);
  size_t b = nl4 * dF * (L + 2 + 4);          // tree, F, I, Gb, G, H, V
  b += 7 * (cgbn / 8) * dF;                   // root generation temporaries (output ring, inverses)
  b += 4 * P * 2 * T2;                        // cached transforms of F and 1/F
  b *= B;
  b += std::max<size_t>((size_t)768 << 20, 4 * P * 3 * T2 * B);  // transient products
  b += 8 * (cgbn / 8) * 9 * 16384;            // progression tables (about 8 waves)
  return b + ((size_t)256 << 20);
}

size_t stage2_gpu_budget() {
  if (const char *e = getenv("GPUECM2_GPU_MEM"))  // testing aid: pretend this many MB are usable
    return (size_t)(atof(e) * 1048576.);
  size_t free_b = 0, total_b = 0;
  if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
  /* memory the stream-ordered pool keeps from earlier runs is available too */
  cudaMemPool_t pool;
  int dev = 0;
  if (cudaGetDevice(&dev) == cudaSuccess && cudaDeviceGetDefaultMemPool(&pool, dev) == cudaSuccess) {
    uint64_t reserved = 0, used = 0;
    if (cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved) == cudaSuccess &&
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used) == cudaSuccess && reserved > used)
      free_b += (size_t)(reserved - used);
  }
  /* stay well clear of the limit: on Windows/WSL exceeding it makes the
     driver page GPU memory to the host, which is dramatically slow */
  const size_t margin = std::max<size_t>((size_t)512 << 20, free_b / 10);
  return free_b > margin ? free_b - margin : 0;
}

size_t stage2_max_batch(const mpz_t n, const Stage2Params &sp, size_t budget, size_t want) {
  size_t B = std::max<size_t>(1, want);
  while (B > 1 && stage2_gpu_bytes(n, sp, B) > budget) B = B * 3 / 4;
  return B;
}

bool stage2_params_for_gpu(Stage2Params &sp, const Stage2Request &req, const mpz_t n, size_t budget, bool *raised,
                           std::string *err) {
  if (raised) *raised = false;
  if (!compute_stage2_params(sp, req, n, err)) return false;
  if (!sp.dF || !sp.k || !budget || stage2_gpu_bytes(n, sp) <= budget) return true;
  Stage2Request r2 = req;
  for (unsigned long k2 = sp.k + 1; k2 <= 100000; k2 = k2 < 16 ? k2 + 1 : k2 * 5 / 4) {
    r2.k = k2;
    if (!compute_stage2_params(sp, r2, n, err)) break;
    if (stage2_gpu_bytes(n, sp) <= budget) {
      if (raised) *raised = true;
      return true;
    }
  }
  char buf[120];
  snprintf(buf, sizeof buf, "Error: not enough GPU memory for this stage 2 (%.1f GB free)", budget / 1e9);
  *err = buf;
  return false;
}

size_t stage2_auto_batch(const mpz_t n, const Stage2Params &sp, size_t budget) {
  /* about 2^17 coefficients per polynomial operation keep an RTX 4070 Ti
     SUPER (66 SMs) busy (measured); scaled with the number of SMs */
  constexpr size_t MAX_BATCH = 512;
  if (sp.dF == 0) return 1;
  int dev = 0, sms = 66;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess)
    sms = 66;
  const size_t target = ((size_t)1 << 17) * (size_t)std::max(sms, 1) / 66;
  return stage2_max_batch(n, sp, budget, std::min(MAX_BATCH, std::max<size_t>(1, target / sp.dF)));
}

Stage2Engine::Stage2Engine(const mpz_t n, const Stage2Params &sp) : sp_(sp) {
  mpz_set(n_.get(), n);
  if (sp.dF == 0 || sp.k == 0) { err_ = "no stage 2 (B2 < B2min)"; return; }
  int want_logT = std::max(12, clog2(2 * sp.dF));
  if (const char *e = getenv("GPUECM2_MAX_LOGT"))  // testing aid: shorter NTTs force the long-product paths
    want_logT = std::max(12, std::min(want_logT, atoi(e)));
  if (!ctx_.init(n, 2 * sp.dF + 2, want_logT, &err_)) return;
  /* transient buffers come from the default stream-ordered pool; keep its
     memory between synchronisations instead of returning it to the driver
     (mapping it again for every curve is slow, notably on Windows/WSL) */
  cudaMemPool_t pool;
  if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
    uint64_t keep = UINT64_MAX;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &keep);
  }
  /* largest arguments: F: j*d2 with the dF-th j == 1 (mod 6) coprime to d1;
     G: i*d1 with the (k dF)-th i >= i0 coprime to d2 */
  jmax_ = 2.0 * sp.dF * sp.d1 / eulerphi(sp.d1);
  imax_ = std::fabs(mpz_get_d(sp.i0.get())) + (double)sp.k * sp.dF * sp.d2 / eulerphi(sp.d2);
  cap_ = root_step_capacity(mpz_sizeinbase(n, 2));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_poly_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaStreamCreateWithFlags(&s_roots_, cudaStreamNonBlocking));
  for (auto &e : ev_) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  ok_ = true;
}

const Stage2Engine::Plans &Stage2Engine::plans_for(size_t batch) {
  std::unique_ptr<Plans> &p = plans_[batch];
  if (!p) {
    p.reset(new Plans);
    p->F = plan_roots_F(sp_, nprog_target(sp_.S, sp_.dF, jmax_ * sp_.d2, cap_, batch));
    p->G = plan_roots_G(sp_, nprog_target(sp_.S, sp_.k * sp_.dF, imax_ * sp_.d1, cap_, batch));
    if (getenv("GPUECM2_DEBUG"))
      fprintf(stderr, "roots (batch %zu): wave %zu additions; F %zu progressions x %zu steps, G %zu x %zu\n", batch,
              cap_, p->F.nprog, p->F.steps, p->G.nprog, p->G.steps);
    p->gF.reset(new RootPlanGPU(p->F, sp_.dF));
    p->gG.reset(new RootPlanGPU(p->G, sp_.dF));
    p->scalar_bits = std::max(p->gF->scalar_bits, p->gG->scalar_bits);
  }
  return *p;
}

Stage2Engine::~Stage2Engine() {
  if (s_poly_) cudaStreamSynchronize(s_poly_);
  if (s_roots_) cudaStreamSynchronize(s_roots_);
  for (auto &e : ev_)
    if (e) cudaEventDestroy(e);
  if (s_poly_) cudaStreamDestroy(s_poly_);
  if (s_roots_) cudaStreamDestroy(s_roots_);
}

std::vector<Stage2Outcome> Stage2Engine::run_batch(const MontgomeryCurve *Cs, size_t count, bool timed,
                                                   bool selfcheck) {
  std::vector<Stage2Outcome> outs(count);
  if (count == 0) return outs;
  const RnsContext &ctx = ctx_;
  const size_t dF = sp_.dF, k = sp_.k, B = count, N = B * dF;
  const Plans &pl = plans_for(B);
  const cudaStream_t sp = s_poly_, sr = timed ? s_poly_ : s_roots_;
  Stage2Times tm;
  tm.batch = B;
  const double t_start = now();
  double t0 = now();
  auto lap = [&](double &acc) {
    if (timed) CUDA_CHECK(cudaDeviceSynchronize());
    double t = now();
    acc += t - t0;
    t0 = t;
  };
  auto after = [&](cudaStream_t waiter, cudaStream_t s, cudaEvent_t e) {  // waiter waits for s's work so far
    if (waiter == s) return;
    CUDA_CHECK(cudaEventRecord(e, s));
    CUDA_CHECK(cudaStreamWaitEvent(waiter, e, 0));
  };
  auto sync = [&]() {
    CUDA_CHECK(cudaStreamSynchronize(sp));
    CUDA_CHECK(cudaStreamSynchronize(sr));
    tm.total = now() - t_start;
    for (Stage2Outcome &o : outs) o.t = tm;
  };
  Z rfac;  // the gcd when root generation meets a non-invertible value
  /* root generation failed: a factor (for one curve: the result; for a batch:
     redo the curves one by one, the culprit is not known) or an error */
  auto root_failure = [&](int r) {
    sync();
    if (r == ROOTS_FACTOR && B > 1) {
      for (size_t c = 0; c < B; c++) outs[c] = run_batch(&Cs[c], 1, timed, selfcheck)[0];
      return outs;
    }
    for (Stage2Outcome &o : outs) o.status = r == ROOTS_FACTOR ? 2 : -1;
    if (r == ROOTS_FACTOR) mpz_set(outs[0].factor.get(), rfac.get());
    return outs;
  };
  /* roots of F (the points' tables are shared with G), F and its product tree, 1/F */
  RootGenerator gen;
  int r = gen.init(Cs, B, pl.scalar_bits, rfac.get(), sr);
  uint32_t *FR = limb_alloc(ctx, N, sr);
  if (r == ROOTS_OK) r = gen.start(*pl.gF);
  if (r == ROOTS_OK) r = gen.next_block(ctx, FR, rfac.get());
  lap(tm.rootsF);
  if (r != ROOTS_OK) {
    limb_free(FR, sr);
    return root_failure(r);
  }
  after(sp, sr, ev_[0]);
  uint32_t *F = limb_alloc(ctx, N, sp), *I = limb_alloc(ctx, N, sp);
  ProductTree tree;
  /* with set_tree_cache, the transforms of F's tree are kept for the
     evaluation as far as the GPU memory left over by this run allows (80% of it) */
  size_t keep = 0;
  if (tree_cache_) {
    const size_t budget = stage2_gpu_budget(), need = stage2_gpu_bytes(n_.get(), sp_, B);
    if (budget > need) keep = (budget - need) / 10 * 8;
  }
  tree_build(ctx, FR, dF, F, &tree, sp, B, keep);
  limb_free(FR, sp);
  if (const char *fi = getenv("GPUECM2_INJECT_FAULT"))  // testing aid: 2: corrupt F (caught by the tree check)
    if (atoi(fi) == 2) {
      uint32_t w;
      CUDA_CHECK(cudaMemcpyAsync(&w, F + 7, 4, cudaMemcpyDeviceToHost, sp));
      CUDA_CHECK(cudaStreamSynchronize(sp));
      w ^= 1;
      CUDA_CHECK(cudaMemcpyAsync(F + 7, &w, 4, cudaMemcpyHostToDevice, sp));
    }
  lap(tm.treeF);
  /* self-check data per curve: a random point r, and a few roots f_i of F */
  constexpr int NCHK = 3;
  struct Check {
    Z val[1 + NCHK], acc[NCHK], x[1 + NCHK];  // x = {r, -f_i}
    size_t idx[NCHK] = {};
    bool ok = true;
  };
  std::vector<Check> chk(selfcheck ? B : 0);
  const int nl = ctx.nlimbs();
  for (size_t c = 0; c < chk.size(); c++) {
    Check &K = chk[c];
    gmp_randstate_t rs;
    gmp_randinit_default(rs);
    gmp_randseed_ui(rs, (unsigned long)mpz_get_ui(Cs[c].u.get()) ^ 0x5eed);
    mpz_urandomm(K.x[0].get(), rs, n_.get());
    std::vector<uint32_t> w(nl);
    for (int t = 0; t < NCHK; t++) {
      K.idx[t] = (size_t)gmp_urandomm_ui(rs, dF);
      CUDA_CHECK(cudaMemcpy2DAsync(w.data(), 4, tree.level[0] + c * dF + K.idx[t], N * 4, 4, nl,
                                   cudaMemcpyDeviceToHost, sp));
      CUDA_CHECK(cudaStreamSynchronize(sp));
      mpz_import(K.x[1 + t].get(), nl, -1, 4, 0, 0, w.data());
      mpz_sub(K.x[1 + t].get(), n_.get(), K.x[1 + t].get());
      mpz_mod(K.x[1 + t].get(), K.x[1 + t].get(), n_.get());
      mpz_set_ui(K.acc[t].get(), 1);
    }
    gmp_randclear(rs);
    Z fr;
    mpz_srcptr rr = K.x[0].get();
    mpz_ptr pr = K.val[0].get();
    poly_value(ctx, n_.get(), F + c * dF, N, dF, true, K.x[0].get(), fr.get(), sp);
    roots_value(ctx, n_.get(), tree.level[0] + c * dF, N, dF, &rr, 1, &pr, sp);
    K.ok &= mpz_cmp(fr.get(), K.val[0].get()) == 0;
  }
  poly_invert(ctx, F, dF, I, sp, B);
  lap(tm.invF);
  /* G block by block: its roots (block b+1 generated on the root stream
     while block b is processed), G from its roots, H = (G - F) H mod F */
  uint32_t *Gb[2] = {limb_alloc(ctx, N, sp), limb_alloc(ctx, N, sp)};
  uint32_t *G = limb_alloc(ctx, N, sp), *H = limb_alloc(ctx, N, sp), *V = limb_alloc(ctx, N, sp);
  auto free_all = [&]() {
    for (uint32_t *p : {Gb[0], Gb[1], G, H, V, F, I}) limb_free(p, sp);
    tree_free(tree, sp);
  };
  after(sr, sp, ev_[3]);  // the allocations above
  r = gen.start(*pl.gG);
  {
    MulModF mm(ctx, F, I, dF, sp, B);
    for (size_t b = 0; b < k && r == ROOTS_OK; b++) {
      if (b >= 2) CUDA_CHECK(cudaStreamWaitEvent(sr, ev_[3 + b % 2], 0));  // Gb[b%2] consumed by block b-2
      r = gen.next_block(ctx, Gb[b % 2], rfac.get());
      lap(tm.rootsG);
      if (r != ROOTS_OK) break;
      after(sp, sr, ev_[1 + b % 2]);
      tree_build(ctx, Gb[b % 2], dF, G, nullptr, sp, B);
      for (size_t c = 0; c < chk.size(); c++) {  // G(r) = prod (r + g_j); accumulate prod (g_j - f_i)
        Check &K = chk[c];
        mpz_srcptr xs[1 + NCHK];
        mpz_ptr vs[1 + NCHK];
        for (int t = 0; t <= NCHK; t++) { xs[t] = K.x[t].get(); vs[t] = K.val[t].get(); }
        roots_value(ctx, n_.get(), Gb[b % 2] + c * dF, N, dF, xs, 1 + NCHK, vs, sp);
        Z gr;
        poly_value(ctx, n_.get(), G + c * dF, N, dF, true, K.x[0].get(), gr.get(), sp);
        K.ok &= mpz_cmp(gr.get(), K.val[0].get()) == 0;
        for (int t = 0; t < NCHK; t++) {
          mpz_mul(K.acc[t].get(), K.acc[t].get(), K.val[1 + t].get());
          mpz_mod(K.acc[t].get(), K.acc[t].get(), n_.get());
        }
      }
      if (sp != sr) CUDA_CHECK(cudaEventRecord(ev_[3 + b % 2], sp));
      limb_sub(ctx, G, N, F, N, G, N, N, sp);  // G mod F = G - F (both monic of degree dF)
      lap(tm.treesG);
      if (b == 0) limb_copy(ctx, G, N, H, N, N, false, sp);
      else mm.mulmod(G, H, sp);
      lap(tm.mulmod);
    }
  }
  if (r != ROOTS_OK) {
    free_all();
    return root_failure(r);
  }
  if (const char *fi = getenv("GPUECM2_INJECT_FAULT")) {  // testing aid for the self-check
    /* 1: corrupt H (of the first curve) before the evaluation (caught by the final check) */
    if (atoi(fi) == 1) {
      uint32_t w;
      CUDA_CHECK(cudaMemcpyAsync(&w, H + 7, 4, cudaMemcpyDeviceToHost, sp));
      CUDA_CHECK(cudaStreamSynchronize(sp));
      w ^= 1;
      CUDA_CHECK(cudaMemcpyAsync(H + 7, &w, 4, cudaMemcpyHostToDevice, sp));
      CUDA_CHECK(cudaStreamSynchronize(sp));
    }
  }
  poly_eval_tree(ctx, tree, I, H, V, sp);
  lap(tm.eval);
  /* first values (GMP-ECM's "Product of G(f_i)" debug line) and the products */
  {
    std::vector<uint32_t> v0((size_t)nl * B);
    for (size_t c = 0; c < B; c++)  // limb j of curve c's first value at V[j*N + c*dF]
      CUDA_CHECK(cudaMemcpy2DAsync(v0.data() + c * nl, 4, V + c * dF, N * 4, 4, nl, cudaMemcpyDeviceToHost, sp));
    CUDA_CHECK(cudaStreamSynchronize(sp));
    for (size_t c = 0; c < B; c++) {
      mpz_import(outs[c].first_value.get(), nl, -1, 4, 0, 0, v0.data() + c * nl);
      outs[c].have_first = true;
    }
    for (size_t c = 0; c < chk.size(); c++)  // H(-f_i) = prod over all blocks of prod (g_j - f_i)
      for (int t = 0; t < NCHK; t++) {
        CUDA_CHECK(cudaMemcpy2DAsync(v0.data(), 4, V + c * dF + chk[c].idx[t], N * 4, 4, nl, cudaMemcpyDeviceToHost,
                                     sp));
        CUDA_CHECK(cudaStreamSynchronize(sp));
        Z v;
        mpz_import(v.get(), nl, -1, 4, 0, 0, v0.data());
        chk[c].ok &= mpz_cmp(v.get(), chk[c].acc[t].get()) == 0;
      }
  }
  std::vector<Z> g(B);
  std::vector<mpz_ptr> gp(B);
  for (size_t c = 0; c < B; c++) gp[c] = g[c].get();
  product_gcd_batch(ctx, n_.get(), V, dF, B, gp.data(), sp);
  lap(tm.prod);
  free_all();
  sync();
  for (size_t c = 0; c < B; c++) {
    Stage2Outcome &o = outs[c];
    if (!chk.empty() && !chk[c].ok) {
      o.check_failed = true;
      o.status = -1;
    } else if (mpz_cmp_ui(g[c].get(), 1) != 0) {
      mpz_set(o.factor.get(), g[c].get());
      o.status = 2;
    } else {
      o.status = 0;
    }
  }
  return outs;
}
