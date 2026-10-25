/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   ECM stage 2 on the GPU with GMP-ECM's algorithm and parameters, for one
   curve or a batch of curves at a time. One engine per modulus and parameter
   set (it holds the NTT primes and the root plans); run() / run_batch() may
   be called for many curves. Batches give the GPU enough parallel work when
   a single curve's stage 2 is small (small B1). */
#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gmp.h>

#include "gpu/polyalg.h"
#include "gpu/rns.h"
#include "gpu/roots.h"
#include "host/params.h"
#include "host/roots_plan.h"

struct Stage2Times {  // seconds, for the whole batch
  double rootsF = 0, treeF = 0, invF = 0, rootsG = 0, treesG = 0, mulmod = 0, eval = 0, prod = 0, total = 0;
  size_t batch = 1;  // curves that shared these times
};

struct Stage2Outcome {
  int status = 0;    // 0: no factor, 2: factor found (in factor; may equal n), -1: error
  bool check_failed = false;  // the self-check found an inconsistency (status -1)
  Z factor;
  Z first_value;     // H(-f_0), what GMP-ECM prints as "Product of G(f_i)" at -v -v
  bool have_first = false;
  /* prod_i H(-f_i) = prod over all roots f_i of F and g_j of G of (g_j - f_i)
     mod n, whose gcd with n is the result; for tests (GMP-ECM does not print it) */
  Z full_product;
  bool have_full = false;
  Stage2Times t;
};

/* How the CPU waits for the GPU (under WSL2 the CUDA driver polls in both
   cases, keeping a core busy): false (default) sleeps until the GPU is
   done, leaving the core to other work such as CPU stage 1; true
   busy-waits (a full core, slightly lower latency). Call before any other
   GPU work. */
void stage2_cpu_wait(bool spin);

/* How the polynomials' coefficients are held on the GPU: as limbs of residues
   mod n, or in RNS storage (gpu/rns.h: residues mod a second set of NTT
   primes, which halves the conversion work around each product; a few percent
   more or less GPU memory). The results are identical. By default the choice
   is automatic: RNS storage for moduli of STAGE2_RNS_AUTO_BITS bits and more,
   where it made stage 2 8-25% faster with no more memory, limbs below (where
   it gains or loses a few percent). stage2_set_rns_store forces one of them
   for all sizes, the environment variable GPUECM2_RNS=1 or 0 likewise (also
   for programs using the C interface). Taken by an engine when it is made,
   and by the memory estimates. */
constexpr size_t STAGE2_RNS_AUTO_BITS = 3072;
void stage2_set_rns_store(bool on);
void stage2_set_rns_store_auto();
/* whether RNS storage is used for moduli of `bits` bits */
bool stage2_rns_store(size_t bits);

/* Rough upper estimate of the GPU memory (bytes) a stage 2 run of `batch`
   curves needs. */
size_t stage2_gpu_bytes(const mpz_t n, const Stage2Params &sp, size_t batch = 1);
/* Free GPU memory that may be used (a margin is kept for the display etc.). */
size_t stage2_gpu_budget();
/* A batch size for this modulus and these parameters: enough curves to keep
   the GPU busy, as far as budget (bytes) allows; 1 when one curve suffices. */
size_t stage2_auto_batch(const mpz_t n, const Stage2Params &sp, size_t budget);
/* The largest batch size up to want that fits in budget (at least 1). */
size_t stage2_max_batch(const mpz_t n, const Stage2Params &sp, size_t budget, size_t want);
/* GMP-ECM's stage 2 parameters for req (compute_stage2_params); if they
   need more GPU memory than budget (bytes; 0: do not check), the parameters
   GMP-ECM would choose with the smallest larger -k that fits (more, smaller
   blocks, still covering B2). *raised tells whether that happened. Returns
   false with a message in *err if GMP-ECM reports an error or nothing fits. */
bool stage2_params_for_gpu(Stage2Params &sp, const Stage2Request &req, const mpz_t n, size_t budget, bool *raised,
                           std::string *err);

class Stage2Engine {
 public:
  Stage2Engine(const mpz_t n, const Stage2Params &sp);
  ~Stage2Engine();
  Stage2Engine(const Stage2Engine &) = delete;
  Stage2Engine &operator=(const Stage2Engine &) = delete;
  bool ok() const { return ok_; }
  const std::string &error() const { return err_; }
  const RnsContext &rns() const { return ctx_; }
  /* C: the curve and the stage 1 point.
     timed: synchronise between phases so that the times are meaningful
     (this also turns off the overlap of root generation and polynomial
     arithmetic).
     selfcheck: verify F(r) = prod (r + f_i) and G(r) = prod (r + g_i) for
     every product tree at a random r, and the final values H(-f_i) =
     prod_{blocks, j} (g_j - f_i) at a few random roots f_i of F (reduction
     mod F does not change values at its roots, so this covers the
     products mod F and the evaluation as well). Costs a few percent. */
  Stage2Outcome run(const MontgomeryCurve &C, bool timed, bool selfcheck = false) {
    return run_batch(&C, 1, timed, selfcheck)[0];
  }
  /* Stage 2 for count curves (same n) at once; one outcome per curve, each
     exactly as run() would give it. If root generation meets a
     non-invertible value, the curves are redone one by one to find out
     which curve it belongs to (rare). The times are those of the batch. */
  std::vector<Stage2Outcome> run_batch(const MontgomeryCurve *C, size_t count, bool timed, bool selfcheck = false);

  /* Keep the transforms of F's product tree for the evaluation (dF a power
     of two, i.e. moduli up to 1920 bits with GMP-ECM's parameters): 2-6%
     faster, for (log2 dF - 2) (8 P - 4 nl) dF bytes more GPU memory per curve,
     as far as the memory left over by the rest of the run allows. Off by
     default. */
  void set_tree_cache(bool on) { tree_cache_ = on; }

 private:
  bool tree_cache_ = false;
  bool rns_store_ = false;
  Z n_;
  Stage2Params sp_;
  RnsContext ctx_;
  ZnLevels zn_;  // the low levels of the product trees and of the evaluation mod n
  double jmax_ = 0, imax_ = 0;  // largest arguments of the F and G roots
  size_t cap_ = 0;              // point additions per wave (root generation)
  /* root plans (on the GPU) per batch size: the number of progressions per
     curve depends on how many curves share the GPU */
  struct Plans {
    ProgressionPlan F, G;
    std::unique_ptr<RootPlanGPU> gF, gG;
    size_t scalar_bits = 1;  // largest root scalar of both plans
  };
  std::map<size_t, std::unique_ptr<Plans>> plans_;
  const Plans &plans_for(size_t batch);
  /* polynomial arithmetic and root generation run on separate streams, so
     that the roots of the next block of G are computed while the current
     one is processed */
  cudaStream_t s_poly_ = nullptr, s_roots_ = nullptr;
  cudaEvent_t ev_[5] = {};  // F roots done, G block roots done (2), G block consumed (2)
  bool ok_ = false;
  std::string err_;
};
