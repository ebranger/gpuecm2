/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   GPU computation of the stage 2 roots x(f(v) P) (see host/roots_plan.h). */
#pragma once
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <gmp.h>

#include "gpu/rns.h"
#include "host/mpz_class_lite.h"
#include "host/roots_plan.h"

/* The stage 1 result as GMP-ECM passes it to stage 2: the point (u, 1) on the
   Montgomery curve B v^2 = u^3 + A u^2 + u, so B = u^3 + A u^2 + u
   (canonical residues mod n). GMP-ECM's stage 2 works on the short Weierstrass
   model with x = (3u + A) / (3B), y = v / B (montgomery_to_weierstrass), and
   the roots are such x-coordinates. Here the arithmetic is done on the
   equivalent twisted Edwards curve; the roots are the same numbers. */
struct MontgomeryCurve {
  Z n, A, u;
};

enum { ROOTS_OK = 0, ROOTS_FACTOR = 1, ROOTS_ERROR = -1 };

/* A plan prepared for the GPU: its scalars and output slots on the device
   and the block schedule for producing the roots block_len at a time. Made
   once per parameter set and reused for every curve. */
class RootPlanGPU {
 public:
  RootPlanGPU() = default;
  RootPlanGPU(const ProgressionPlan &plan, size_t block_len);
  RootPlanGPU(const RootPlanGPU &) = delete;
  RootPlanGPU &operator=(const RootPlanGPU &) = delete;
  ~RootPlanGPU();
  int S = 0, swords = 0;
  size_t scalar_bits = 1;   // largest scalar
  size_t nprog = 0, steps = 0, count = 0;
  size_t blen = 0, nblocks = 0, nring = 0;  // output blocks and the ring holding them
  std::vector<size_t> last_step;            // per block: the last step with an output in it
  uint32_t *scal = nullptr;                 // nprog*(S+1) magnitudes, swords words each
  int32_t *nbits = nullptr;                 // signed bit lengths
  int32_t *slot = nullptr;                  // steps*nprog output slots (table form), or
  int64_t *base = nullptr;                  // nprog first output indices (affine form)
  int64_t inc = 0;                          // ... and the index increment per step
};

/* Root generation for one curve or a batch of curves with the same n.
   init() builds the fixed-base tables of the points for scalars of up to
   maxbits bits (shared by all plans computed afterwards). start() begins a
   plan whose roots are then produced in plan order by next_block(),
   plan.blen at a time for every curve (the last block may be shorter): each
   call writes the next block of curve c to entries c*plan.blen .. of out
   (limb array, leading dimension count*plan.blen, ctx.nlimbs() limbs); the
   plan must stay alive. The progression state is kept between calls, so the
   memory needed is O(block_len), not O(plan.count). compute() (one curve
   only) is start() plus one block holding all roots. The point at infinity
   gives the value 0, as in GMP-ECM. All return ROOTS_FACTOR with the gcd in
   factor if a non-invertible value is met; failed_curve() then tells which
   curve it belonged to, if known (else SIZE_MAX). Work is on stream s. */
class RootGenerator {
 public:
  RootGenerator() = default;
  RootGenerator(const RootGenerator &) = delete;
  RootGenerator &operator=(const RootGenerator &) = delete;
  ~RootGenerator();
  int init(const MontgomeryCurve *C, size_t count, size_t maxbits, mpz_t factor, cudaStream_t s = 0);
  int init(const MontgomeryCurve &C, size_t maxbits, mpz_t factor, cudaStream_t s = 0) {
    return init(&C, 1, maxbits, factor, s);
  }
  int start(const RootPlanGPU &plan);
  int next_block(const RnsContext &ctx, uint32_t *out, mpz_t factor);
  int compute(const RnsContext &ctx, const ProgressionPlan &plan, uint32_t *out, mpz_t factor);
  size_t failed_curve() const { return failed_; }
  struct Impl;

 private:
  Impl *impl_ = nullptr;
  size_t count_ = 0, failed_ = SIZE_MAX;
};

/* how many point additions of one progression step run concurrently on this
   GPU for a modulus of nbits bits (one "wave") */
size_t root_step_capacity(size_t nbits);

/* largest scalar (bits) a plan needs */
size_t plan_scalar_bits(const ProgressionPlan &plan);

/* convenience: one plan with its own table */
int compute_roots(const RnsContext &ctx, const MontgomeryCurve &C, const ProgressionPlan &plan, uint32_t *out,
                  mpz_t factor, cudaStream_t s = 0);

/* For tests: R = P + Q, D = 2P and M = P + Q with Q normalised (the mixed
   addition), computed on the GPU on the Edwards model of C. Inputs and
   outputs are affine points of GMP-ECM's Weierstrass model (or infinity).
   Returns false if a result is not representable. */
struct AffinePoint {
  Z x, y;
  bool inf = false;
};
bool ec_test_ops(const MontgomeryCurve &C, const AffinePoint &P, const AffinePoint &Q, AffinePoint &R,
                 AffinePoint &D, AffinePoint &M);
