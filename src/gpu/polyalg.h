/* Copyright 2026 Erik Branger.
   The algorithms follow GMP-ECM's stage 2 (stage2.c, ecm_ntt.c, polyeval.c:
   Copyright 2001-2012 Paul Zimmermann, Alexander Kruppa, Dave Newman et al.,
   GNU LGPL version 3 or later), reorganised for batched GPU execution.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Polynomials over Z/nZ for ECM stage 2. All arrays are limb arrays (see rns.h)
   whose leading dimension equals their own length unless stated otherwise.
   Monic polynomials of degree d are stored as their d low coefficients
   (the leading 1 is implicit), as in GMP-ECM. With roots f_0..f_{n-1},
     F(x) = prod (x + f_i)
   (GMP-ECM's sign convention), so F vanishes at x = -f_i. */
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "gpu/rns.h"

/* Product tree over n leaves. Level l (0 <= l < L, L = ceil(log2 n)) holds the
   products of 2^l consecutive leaves: node j has degree min(2^l, n - j 2^l)
   and its coefficients start at entry j*2^l of the level's n-entry array.
   Level 0 holds the roots themselves; the top (degree n) is F. */
struct ProductTree {
  size_t n = 0;
  size_t batch = 1;  // number of trees; tree c uses entries c*n .. c*n+n-1 of every level
  int L = 0;
  std::vector<uint32_t *> level;
};

/* All functions below work on `batch` independent problems of the same size
   at once (one per curve): problem c's n coefficients are entries
   c*n .. c*n+n-1 of each array, whose leading dimension is batch*n. */

/* F (n coefficients) := prod (x + roots[i]). With tree != nullptr the levels
   0..L-1 are kept in *tree (free with tree_free). */
void tree_build(const RnsContext &ctx, const uint32_t *roots, size_t n, uint32_t *F, ProductTree *tree,
                cudaStream_t s = 0, size_t batch = 1);
void tree_free(ProductTree &tree, cudaStream_t s = 0);

/* I (n coefficients) := 1 / rev(F) mod x^n, where rev(F) = x^n F(1/x)
   = 1 + F[n-1] x + ... + F[0] x^n. */
void poly_invert(const RnsContext &ctx, const uint32_t *F, size_t n, uint32_t *I, cudaStream_t s = 0,
                 size_t batch = 1);

/* Precomputed data for repeated reductions modulo F (one F per problem). */
class MulModF {
 public:
  MulModF(const RnsContext &ctx, const uint32_t *F, const uint32_t *I, size_t n, cudaStream_t s = 0,
          size_t batch = 1);
  ~MulModF();
  /* H := G * H mod F, with deg G, deg H < n. G and H have n coefficients. */
  void mulmod(const uint32_t *G, uint32_t *H, cudaStream_t s = 0);

 private:
  const RnsContext &ctx_;
  size_t n_, batch_;
  int logT2_, logTq_;  // transform lengths for products of length 2n and for q*F
  bool wrap_;          // q*F computed with a length-n cyclic convolution
  uint32_t *I_hat_ = nullptr, *F_hat_ = nullptr;
  const uint32_t *F_ = nullptr, *I_ = nullptr;  // set when products exceed the NTT length (fallback)
  cudaStream_t s_;  // stream the cached transforms were allocated on (freed stream-ordered)
};

/* values[i] := H(-f_i) for the roots f_i of F (the leaves of tree), deg H < n,
   using GMP-ECM's transposed ("Tellegen") product tree descent. */
void poly_eval_tree(const RnsContext &ctx, const ProductTree &tree, const uint32_t *I, const uint32_t *H,
                    uint32_t *values, cudaStream_t s = 0);
