/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <gmp.h>

#include "gpu/polyalg.h"
#include "gpu/rns.h"

/* g := gcd(prod V_i, n) for count residues in limb array V (leading dimension count) */
void product_gcd(const RnsContext &ctx, const mpz_t n, const uint32_t *V, size_t count, mpz_t g,
                 cudaStream_t s = 0);
/* g[c] := gcd(prod_i V_{c*count + i}, n) for c < batch (leading dimension batch*count);
   with prod, also prod[c] := prod_i V_{c*count + i} mod n (canonical) */
void product_gcd_batch(const RnsContext &ctx, const mpz_t n, const uint32_t *V, size_t count, size_t batch,
                       mpz_ptr *g, cudaStream_t s = 0, mpz_ptr *prod = nullptr);

/* For the self-check. out := sum_{i<len} coef[i] x^i (+ x^len if monic) mod n,
   coef a limb array with leading dimension ld. */
void poly_value(const RnsContext &ctx, const mpz_t n, const uint32_t *coef, size_t ld, size_t len, bool monic,
                const mpz_t x, mpz_t out, cudaStream_t s = 0);
/* out[t] := prod_{i<len} (roots[i] + r[t]) mod n for t < nr (roots: limb array,
   leading dimension ld; r[t] canonical) */
void roots_value(const RnsContext &ctx, const mpz_t n, const uint32_t *roots, size_t ld, size_t len,
                 const mpz_srcptr *r, int nr, mpz_ptr *out, cudaStream_t s = 0);

/* The low levels of the product trees and of the evaluation mod n (see ZnLevels in
   polyalg.h) for n of a size this build has: levels 0 .. levels-1. False if n is too
   large for the build. */
bool zn_levels_init(ZnLevels *z, const mpz_t n, int levels);
void zn_levels_free(ZnLevels *z);
