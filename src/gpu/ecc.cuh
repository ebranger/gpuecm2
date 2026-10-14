/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Elliptic curve arithmetic mod n with CGBN (cooperative groups of TPI
   threads per number), on twisted Edwards curves a x^2 + y^2 = 1 + d x^2 y^2
   in extended coordinates (X : Y : Z : T), x = X/Z, y = Y/Z, T = XY/Z,
   neutral element (0 : 1 : 1 : 0) (Hisil, Wong, Carter, Dawson, "Twisted
   Edwards curves revisited", 2008). Every Montgomery curve, hence every
   GMP-ECM curve, has such a model.
   All residues are kept in Montgomery form (R = 2^BITS), fully reduced. */
#pragma once
#include <cstdint>

#include "cgbn/cgbn.h"

template <uint32_t TPI_, uint32_t BITS_>
struct EcEnv {
  static constexpr uint32_t TPI = TPI_, BITS = BITS_;
  /* CGBN pads numbers whose limb count is not a multiple of TPI; that mode is
     slow and gave wrong results with this code, so it is not allowed */
  static_assert((BITS / 32) % TPI == 0, "padded CGBN configuration (BITS/32 not a multiple of TPI)");
  typedef cgbn_context_t<TPI> context_t;
  typedef cgbn_env_t<context_t, BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;
  typedef cgbn_mem_t<BITS> mem_t;

  env_t env;
  bn_t n;
  uint32_t np0;

  __device__ __forceinline__ EcEnv(context_t &ctx, const mem_t *nm, uint32_t np0_) : env(ctx.template env<env_t>()), np0(np0_) {
    cgbn_load(env, n, const_cast<mem_t *>(nm));
  }

  /* CGBN's Montgomery product is only reduced to [0, 2^BITS) (it subtracts n
     once on a carry out of the top word), so normalise to [0, n) with one more
     conditional subtraction: all other operations here assume fully reduced
     inputs. Any odd n < 2^BITS. */
  __device__ __forceinline__ void mul(bn_t &r, const bn_t &a, const bn_t &b) {
    cgbn_mont_mul(env, r, a, b, n, np0);
    if (cgbn_compare(env, r, n) >= 0) cgbn_sub(env, r, r, n);
  }
  /* conversions into / out of Montgomery form, fully reduced */
  __device__ __forceinline__ void to_mont(bn_t &r, const bn_t &a) {
    cgbn_bn2mont(env, r, a, n);
    if (cgbn_compare(env, r, n) >= 0) cgbn_sub(env, r, r, n);
  }
  __device__ __forceinline__ void from_mont(bn_t &r, const bn_t &a) {
    cgbn_mont2bn(env, r, a, n, np0);
    if (cgbn_compare(env, r, n) >= 0) cgbn_sub(env, r, r, n);
  }
  __device__ __forceinline__ void add(bn_t &r, const bn_t &a, const bn_t &b) {
    int32_t c = cgbn_add(env, r, a, b);
    if (c || cgbn_compare(env, r, n) >= 0) cgbn_sub(env, r, r, n);
  }
  __device__ __forceinline__ void sub(bn_t &r, const bn_t &a, const bn_t &b) {
    int32_t br = cgbn_sub(env, r, a, b);
    if (br) cgbn_add(env, r, r, n);
  }
  __device__ __forceinline__ void neg(bn_t &r, const bn_t &a) {
    if (cgbn_equals_ui32(env, a, 0)) cgbn_set(env, r, a);
    else cgbn_sub(env, r, n, a);
  }
  __device__ __forceinline__ bool is_zero(const bn_t &a) { return cgbn_equals_ui32(env, a, 0); }

  struct Pt { bn_t X, Y, Z, T; };

  __device__ __forceinline__ void set_neutral(Pt &P, const bn_t &one) {
    cgbn_set_ui32(env, P.X, 0);
    cgbn_set(env, P.Y, one);
    cgbn_set(env, P.Z, one);
    cgbn_set_ui32(env, P.T, 0);
  }
  __device__ __forceinline__ void pneg(Pt &P) {
    neg(P.X, P.X);
    neg(P.T, P.T);
  }

  /* The two addition formulas, up to the last four products: X3 = e f,
     Y3 = g h, T3 = e h, Z3 = f g. If MIXED, Q.Z = 1 and is not read.
     Dedicated ("add-2008-hwcd-2", 9M + 1 product by a; 8M + 1 if MIXED):
     x3 = (x1y1 + x2y2) / (y1y2 + a x1x2), y3 = (x1y1 - x2y2) / (x1y2 - y1x2);
     gives Z3 = 0 for P = Q and when P - Q has order 2 or 4.
     Unified ("add-2008-hwcd", 2M more): also doubles. */
  template <bool MIXED>
  __device__ __forceinline__ void add_dedicated(bn_t &e, bn_t &f, bn_t &g, bn_t &h, const Pt &P, const Pt &Q,
                                                const bn_t &a) {
    bn_t m1, m2, m3, m4;
    mul(m1, P.X, Q.X);                // A = X1 X2
    mul(m2, P.Y, Q.Y);                // B = Y1 Y2
    mul(m3, P.Z, Q.T);                // C = Z1 T2
    if (MIXED) cgbn_set(env, m4, P.T);
    else mul(m4, P.T, Q.Z);           // D = T1 Z2
    add(e, m4, m3);                   // E = D + C
    sub(h, m4, m3);                   // H = D - C
    sub(f, P.X, P.Y);
    add(g, Q.X, Q.Y);
    mul(f, f, g);
    add(f, f, m2);
    sub(f, f, m1);                    // F = (X1 - Y1)(X2 + Y2) + B - A
    mul(m1, a, m1);
    add(g, m2, m1);                   // G = B + a A
  }
  template <bool MIXED>
  __device__ __forceinline__ void add_unified(bn_t &e, bn_t &f, bn_t &g, bn_t &h, const Pt &P, const Pt &Q,
                                              const bn_t &a, const bn_t &d) {
    bn_t m1, m2, m3, m4;
    mul(m1, P.X, Q.X);                // A = X1 X2
    mul(m2, P.Y, Q.Y);                // B = Y1 Y2
    mul(m3, P.T, Q.T);
    mul(m3, m3, d);                   // C = d T1 T2
    if (MIXED) cgbn_set(env, m4, P.Z);
    else mul(m4, P.Z, Q.Z);           // D = Z1 Z2
    add(e, P.X, P.Y);
    add(f, Q.X, Q.Y);
    mul(e, e, f);
    sub(e, e, m1);
    sub(e, e, m2);                    // E = (X1 + Y1)(X2 + Y2) - A - B
    sub(f, m4, m3);                   // F = D - C
    add(g, m4, m3);                   // G = D + C
    mul(m1, a, m1);
    sub(h, m2, m1);                   // H = B - a A
  }

  /* R = P + Q (R may alias P or Q): the dedicated formula, and the unified
     one where that gives Z3 = 0 (rare). If MIXED, Q.Z = 1 and is not read. */
  template <bool MIXED>
  __device__ __forceinline__ void padd(Pt &R, const Pt &P, const Pt &Q, const bn_t &a, const bn_t &d) {
    bn_t e, f, g, h, z;
    add_dedicated<MIXED>(e, f, g, h, P, Q, a);
    mul(z, f, g);
    if (is_zero(z)) {
      add_unified<MIXED>(e, f, g, h, P, Q, a, d);
      mul(z, f, g);
    }
    cgbn_set(env, R.Z, z);
    mul(R.X, e, f);
    mul(R.Y, g, h);
    mul(R.T, e, h);
  }

  /* R = 2P ("dbl-2008-hwcd", 4M + 4S + 1 product by a) */
  __device__ void pdbl(Pt &R, const Pt &P, const bn_t &a) {
    bn_t A, B, C, D, e, f, g, h;
    mul(A, P.X, P.X);
    mul(B, P.Y, P.Y);
    mul(C, P.Z, P.Z);
    add(C, C, C);
    mul(D, a, A);
    add(e, P.X, P.Y);
    mul(e, e, e);
    sub(e, e, A);
    sub(e, e, B);
    add(g, D, B);
    sub(f, g, C);
    sub(h, D, B);
    mul(R.X, e, f);
    mul(R.Y, g, h);
    mul(R.T, e, h);
    mul(R.Z, f, g);
  }
};
