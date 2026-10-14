/* Copyright 2026 Erik Branger.
   Ported from GMP-ECM's bestd.c, ecm.c (set_stage_2_params, choose_S),
   auxarith.c and stage2.c (memory_use): Copyright 2001-2012 Paul Zimmermann,
   Alexander Kruppa et al., GNU LGPL version 3 or later.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "host/params.h"

#include <cmath>

namespace {

/* GMP-ECM constants (ecm-impl.h, ecm-ecm.h, sp.h, x86_64 params.h) */
const double ECM_COST = 11.0 / 6.0;
const double DEFAULT_B2_EXPONENT = 1.43;
const unsigned NTT_SIZE_THRESHOLD = 30;  // in 64-bit limbs
const unsigned SP_NUMB_BITS = 62;        // 64-bit builds
const double MUL_NTT_THRESHOLD = 16384;
const double MPZSPV_NORMALISE_STRIDE = 512;

/* number of 64-bit limbs of n, as mpz_size() on a 64-bit-limb GMP */
size_t limbs64(const mpz_t n) { return (mpz_sizeinbase(n, 2) + 63) / 64; }

/* GMP-ECM's estimate of its own (CPU) stage 2 memory use, in bytes.
   Only used to reproduce GMP-ECM's parameter choice under -maxmem. */
double memory_use(unsigned long dF, unsigned sp_num, unsigned Ftreelvl, const mpz_t n) {
  double mem = 9.0;
  mem += (double)Ftreelvl;
  mem *= (double)dF;
  mem += 2. * (2.0 * dF);                              /* list_mul_mem(dF) = 2*dF */
  mem += (24.0 + 1.0) * (double)(sp_num ? std::min(MUL_NTT_THRESHOLD, (double)dF) : dF);
  mem *= (double)limbs64(n) * 8.0 + 16.0;              /* limbs + sizeof(mpz_t) */
  if (sp_num)
    mem += 4.0 * dF * sp_num * 8.0
         + MPZSPV_NORMALISE_STRIDE * ((double)sp_num * 8.0 + 6.0 * 8.0 + 4.0)
         + 3.0 * dF * sp_num * 8.0;
  return mem;
}

/* bestD: see the long comment in GMP-ECM's bestd.c */
bool bestD(Stage2Params &p, unsigned long k_req, const mpz_t B2min, mpz_t B2, bool po2,
           bool use_ntt, double maxmem, const mpz_t n, std::string *err) {
  static const unsigned int l[] = {12, 18, 30, 42, 60, 90, 120, 150, 210, 240, 270, 330, 420, 510, 630, 840, 1050, 1260, 1470, 1680, 1890, 2310, 2730, 3150, 3570, 3990, 4620, 5460, 6090, 6930, 8190, 9240, 10920, 12180, 13860, 16170, 18480, 20790, 23100, 30030, 34650, 39270, 43890, 48510, 60060, 66990, 78540, 90090, 99330, 120120, 133980, 150150, 180180, 210210, 240240, 270270, 300300, 334950, 371280, 420420, 510510, 570570, 600600, 630630, 746130, 870870, 1021020, 1141140, 1291290, 1531530, 1711710, 1891890, 2081310, 2312310, 2552550, 2852850, 3183180, 3573570, 3993990, 4594590, 5105100, 5705700, 6322470, 7147140, 7987980, 8978970, 10210200, 11741730, 13123110, 14804790, 16546530, 19399380, 21411390, 23993970, 26816790, 29609580, 33093060, 36606570, 40330290, 44414370, 49639590, 54624570, 60090030, 67897830, 77597520, 87297210, 96996900, 107056950, 118107990};
  static const unsigned int lpo2[] = {12, 30, 60, 120, 240, 510, 1020, 2310, 4620, 9240, 19110, 39270, 79170, 158340, 324870, 690690, 1345890, 2852850, 5705700, 11741730, 23130030, 48498450, 96996900};
  const unsigned long N = sizeof l / sizeof l[0], Npo2 = sizeof lpo2 / sizeof lpo2[0];

  if (mpz_cmp(B2, B2min) < 0) {  // no stage 2
    p.k = 0; p.dF = 0;
    return true;
  }
  Z i0, i1, j, t;
  unsigned long i, d1 = 0, d2 = 0, dF = 0, phid, k = k_req;
  unsigned long maxN = po2 ? Npo2 : N;

  if (maxmem != 0.) {
    for (i = 0; i < maxN; i++) {
      d1 = po2 ? lpo2[i] : l[i];
      phid = eulerphi(d1) / 2;
      dF = po2 ? 1UL << ceil_log2(phid) : phid;
      unsigned lg_dF = ceil_log2(dF), sp_num = 0;
      if (use_ntt) sp_num = (unsigned)((2 * mpz_sizeinbase(n, 2) + lg_dF) / SP_NUMB_BITS + 4);
      if (memory_use(dF, sp_num, lg_dF, n) > maxmem) break;
    }
    maxN = i;
  }

  for (i = 0; i < maxN; i++) {
    d1 = po2 ? lpo2[i] : l[i];
    phid = eulerphi(d1) / 2;
    dF = po2 ? 1UL << ceil_log2(phid) : phid;
    /* smallest prime 5 <= d2 < 25 (not 3) that does not divide d1 */
    for (d2 = 5; d2 < 25; d2 += 2) {
      if (d2 % 3 == 0) continue;
      if (d1 % d2 > 0) break;
    }
    if (d2 >= 25 || d2 - 1 > dF) d2 = 1;

    mpz_set_ui(i0, d1 - 1);
    mpz_mul_ui(i0, i0, d2);
    mpz_add(i1, B2, i0);            /* i1 = B2 + (d1 - 1) * d2 */
    mpz_sub(i0, B2min, i0);         /* i0 = B2min - (d1 - 1) * d2 */
    mpz_cdiv_q_ui(i0, i0, d1);
    mpz_fdiv_q_ui(i1, i1, d1);
    mpz_sub(j, i1, i0);
    mpz_add_ui(j, j, 1);
    if (d2 > 1) {
      mpz_fdiv_q_ui(t, i1, d2); mpz_sub(j, j, t);
      mpz_fdiv_q_ui(t, i0, d2); mpz_add(j, j, t);
    }
    mpz_cdiv_q_ui(j, j, dF);        /* number of blocks needed */
    if ((k != 0 && mpz_cmp_ui(j, k) <= 0) || (k == 0 && mpz_cmp_ui(j, po2 ? 6 : 2) <= 0)) break;
  }

  if (i == maxN) {
    if (k != 0) { *err = "Error: too large step 2 bound, increase -k"; return false; }
    if (!mpz_fits_ulong_p(j)) {
      *err = "Error: stage 2 interval too large, cannot generate suitable parameters.\nTry a smaller B2 value.";
      return false;
    }
    if (maxN == 0) { *err = "Error: stage 2 not possible with memory allowed by -maxmem."; return false; }
  }
  if (k == 0) k = mpz_get_ui(j);

  /* real i1 for k*dF roots of G starting at i0, skipping multiples of d2 */
  mpz_set_ui(j, k);
  mpz_mul_ui(j, j, dF);
  if (d2 == 1) {
    mpz_add(i1, i0, j);
    mpz_sub_ui(i1, i1, 1);
  } else {
    mpz_add(j, j, i0);
    mpz_cdiv_q_ui(t, i0, d2);
    mpz_sub(j, j, t);
    mpz_fdiv_qr_ui(j, t, j, d2 - 1);
    mpz_mul_ui(j, j, d2);
    mpz_add(i1, j, t);
  }
  p.d1 = d1; p.d2 = d2; p.dF = dF; p.k = k;
  mpz_set(p.i0, i0);
  /* B2' = (i1 - d2 + 1) * d1 + d2 - 1 */
  mpz_sub_ui(i1, i1, d2 - 1);
  mpz_mul_ui(B2, i1, d1);
  mpz_add_ui(B2, B2, d2 - 1);
  return true;
}

}  // namespace

unsigned long eulerphi(unsigned long n) {
  unsigned long phi = 1UL, p;
  for (p = 2UL; p * p <= n; p += 2UL) {
    if (n % p == 0UL) {
      phi *= p - 1UL;
      n /= p;
      while (n % p == 0UL) { phi *= p; n /= p; }
    }
    if (p == 2UL) p--;
  }
  return (n == 1UL) ? phi : phi * (n - 1UL);
}

unsigned int ceil_log2(unsigned long n) {
  unsigned int k = 0;
  n--;
  while (n) { k++; n >>= 1; }
  return k;
}

int choose_S(const mpz_t B2len) {
  if (mpz_cmp_d(B2len, 1e7) < 0) return 1;
  if (mpz_cmp_d(B2len, 1e8) < 0) return 2;
  if (mpz_cmp_d(B2len, 1e9) < 0) return -3;
  if (mpz_cmp_d(B2len, 1e10) < 0) return -6;
  if (mpz_cmp_d(B2len, 3e11) < 0) return -12;
  return -30;
}

bool compute_stage2_params(Stage2Params &out, const Stage2Request &req, const mpz_t n, std::string *err) {
  out.B1 = req.B1;
  /* GMP-ECM main.c: by default use NTT only for small input */
  out.use_ntt = req.use_ntt == 1 ? (limbs64(n) <= NTT_SIZE_THRESHOLD) : (req.use_ntt != 0);
  out.po2 = out.use_ntt;

  mpz_set(out.B2, req.B2.get());
  if (mpz_cmp_si(out.B2, -1) == 0)
    mpz_set_d(out.B2, std::pow(ECM_COST * req.B1, DEFAULT_B2_EXPONENT));
  mpz_set(out.B2min, req.B2min.get());
  if (mpz_sgn(out.B2min) < 0) mpz_set_d(out.B2min, req.B1);

  if (!bestD(out, req.k, out.B2min, out.B2, out.po2, out.use_ntt, req.maxmem, n, err)) return false;

  out.S = req.S;
  if (out.S == 0) {
    Z t;
    mpz_sub(t, out.B2, out.B2min);
    out.S = choose_S(t);
  }
  return true;
}
