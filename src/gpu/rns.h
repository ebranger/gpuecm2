/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   Residue number system (RNS) used for polynomial arithmetic over Z/nZ:
   coefficients mod n are mapped to residues modulo P NTT primes
   p = c*2^k + 1 with 2^28 < p < 2^31, polynomials are multiplied with NTTs
   modulo each prime, and results are mapped back with an exact CRT.

   Device layouts:
     residues mod n ("limb arrays"): limb j of coefficient g at A[j*ld + g],
       32-bit limbs, nlimbs = ceil(bits(n)/32); g = row*len + i for batches.
     RNS arrays: X[(q*rows + row)*T + i] for prime q, polynomial row, index i.

   Two ways of storing the coefficients of polynomials (the "limb arrays" of
   the polynomial code, coef_words() 32-bit words per coefficient, same layout):
     limbs (default): the residue mod n in nlimbs limbs, fully reduced;
     RNS storage (init with rns_store): the primes are split into a base A (the
       first PA) and a base B (the last PB), and a coefficient c is stored as
       the residues mod the primes of B of an x = c M_A mod n (Montgomery form,
       M_A the product of the primes of A), 0 <= x < 64 n. Products then need
       no conversion to limbs: a base extension B -> A before the transform
       ("from_limbs") and a Montgomery reduction after it (q mod A, extension
       A -> B, r = (z + q n) / M_A in B; "to_limbs"), about half the work of
       the two conversions. The constant 1 is stored as M_A mod n. Values
       given to or taken from the GPU code outside the polynomial arithmetic
       (roots, final values) are converted with import_limbs / export_limbs. */
#pragma once
#include <cstddef>
#include <cstdint>
#include <climits>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <gmp.h>

/* Where row r of a batch of rows starts in a limb array: rows come in groups
   of grows rows (e.g. the nodes of one curve's tree level), consecutive rows
   rstride apart and consecutive groups gstride apart; row0 is added to r
   (for processing a batch in chunks). */
struct RowMap {
  size_t rstride = 0, grows = SIZE_MAX, gstride = 0, row0 = 0;
  static RowMap dense(size_t stride) { RowMap m; m.rstride = stride; return m; }
  static RowMap grouped(size_t rstride, size_t grows, size_t gstride, size_t row0 = 0) {
    RowMap m;
    m.rstride = rstride; m.grows = grows ? grows : 1; m.gstride = gstride; m.row0 = row0;
    return m;
  }
  RowMap from(size_t r0) const { RowMap m = *this; m.row0 += r0; return m; }
  __host__ __device__ size_t operator()(size_t r) const {
    const size_t g = row0 + r;
    return grows == SIZE_MAX ? g * rstride : (g / grows) * gstride + (g % grows) * rstride;
  }
};

struct RnsTables {  // device pointers, passed to kernels by value
  int P = 0, nlimbs = 0, logTmax = 0;
  const uint32_t *p, *pinv, *k32, *k64, *v;  // [P]
  const uint32_t *rinv;                      // [P] 2^-32 one^2 mod p (the wrapped lead)
  const uint64_t *mu;                        // [P] floor(2^64/p)
  const uint32_t *pow32;                     // [P][nlimbs]  2^(32j) mod p
  const uint32_t *Mi;                        // [P][nlimbs]  (M/p) mod n
  const uint32_t *n, *nmM;                   // [nlimbs]     n, n - (M mod n)
  double ntop;                               // n / 2^(32*(nlimbs-1))
  const uint32_t *crt, *crtp;                // [logTmax+1][2][P] CRT constants (+Shoup)
  /* NTT twiddles: w = omega_4096^t, t < 2048 (+Shoup), forward and inverse;
     twist tables in Montgomery form: TL = omega_Tmax^t (t < 2048),
     TH = omega_Tmax^(2048 h) (h < Tmax/2048) */
  const uint32_t *w, *wp, *iw, *iwp;         // [P][2048]
  const uint32_t *TL, *TH, *iTL, *iTH;       // [P][2048], [P][Tmax/2048]
  /* the representation of the constant 1 in RNS arrays: 1 (limbs) or M_A mod n */
  const uint32_t *one;                       // [P]
  /* RNS storage (PA = PB = 0 with limbs); base A = primes 0..PA-1, B = PA..P-1 */
  int PA = 0, PB = 0, cw = 0;                // cw: words per stored coefficient (nlimbs or PB)
  const uint32_t *eBA;                       // [PA][PB] (M_B / b_j) mod a_i
  const uint32_t *ibB, *ibBp;                // [PB] (M_B / b_j)^-1 mod b_j (+Shoup)
  const uint32_t *mBA;                       // [PA] M_B mod a_i
  const uint32_t *eAB;                       // [PB][PA] (M_A / a_i) mod b_j
  const uint32_t *sA, *sAp;                  // [logTmax+1][2][PA] scale * (-n^-1) * (M_A/a_i)^-1 mod a_i (+Shoup)
  const uint32_t *sB, *sBp;                  // [logTmax+1][2][PB] scale mod b_j (+Shoup)
  const uint32_t *mAB, *nB, *n16B;           // [PB] M_A, n, 16 n mod b_j
  const uint32_t *iAB, *iABp;                // [PB] M_A^-1 mod b_j (+Shoup)
  const uint32_t *cimp, *cexp;               // [P] (M_A^2 mod n), (M_A^-1 mod n) mod p_q: import / export
};

class RnsContext {
 public:
  RnsContext() = default;
  ~RnsContext();
  RnsContext(const RnsContext &) = delete;
  RnsContext &operator=(const RnsContext &) = delete;

  /* Prepares primes and tables for modulus n. Products of polynomials whose
     shorter factor has at most max_operand_len coefficients are exact.
     want_logT is the largest NTT length (log2) wanted; a smaller one is used
     if there are not enough suitable primes. */
  bool init(const mpz_t n, size_t max_operand_len, int want_logT, std::string *err, bool rns_store = false);

  const RnsTables &tables() const { return t_; }
  int P() const { return t_.P; }
  int nlimbs() const { return t_.nlimbs; }
  bool rns_store() const { return t_.PB > 0; }
  /* 32-bit words per coefficient of the polynomial code's arrays */
  int coef_words() const { return t_.cw; }
  /* stored coefficients from / to plain limb arrays (fully reduced residues
     mod n; with limbs a copy): count coefficients, leading dimensions lda, ldo */
  void import_limbs(const uint32_t *A, size_t lda, size_t count, uint32_t *Out, size_t ldo, cudaStream_t s = 0) const;
  void export_limbs(const uint32_t *A, size_t lda, size_t count, uint32_t *Out, size_t ldo, cudaStream_t s = 0) const;
  int logTmax() const { return t_.logTmax; }
  const std::vector<uint32_t> &primes() const { return primes_; }

  /* X (rows x T per prime) := residues of rows polynomials of len coefficients
     from limb array A (leading dimension ld); positions len..T-1 are zeroed. */
  void from_limbs(const uint32_t *A, size_t ld, size_t rows, size_t len, uint32_t *X, size_t T,
                  cudaStream_t s = 0) const {
    from_limbs_pieces(A, ld, rows, len, len, 1, X, T, s);
  }
  /* As from_limbs, but each source row (src_len coefficients) is cut into
     npieces pieces of piece_len coefficients (the last one zero-padded);
     piece i of row r becomes RNS row r*npieces + i. */
  void from_limbs_pieces(const uint32_t *A, size_t ld, size_t rows, size_t src_len, size_t piece_len,
                         size_t npieces, uint32_t *X, size_t T, cudaStream_t s = 0) const {
    from_limbs_ex(A, ld, rows, src_len, piece_len, npieces, X, T, 0, false, s);
  }
  /* Most general form: piece coefficient t goes to RNS position dst_offset + t,
     or dst_offset + piece_len - 1 - t if reverse; all other positions are 0. */
  void from_limbs_ex(const uint32_t *A, size_t ld, size_t rows, size_t src_len, size_t piece_len, size_t npieces,
                     uint32_t *X, size_t T, size_t dst_offset, bool reverse, cudaStream_t s = 0) const {
    from_limbs_map(A, ld, RowMap::dense(src_len), rows, src_len, piece_len, npieces, X, T, dst_offset, reverse, s);
  }
  /* ... with source row r starting at coefficient src(r) of A; with
     lead_pos >= 0, position lead_pos of every RNS row is lead_c (or gets
     lead_c added, if it holds a coefficient). With pad = false the
     positions outside [dst_offset, dst_offset + piece_len) are left as they
     are (for ntt_forward_padded, which treats them as zero; a lead_pos
     outside that range is then ignored too). */
  void from_limbs_map(const uint32_t *A, size_t ld, const RowMap &src, size_t rows, size_t src_len, size_t piece_len,
                      size_t npieces, uint32_t *X, size_t T, size_t dst_offset, bool reverse,
                      cudaStream_t s = 0, long lead_pos = -1, int lead_c = 1, bool pad = true) const;
  /* X[row][pos] += c (mod each prime), c a small signed constant, for all rows */
  void add_const(uint32_t *X, size_t rows, size_t T, size_t pos, int c, cudaStream_t s = 0) const;

  /* (With RNS storage, "limbs" below means stored coefficients.)
     Out := coefficients lo..lo+cnt-1 of each row of X, mapped back mod n.
     The residues are first multiplied by 2^(32*mont) / 2^logT_scale, which
     undoes `mont` Montgomery multiplications and an unnormalised inverse NTT
     of length 2^logT_scale (use 0 for neither). Output coefficient
     row*cnt + (i - lo) goes to Out with leading dimension ld. */
  void to_limbs(const uint32_t *X, size_t rows, size_t T, size_t lo, size_t cnt, uint32_t *Out, size_t ld,
                int logT_scale, int mont, cudaStream_t s = 0) const {
    to_limbs_map(X, rows, T, lo, cnt, Out, ld, RowMap::dense(cnt), logT_scale, mont, s);
  }
  /* ... with output row r starting at coefficient dst(r) of Out */
  void to_limbs_map(const uint32_t *X, size_t rows, size_t T, size_t lo, size_t cnt, uint32_t *Out, size_t ld,
                    const RowMap &dst, int logT_scale, int mont, cudaStream_t s = 0) const;

 private:
  /* the conversions between limbs and RNS arrays (for every mode) */
  void limbs_to_rns(const uint32_t *A, size_t ld, const RowMap &src, size_t rows, size_t src_len, size_t piece_len,
                    size_t npieces, uint32_t *X, size_t T, size_t dst_offset, bool reverse, cudaStream_t s) const;
  void rns_to_limbs(const uint32_t *X, size_t rows, size_t T, size_t lo, size_t cnt, uint32_t *Out, size_t ld,
                    const RowMap &dst, int logT_scale, int mont, cudaStream_t s) const;
  /* RNS storage: base extension B -> A (all primes) and Montgomery reduction to B */
  void store_to_rns(const uint32_t *A, size_t ld, const RowMap &src, size_t rows, size_t src_len, size_t piece_len,
                    size_t npieces, uint32_t *X, size_t T, size_t dst_offset, bool reverse, cudaStream_t s) const;
  void rns_to_store(const uint32_t *X, size_t rows, size_t T, size_t lo, size_t cnt, uint32_t *Out, size_t ld,
                    const RowMap &dst, int logT_scale, int mont, cudaStream_t s) const;
  RnsTables t_;
  std::vector<uint32_t> primes_;
  std::vector<void *> allocs_;
  template <class T> const T *upload(const std::vector<T> &v);
};
