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
     RNS arrays: X[(q*rows + row)*T + i] for prime q, polynomial row, index i. */
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
  const uint32_t *rinv;                      // [P] 2^-32 mod p
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
  bool init(const mpz_t n, size_t max_operand_len, int want_logT, std::string *err);

  const RnsTables &tables() const { return t_; }
  int P() const { return t_.P; }
  int nlimbs() const { return t_.nlimbs; }
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
  /* ... with source row r starting at coefficient src(r) of A */
  void from_limbs_map(const uint32_t *A, size_t ld, const RowMap &src, size_t rows, size_t src_len, size_t piece_len,
                      size_t npieces, uint32_t *X, size_t T, size_t dst_offset, bool reverse,
                      cudaStream_t s = 0) const;
  /* X[row][pos] += c (mod each prime), c a small signed constant, for all rows */
  void add_const(uint32_t *X, size_t rows, size_t T, size_t pos, int c, cudaStream_t s = 0) const;

  /* Out := coefficients lo..lo+cnt-1 of each row of X, mapped back mod n.
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
  RnsTables t_;
  std::vector<uint32_t> primes_;
  std::vector<void *> allocs_;
  template <class T> const T *upload(const std::vector<T> &v);
};
