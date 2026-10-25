/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING.

   gpuecm2: ECM stage 2 on the GPU for residues saved by GMP-ECM.
   Usage: gpuecm2 [options] -resume <file> B1 [B2 | B2min-B2max]
   Output and parameter choices follow GMP-ECM's `ecm -resume`. */
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "api/stage2.h"
#include "gpu/cgbn_sizes.h"
#include "host/curve.h"
#include "host/params.h"
#include "host/resume.h"
#include "host/version.h"

namespace {

int verbose = 1;  // 0 quiet, 1 normal, 2 verbose (-v), ...
bool print_curve = false;  // testing aid: print the Montgomery A computed from sigma
bool one = false;          // -one: stop with a number once a factor is found
bool selfcheck = false;    // -selfcheck: verify the stage 2 arithmetic of every curve
bool tree_cache = false;   // -treecache: keep the transforms of F's product tree
int rns_store = -1;        // -rns / -no-rns: RNS storage of the polynomials always / never (-1: automatic)
long batch_opt = 0;        // -batch n: curves per GPU run (0: automatic)
bool spin = false;         // -spin: busy-wait for the GPU

/* Output of one residue, collected while its batch is being formed and
   printed once its stage 2 is done, so that the output is in file order. */
struct Text {
  std::string out, err;
  void add(std::string &dst, const char *fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);
    int len = gmp_vsnprintf(nullptr, 0, fmt, ap2);
    va_end(ap2);
    if (len <= 0) return;
    std::string buf((size_t)len + 1, '\0');
    gmp_vsnprintf(&buf[0], buf.size(), fmt, ap);
    buf.resize((size_t)len);
    dst += buf;
  }
  void printf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    add(out, fmt, ap);
    va_end(ap);
  }
  void eprintf(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    add(err, fmt, ap);
    va_end(ap);
  }
  void flush() {
    fputs(out.c_str(), stdout);
    fflush(stdout);
    if (!err.empty()) { fputs(err.c_str(), stderr); fflush(stderr); }
    out.clear();
    err.clear();
  }
};

void usage(FILE *f) {
  fprintf(f,
          "Usage: gpuecm2 [options] -resume <file> B1 [B2 | B2min-B2max]\n"
          "Runs ECM stage 2 on the GPU for residues saved by GMP-ECM (ecm -save / -gpu -save).\n"
          "B1 must not exceed the B1 the residues were saved at.\n"
          "Numbers of up to %d bits (the GPUECM2_MAX_BITS of this build; at most 32768).\n"
          "Options:\n"
          "  -v            more verbose output (repeatable; -v -v -v adds per-phase times,\n"
          "                which makes step 2 slightly slower)\n"
          "  -q            quiet\n"
          "  -k n          number of stage 2 blocks (as GMP-ECM)\n"
          "  -power n      use x^n for Brent-Suyama's extension\n"
          "  -dickson n    use Dickson polynomial of degree n\n"
          "  -maxmem n     limit memory to n MB (GMP-ECM's CPU memory model)\n"
          "  -ntt, -no-ntt force or forbid GMP-ECM's NTT parameter rule\n"
          "  -one          stop with a number once a factor has been found\n"
          "  -batch n      process up to n residues of the same number together on the GPU\n"
          "                (default: automatic; 1 turns batching off)\n"
          "  -spin         busy-wait for the GPU (uses a full CPU core; by default the\n"
          "                CPU thread blocks, and sleeps on native Linux; under WSL2 the\n"
          "                driver polls, which keeps a core busy either way)\n"
          "  -selfcheck    verify every curve's stage 2 at random points, and a few roots\n"
          "                against a CPU computation (5-20%% slower; catches hardware\n"
          "                errors and bugs)\n"
          "  -treecache    keep the transforms of F's product tree for the evaluation:\n"
          "                stage 2 2-6%% faster for numbers of up to 1920 bits (measured at\n"
          "                GMP-ECM's default parameters), but more GPU memory, per curve\n"
          "                of a batch (GMP-ECM's default B2):\n"
          "                  bits    B1=1e6   3e6    11e6    43e6    11e7    26e7\n"
          "                  512       8 MB  39 MB   84 MB  182 MB  390 MB  832 MB\n"
          "                  1024     16 MB  76 MB  166 MB  357 MB  765 MB  1.6 GB\n"
          "                  1536     24 MB 114 MB  247 MB  532 MB  1.1 GB  2.4 GB\n"
          "                at most what is free after the rest of stage 2; no effect\n"
          "                above 1920 bits\n"
          "  -rns, -no-rns hold the polynomials as residues modulo NTT primes (RNS storage)\n"
          "                or as limbs, for all sizes (default: RNS storage from %d bits\n"
          "                on, where it is faster; the results are the same)\n",
          (int)GPUECM2_MAX_BITS, (int)STAGE2_RNS_AUTO_BITS);
}

/* B2 bounds: integer, or floating point like 7.7e11 (finite, not negative) */
bool parse_bound(Z &z, const std::string &s) {
  if (s.empty()) return false;
  bool digits = true;
  for (char c : s) if (!std::isdigit(static_cast<unsigned char>(c))) digits = false;
  if (digits) return mpz_set_str(z.get(), s.c_str(), 10) == 0;
  char *end;
  double d = std::strtod(s.c_str(), &end);
  if (*end != 0 || !std::isfinite(d) || d < 0) return false;
  mpz_set_d(z.get(), d);
  return true;
}

/* GMP-ECM's largest B1 (ecm-ecm.h: MAX_B1 = 2^53 - 1) */
const double MAX_B1 = 9007199254740991.0;
/* largest degree of the Brent-Suyama polynomial accepted (GMP-ECM chooses at most 30) */
const long MAX_DEGREE = 1000;

/* option values: the whole string must be a number in [lo, hi] */
bool parse_long(const char *s, long lo, long hi, long *v) {
  char *end;
  errno = 0;
  const long x = std::strtol(s, &end, 10);
  if (end == s || *end != 0 || errno != 0 || x < lo || x > hi) return false;
  *v = x;
  return true;
}
bool parse_double(const char *s, double lo, double hi, double *v) {
  char *end;
  const double x = std::strtod(s, &end);
  if (end == s || *end != 0 || !std::isfinite(x) || x < lo || x > hi) return false;
  *v = x;
  return true;
}

unsigned exact_digits(const mpz_t n) {
  char *s = mpz_get_str(nullptr, 10, n);
  unsigned d = (unsigned)strlen(s) - (s[0] == '-');
  void (*freefunc)(void *, size_t);
  mp_get_memory_functions(nullptr, nullptr, &freefunc);
  freefunc(s, strlen(s) + 1);
  return d;
}

bool is_plain_decimal(const std::string &s) {
  if (s.empty()) return false;
  for (char c : s) if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  return true;
}

void print_input_number(Text &o, const ResumeRecord &r) {
  const unsigned MAX_PRINT = 1000;
  unsigned nd = exact_digits(r.n.get());
  if (!r.n_text.empty() && !is_plain_decimal(r.n_text) && r.n_text.size() < MAX_PRINT)
    o.printf("Input number is %s (%u digits)\n", r.n_text.c_str(), nd);
  else if (nd < MAX_PRINT)
    o.printf("Input number is %Zd (%u digits)\n", r.n.get(), nd);
  else {
    Z t, u;
    mpz_ui_pow_ui(u, 5, nd - 10);
    mpz_tdiv_q_2exp(t, r.n.get(), nd - 10);
    mpz_tdiv_q(t, t, u);
    o.printf("Input number is %Zd...", t.get());
    mpz_ui_pow_ui(u, 10, 10);
    mpz_tdiv_r(t, r.n.get(), u);
    o.printf("%Zd (%u digits)\n", t.get(), nd);
  }
}

unsigned digits(const mpz_t z) { return exact_digits(z); }

/* GMP-ECM style report of a factor found in a stage; returns the exit code bits */
int report_factor(Text &o, const mpz_t f, const ResumeRecord &rec, unsigned step) {
  if (verbose >= 1) o.printf("********** Factor found in step %u: %Zd\n", step, f);
  else o.printf("%Zd\n", f);
  if (mpz_cmp(f, rec.n.get()) == 0) {
    if (verbose >= 1) o.printf("Found input number N\n");
    return 8;
  }
  Z cof;
  mpz_divexact(cof.get(), rec.n.get(), f);
  const bool fp = mpz_probab_prime_p(f, 30) > 0, cp = mpz_probab_prime_p(cof.get(), 30) > 0;
  if (verbose >= 1) {
    o.printf("Found %s factor of %u digits: %Zd\n", fp ? "prime" : "composite", digits(f), f);
    o.printf("%s cofactor %Zd has %u digits\n", cp ? "Probable prime" : "Composite", cof.get(), digits(cof.get()));
  }
  return fp ? (cp ? 14 : 6) : (cp ? 10 : 2);
}

/* State carried from one residue to the next (GMP-ECM semantics: for
   consecutive lines with the same N, factors found earlier are divided out;
   stop once the cofactor is a probable prime, or a factor was found with
   -one). */
struct NState {
  Z last_n, last_fac;
  bool last_prp = false;
};

/* A residue prepared for stage 2 (or already finished: kind != STAGE2). */
struct Job {
  enum Kind { SKIP, DONE, STAGE2 } kind = SKIP;
  ResumeRecord orig;  // as read (to redo it later)
  ResumeRecord rec;   // N with earlier factors divided out
  bool new_n = false; // starts a new N (resets the state when emitted)
  Z A;
  Stage2Params sp;
  Text text;          // output so far
  int status = 0;     // exit status contribution
  Z fac;              // factor found before stage 2 (kind DONE)
  bool fac_found = false;
};

/* Everything before stage 2, without side effects on the state. */
Job prepare(const ResumeRecord &orig, const NState &st, const Stage2Request &req) {
  Job J;
  J.orig = orig;
  J.rec = orig;
  ResumeRecord &rec = J.rec;
  Text &o = J.text;
  if (rec.method != METHOD_ECM) {
    o.eprintf("Line %lu: only ECM residues are supported (P-1/P+1 not yet), skipping\n", rec.line_number);
    return J;
  }
  if (rec.etype != ETYPE_MONTGOMERY) {
    o.eprintf("Line %lu: only Montgomery curves are supported (ETYPE=%d), skipping\n", rec.line_number, rec.etype);
    return J;
  }
  if (mpz_cmp(rec.n.get(), st.last_n.get()) == 0) {
    if (st.last_prp || (one && mpz_cmp_ui(st.last_fac.get(), 1) != 0)) return J;
    if (mpz_cmp_ui(st.last_fac.get(), 1) > 0) {
      mpz_divexact(rec.n.get(), rec.n.get(), st.last_fac.get());
      rec.n_text.clear();  // print the cofactor itself
      mpz_mod(rec.x.get(), rec.x.get(), rec.n.get());
    }
  } else {
    J.new_n = true;
  }
  if (verbose >= 1) {
    o.printf("Resuming ECM residue ");
    if (!rec.program.empty() || !rec.who.empty() || !rec.time.empty()) o.printf("saved ");
    if (!rec.who.empty()) o.printf("by %s ", rec.who.c_str());
    if (!rec.program.empty()) o.printf("with %s ", rec.program.c_str());
    if (!rec.time.empty()) o.printf("on %s ", rec.time.c_str());
    if (!rec.comment.empty()) o.printf("(%s)", rec.comment.c_str());
    o.printf("\n");
    print_input_number(o, rec);
  }
  J.kind = Job::DONE;
  if (req.B1 > rec.B1) {
    o.eprintf("Error: residue was saved at B1=%.0f; continuing stage 1 to B1=%.0f is not supported\n", rec.B1,
              req.B1);
    J.status = 1;
    return J;
  }
  if (mpz_cmp_ui(rec.n.get(), 1) <= 0 || mpz_even_p(rec.n.get())) {
    o.eprintf("Error: the number must be odd and greater than 1 (remove factors of 2 first), skipping this "
              "residue\n");
    J.status = 1;
    return J;
  }
  if (mpz_sizeinbase(rec.n.get(), 2) > GPUECM2_MAX_BITS) {
    o.eprintf("Error: the number has %zu bits; gpuecm2 supports at most %d%s, skipping this residue\n",
              mpz_sizeinbase(rec.n.get(), 2), GPUECM2_MAX_BITS,
              GPUECM2_MAX_BITS < 32768 ? " (build with -DGPUECM2_MAX_BITS=32768 for up to 32768)" : "");
    J.status = 1;
    return J;
  }

  /* curve: A from sigma (or given A), point x from the file */
  Z x0, fac;
  if (rec.have_sigma) {
    int r = curve_from_param(fac, J.A, x0, rec.param, rec.sigma, rec.n);
    if (r == CURVE_ERROR) {
      o.eprintf("Line %lu: invalid sigma/param, skipping\n", rec.line_number);
      J.status = 1;
      return J;
    }
  } else {
    mpz_mod(J.A, rec.A.get(), rec.n.get());
  }
  if (print_curve) o.printf("CURVE line=%lu A=0x%Zx\n", rec.line_number, J.A.get());

  Stage2Params &sp = J.sp;
  std::string err;
  /* GMP-ECM's parameters, unless they need more GPU memory than is free:
     then use more blocks (smaller dF), which still covers at least B2 */
  Stage2Params sp0;
  if (!compute_stage2_params(sp0, req, rec.n, &err)) {
    o.printf("%s\n", err.c_str());
    J.status = 1;
    return J;
  }
  const size_t budget = sp0.dF && sp0.k ? stage2_gpu_budget() : 0;
  bool raised = false;
  if (!stage2_params_for_gpu(sp, req, rec.n, budget, &raised, &err)) {
    o.eprintf("%s\n", err.c_str());
    J.status = 1;
    return J;
  }
  if (raised && verbose >= 1)
    o.printf("Note: GPU memory (%.1f GB usable) is too small for dF=%lu, k=%lu; using dF=%lu, k=%lu\n",
             budget / 1e9, sp0.dF, sp0.k, sp.dF, sp.k);
  if (verbose >= 1) {
    o.printf("Using B1=%.0f-%.0f, ", rec.B1, req.B1);
    /* GMP-ECM prints the user's B2 if one was given, else the effective one */
    const mpz_srcptr b2shown = mpz_cmp_si(req.B2.get(), -1) != 0 ? req.B2.get() : sp.B2.get();
    if (mpz_sgn(req.B2min.get()) < 0) o.printf("B2=%Zd", b2shown);
    else o.printf("B2=%Zd-%Zd", sp.B2min.get(), b2shown);
    if (sp.S > 0) o.printf(", polynomial x^%d", sp.S);
    else o.printf(", polynomial Dickson(%d)", -sp.S);
    if (rec.have_sigma) o.printf(", sigma=%d:%Zd\n", rec.param, rec.sigma.get());
    else o.printf(", A=%Zd\n", rec.A.get());
  }
  if (verbose >= 2) {
    o.printf("dF=%lu, k=%lu, d=%lu, d2=%lu, i0=%Zd\n", sp.dF, sp.k, sp.d1, sp.d2, sp.i0.get());
    if (sp.dF && sp.k) {
      const size_t one = stage2_gpu_bytes(rec.n, sp, 1), two = stage2_gpu_bytes(rec.n, sp, 2);
      o.printf("GPU memory estimate %.1f MB for one curve, %.1f MB per further curve\n", one / 1048576.,
               (two - one) / 1048576.);
    }
  }

  Z x(rec.x), y, a(J.A);
  mpz_set_ui(y, 1);
  if (montgomery_to_weierstrass(fac, x, y, a, rec.n) == CURVE_FACTOR_FOUND) {
    mpz_set(J.fac, fac);
    J.fac_found = true;
    return J;
  }
  if (sp.dF == 0 || sp.k == 0) return J;  // B2 < B2min: nothing to do
  {  /* B y^2 = x^3 + A x^2 + x is singular when A^2 = 4 (GMP-ECM would run on it; the
        Edwards model used here does not exist then) */
    Z t, g;
    mpz_mul(t, J.A, J.A);
    mpz_sub_ui(t, t, 4);
    mpz_gcd(g, t, rec.n);
    if (mpz_cmp_ui(g, 1) != 0) {
      o.eprintf("Line %lu: the curve is singular (A^2 = 4 modulo %Zd), skipping\n", rec.line_number, g.get());
      J.status = 1;
      return J;
    }
  }
  if (getenv("GPUECM2_PARAMS_ONLY")) return J;  // testing aid: print the parameters, no stage 2
  J.kind = Job::STAGE2;
  return J;
}

bool same_params(const Stage2Params &a, const Stage2Params &b) {
  return a.dF == b.dF && a.k == b.k && a.d1 == b.d1 && a.d2 == b.d2 && a.S == b.S &&
         mpz_cmp(a.i0.get(), b.i0.get()) == 0;
}

}  // namespace

int main(int argc, char **argv) {
  Stage2Request req;
  const char *resume_file = nullptr;
  std::string b1_arg, b2_arg;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto need = [&](const char *opt) -> const char * {
      if (i + 1 >= argc) { fprintf(stderr, "Option %s requires an argument\n", opt); exit(1); }
      return argv[++i];
    };
    auto bad = [&](const char *opt, const char *what) {
      fprintf(stderr, "Invalid value for %s: %s (%s)\n", opt, argv[i], what);
      exit(1);
    };
    long lv;
    double dv;
    if (a == "-v") verbose++;
    else if (a == "-q") verbose = 0;
    else if (a == "-k") {
      if (!parse_long(need("-k"), 1, 1000000000L, &lv)) bad("-k", "a number of blocks from 1 to 10^9");
      req.k = (unsigned long)lv;
    } else if (a == "-power" || a == "-dickson") {
      if (!parse_long(need(a.c_str()), 1, MAX_DEGREE, &lv)) bad(a.c_str(), "a degree from 1 to 1000");
      req.S = a == "-power" ? (int)lv : -(int)lv;
    } else if (a == "-maxmem") {
      if (!parse_double(need("-maxmem"), 0, 1e12, &dv)) bad("-maxmem", "megabytes, 0 or more");
      req.maxmem = dv * 1048576.;
    }
    else if (a == "-ntt") req.use_ntt = 2;
    else if (a == "-no-ntt") req.use_ntt = 0;
    else if (a == "-resume") resume_file = need("-resume");
    else if (a == "-printcurve") print_curve = true;
    else if (a == "-one") one = true;
    else if (a == "-selfcheck") selfcheck = true;
    else if (a == "-treecache") tree_cache = true;
    else if (a == "-rns") rns_store = 1;
    else if (a == "-no-rns") rns_store = 0;
    else if (a == "-batch") {
      if (!parse_long(need("-batch"), 0, 1000000L, &lv)) bad("-batch", "a number of residues, 0 for automatic");
      batch_opt = lv;
    }
    else if (a == "-spin") spin = true;
    else if (a == "-h" || a == "--help") { usage(stdout); return 0; }
    else if (!a.empty() && a[0] == '-' && !std::isdigit(static_cast<unsigned char>(a[1]))) {
      fprintf(stderr, "Unknown option: %s\n", a.c_str()); usage(stderr); return 1;
    } else if (b1_arg.empty()) b1_arg = a;
    else if (b2_arg.empty()) b2_arg = a;
    else { fprintf(stderr, "Too many arguments\n"); usage(stderr); return 1; }
  }
  if (!resume_file || b1_arg.empty()) { usage(stderr); return 1; }

  if (!parse_double(b1_arg.c_str(), 0, MAX_B1, &req.B1) || req.B1 <= 0) {
    fprintf(stderr, "Invalid B1: %s (a positive number up to %.0f)\n", b1_arg.c_str(), MAX_B1);
    return 1;
  }
  if (!b2_arg.empty()) {
    /* "B2" or "B2min-B2max" (a '-' after the first character separates them) */
    size_t dash = b2_arg.find('-', 1);
    bool ok = dash == std::string::npos
                  ? parse_bound(req.B2, b2_arg)
                  : parse_bound(req.B2min, b2_arg.substr(0, dash)) && parse_bound(req.B2, b2_arg.substr(dash + 1));
    if (!ok) { fprintf(stderr, "Invalid B2: %s\n", b2_arg.c_str()); return 1; }
  }

  FILE *f = strcmp(resume_file, "-") == 0 ? stdin : fopen(resume_file, "r");
  if (!f) { fprintf(stderr, "Could not open file %s for reading\n", resume_file); return 1; }

  stage2_cpu_wait(spin);
  if (rns_store >= 0) stage2_set_rns_store(rns_store != 0);
  if (verbose >= 1)
    printf("gpuecm2 %s [GMP %s] [ECM stage 2]\n", gpuecm2_version(), gmp_version);

  int status = 0;
  NState st;
  std::unique_ptr<Stage2Engine> engine;
  Z engine_n;
  Stage2Params engine_sp;
  size_t engine_batch = 1;
  ResumeReader reader(f);
  std::deque<ResumeRecord> queue;  // read, not yet processed (or to be redone)
  std::vector<Job> batch;          // residues waiting for a common stage 2

  /* prints a finished job and updates the state */
  auto emit = [&](Job &J, const Stage2Outcome *res, const std::string &pre) {
    if (J.kind == Job::SKIP) {
      J.text.flush();
      return;
    }
    if (J.new_n) {
      mpz_set(st.last_n.get(), J.orig.n.get());
      mpz_set_ui(st.last_fac.get(), 1);
      st.last_prp = false;
    }
    Text &o = J.text;
    o.out += pre;
    auto found = [&](const mpz_t fct, unsigned step) {
      status = report_factor(o, fct, J.rec, step);
      if (mpz_cmp(fct, J.rec.n.get()) != 0) {
        mpz_mul(st.last_fac.get(), st.last_fac.get(), fct);
        st.last_prp = (status & 8) != 0;
      }
    };
    if (!res) {  // finished before stage 2
      if (J.fac_found) found(J.fac.get(), 1);
      else status = J.status;
      o.flush();
      return;
    }
    const Stage2Times &t = res->t;
    const double per = 1e3 / (double)t.batch;  // ms per curve
    if (verbose >= 4) {
      o.printf("Computing roots of F took %.0fms\n", t.rootsF * per);
      if (res->have_first || t.treeF > 0) {
        o.printf("Building F from its roots took %.0fms\n", t.treeF * per);
        o.printf("Computing 1/F took %.0fms\n", t.invF * per);
        o.printf("Computing roots of G took %.0fms\n", t.rootsG * per);
        o.printf("Building G from its roots took %.0fms\n", t.treesG * per);
        o.printf("Computing G * H mod F took %.0fms\n", t.mulmod * per);
        o.printf("Computing polyeval(F,G) took %.0fms\n", t.eval * per);
        o.printf("Computing product of all F(g_i) took %.0fms\n", t.prod * per);
      }
    }
    if (verbose >= 1) o.printf("Step 2 took %.0fms\n", t.total * per);
    if (res->status == 2) {
      found(res->factor.get(), 2);
    } else if (res->status < 0) {
      if (res->check_failed)
        o.eprintf("Error: the stage 2 self-check failed (hardware error or bug); this curve's result is invalid\n");
      else
        o.eprintf("Error in stage 2\n");
      status = 1;
    } else {
      if (verbose >= 3 && res->have_first) o.printf("Product of G(f_i) = %Zd\n", res->first_value.get());
      status = 0;
    }
    o.flush();
  };

  /* runs the batch, prints its jobs in order; once a factor is found the
     rest of the batch is redone (on the cofactor, as GMP-ECM would) */
  auto flush_batch = [&]() {
    if (batch.empty()) return;
    std::string pre;
    const Job &J0 = batch[0];
    if (!engine || mpz_cmp(engine_n.get(), J0.rec.n.get()) != 0 || !same_params(engine_sp, J0.sp)) {
      engine.reset();
      engine.reset(new Stage2Engine(J0.rec.n.get(), J0.sp));
      engine->set_tree_cache(tree_cache);
      if (!engine->ok()) {
        fprintf(stderr, "Error: %s\n", engine->error().c_str());
        engine.reset();
        for (Job &J : batch) {
          J.status = 1;
          J.kind = Job::DONE;
          emit(J, nullptr, "");
        }
        batch.clear();
        return;
      }
      mpz_set(engine_n.get(), J0.rec.n.get());
      engine_sp = J0.sp;
      if (verbose >= 2) {
        char buf[160];
        snprintf(buf, sizeof buf, "Using %d NTT primes of 31 bits, NTT length up to 2^%d%s\n", engine->rns().P(),
                 engine->rns().logTmax(), engine->rns().rns_store() ? ", RNS storage" : "");
        pre = buf;
      }
    }
    std::vector<MontgomeryCurve> C(batch.size());
    for (size_t i = 0; i < batch.size(); i++) {
      mpz_set(C[i].n.get(), batch[i].rec.n.get());
      mpz_set(C[i].A.get(), batch[i].A.get());
      mpz_set(C[i].u.get(), batch[i].rec.x.get());
    }
    /* per-phase times need synchronisation between the phases, which also
       stops root generation from overlapping the polynomial arithmetic: only
       at -v -v -v */
    std::vector<Stage2Outcome> res = engine->run_batch(C.data(), C.size(), verbose >= 4, selfcheck);
    if (verbose >= 2 && batch.size() > 1) {
      char buf[160];
      snprintf(buf, sizeof buf, "Stage 2 of %zu residues together took %.0fms\n", batch.size(),
               res[0].t.total * 1e3);
      pre += buf;
    }
    size_t i = 0;
    for (; i < batch.size(); i++) {
      emit(batch[i], &res[i], i == 0 ? pre : std::string());
      if (res[i].status == 2) break;
    }
    for (size_t j = batch.size(); j-- > i + 1;) queue.push_front(batch[j].orig);  // redo the rest
    batch.clear();
  };

  for (;;) {
    if (queue.empty()) {
      ResumeRecord rec;
      if (!reader.next(rec)) {
        flush_batch();
        if (queue.empty()) break;
        continue;
      }
      queue.push_back(rec);
    }
    if (!batch.empty() && (batch.size() >= engine_batch || mpz_cmp(queue.front().n.get(), batch[0].orig.n.get()) != 0)) {
      flush_batch();
      continue;
    }
    ResumeRecord orig = queue.front();
    queue.pop_front();
    Job J = prepare(orig, st, req);
    if (J.kind == Job::STAGE2 && (batch.empty() || same_params(J.sp, batch[0].sp))) {
      if (batch.empty()) {  // batch size for this number and these parameters
        const size_t budget = stage2_gpu_budget();
        engine_batch = batch_opt > 0 ? stage2_max_batch(J.rec.n.get(), J.sp, budget, (size_t)batch_opt)
                                     : stage2_auto_batch(J.rec.n.get(), J.sp, budget);
      }
      batch.push_back(std::move(J));
      continue;
    }
    if (!batch.empty()) {  // finish the batch first (the state may change)
      queue.push_front(orig);
      flush_batch();
      continue;
    }
    if (J.kind == Job::STAGE2) {
      batch.push_back(std::move(J));
      continue;
    }
    emit(J, nullptr, "");
  }
  if (f != stdin) fclose(f);
  return status;
}
