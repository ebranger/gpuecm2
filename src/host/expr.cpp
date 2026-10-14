/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "host/expr.h"

#include <cctype>
#include <stdexcept>
#include <vector>

#include "host/mpz_class_lite.h"

namespace {

struct ParseError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class Parser {
 public:
  explicit Parser(const std::string &s) : s_(s) {}

  void parse(mpz_t out) {
    Z v = expr();
    skip_ws();
    if (pos_ != s_.size()) fail("unexpected character '" + std::string(1, s_[pos_]) + "'");
    mpz_set(out, v.get());
  }

 private:
  const std::string &s_;
  size_t pos_ = 0;

  [[noreturn]] void fail(const std::string &msg) {
    throw ParseError(msg + " at position " + std::to_string(pos_));
  }
  void skip_ws() {
    while (pos_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[pos_]))) pos_++;
  }
  bool accept(char c) {
    skip_ws();
    if (pos_ < s_.size() && s_[pos_] == c) { pos_++; return true; }
    return false;
  }
  bool peek_digit() {
    skip_ws();
    return pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]));
  }

  // expr := term (('+'|'-') term)*
  Z expr() {
    Z v = term();
    for (;;) {
      if (accept('+')) { Z w = term(); mpz_add(v.get(), v.get(), w.get()); }
      else if (accept('-')) { Z w = term(); mpz_sub(v.get(), v.get(), w.get()); }
      else return v;
    }
  }
  // term := signed (('*'|'.'|'/'|'%') signed)*
  Z term() {
    Z v = signed_factor();
    for (;;) {
      if (accept('*') || accept('.')) { Z w = signed_factor(); mpz_mul(v.get(), v.get(), w.get()); }
      else if (accept('/')) {
        Z w = signed_factor();
        if (mpz_sgn(w.get()) == 0) fail("division by zero");
        if (!mpz_divisible_p(v.get(), w.get())) fail("inexact division");  // as GMP-ECM
        mpz_divexact(v.get(), v.get(), w.get());
      } else if (accept('%')) {
        Z w = signed_factor();
        if (mpz_sgn(w.get()) == 0) fail("modulo by zero");
        mpz_tdiv_r(v.get(), v.get(), w.get());
      } else return v;
    }
  }
  // signed := '-' signed | power      (so -2^2 == -(2^2))
  Z signed_factor() {
    if (accept('-')) { Z v = signed_factor(); mpz_neg(v.get(), v.get()); return v; }
    if (accept('+')) return signed_factor();
    return power();
  }
  // power := postfix ['^' signed]   (right associative)
  Z power() {
    Z b = postfix();
    if (accept('^')) {
      Z e = signed_factor();
      if (mpz_sgn(e.get()) < 0 || !mpz_fits_ulong_p(e.get())) fail("bad exponent");
      mpz_pow_ui(b.get(), b.get(), mpz_get_ui(e.get()));
    }
    return b;
  }
  // postfix := primary ('!' [number] | '#' [number])*
  Z postfix() {
    Z v = primary();
    for (;;) {
      if (accept('!')) {
        unsigned long step = 1;
        if (peek_digit()) step = small_number();
        v = multifactorial(v, step);
      } else if (accept('#')) {
        unsigned long lo = 2;
        if (peek_digit()) lo = small_number();
        v = primorial(v, lo);
      } else return v;
    }
  }
  Z primary() {
    skip_ws();
    if (pos_ >= s_.size()) fail("unexpected end of expression");
    char c = s_[pos_];
    if (c == '(' || c == '[' || c == '{') {
      char close = c == '(' ? ')' : c == '[' ? ']' : '}';
      pos_++;
      Z v = expr();
      if (!accept(close)) fail(std::string("expected '") + close + "'");
      return v;
    }
    if (std::isdigit(static_cast<unsigned char>(c))) return number();
    if (std::isalpha(static_cast<unsigned char>(c))) return function();
    fail("unexpected character '" + std::string(1, c) + "'");
  }
  Z number() {
    size_t start = pos_;
    while (pos_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[pos_]))) pos_++;
    Z v;
    if (mpz_set_str(v.get(), s_.substr(start, pos_ - start).c_str(), 10) != 0) fail("bad number");
    return v;
  }
  unsigned long small_number() {
    Z v = number();
    if (!mpz_fits_ulong_p(v.get())) fail("number too large");
    return mpz_get_ui(v.get());
  }
  Z function() {
    size_t start = pos_;
    while (pos_ < s_.size() && std::isalnum(static_cast<unsigned char>(s_[pos_]))) pos_++;
    std::string name = s_.substr(start, pos_ - start);
    for (auto &ch : name) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (!accept('(')) fail("expected '(' after " + name);
    std::vector<Z> args;
    args.push_back(expr());
    while (accept(',')) args.push_back(expr());
    if (!accept(')')) fail("expected ')'");
    if (name == "gcd") {
      if (args.size() != 2) fail("GCD takes 2 arguments");
      Z r; mpz_gcd(r.get(), args[0].get(), args[1].get()); return r;
    }
    if (name == "phi") {
      if (args.size() != 2) fail("Phi takes 2 arguments");
      return cyclotomic(args[0], args[1]);
    }
    fail("unsupported function " + name + "()");
  }

  Z multifactorial(const Z &n, unsigned long step) {
    if (mpz_sgn(n.get()) < 0 || !mpz_fits_ulong_p(n.get())) fail("bad factorial argument");
    unsigned long m = mpz_get_ui(n.get());
    Z r; mpz_set_ui(r.get(), 1);
    if (step == 0) fail("bad multifactorial step");
    if (step == 1) { mpz_fac_ui(r.get(), m); return r; }
    for (unsigned long i = m; i >= 1; i -= step) {
      mpz_mul_ui(r.get(), r.get(), i);
      if (i <= step) break;
    }
    return r;
  }
  Z primorial(const Z &n, unsigned long lo) {
    if (mpz_sgn(n.get()) < 0 || !mpz_fits_ulong_p(n.get())) fail("bad primorial argument");
    unsigned long hi = mpz_get_ui(n.get());
    Z r; mpz_set_ui(r.get(), 1);
    Z p; mpz_set_ui(p.get(), lo > 2 ? lo - 1 : 1);
    for (;;) {
      mpz_nextprime(p.get(), p.get());
      if (mpz_cmp_ui(p.get(), hi) > 0) break;
      mpz_mul(r.get(), r.get(), p.get());
    }
    return r;
  }
  // Phi_n(x) = prod_{d | n} (x^(n/d) - 1)^mu(d)
  Z cyclotomic(const Z &nz, const Z &x) {
    if (mpz_sgn(nz.get()) <= 0 || !mpz_fits_ulong_p(nz.get())) fail("bad Phi order");
    unsigned long n = mpz_get_ui(nz.get());
    std::vector<unsigned long> primes;
    unsigned long m = n;
    for (unsigned long p = 2; p * p <= m; p++)
      if (m % p == 0) { primes.push_back(p); while (m % p == 0) m /= p; }
    if (m > 1) primes.push_back(m);
    Z num, den, t;
    mpz_set_ui(num.get(), 1); mpz_set_ui(den.get(), 1);
    for (unsigned long mask = 0; mask < (1UL << primes.size()); mask++) {
      unsigned long d = 1; int bits = 0;
      for (size_t i = 0; i < primes.size(); i++) if (mask >> i & 1) { d *= primes[i]; bits++; }
      mpz_pow_ui(t.get(), x.get(), n / d);
      mpz_sub_ui(t.get(), t.get(), 1);
      if (mpz_sgn(t.get()) == 0) fail("Phi(n,x) undefined for this x");
      mpz_mul((bits & 1) ? den.get() : num.get(), (bits & 1) ? den.get() : num.get(), t.get());
    }
    mpz_divexact(num.get(), num.get(), den.get());
    return num;
  }
};

}  // namespace

bool eval_expression(mpz_t result, const std::string &text, std::string *err) {
  try {
    Parser(text).parse(result);
    return true;
  } catch (const ParseError &e) {
    if (err) *err = e.what();
    return false;
  }
}
