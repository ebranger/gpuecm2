/* Copyright 2026 Erik Branger.
   Parsing rules follow read_resumefile_line() in GMP-ECM's resume.c
   (Copyright 2001-2023 Paul Zimmermann, Alexander Kruppa and Cyril Bouvier,
   MIT license).
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#include "host/resume.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "host/expr.h"

namespace {

const unsigned long CHKSUMMOD = 4294967291UL;

std::string trim(const std::string &s) {
  size_t a = 0, b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) a++;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) b--;
  return s.substr(a, b - a);
}

/* mpz_inp_str(.., 0)-like: optional sign, 0x/0X hex, leading 0 octal, else decimal */
bool parse_mpz(Z &z, const std::string &v) {
  std::string t = trim(v);
  if (t.empty()) return false;
  return mpz_set_str(z.get(), t.c_str(), 0) == 0;
}

}  // namespace

bool ResumeReader::read_line(std::string &line) {
  line.clear();
  int c;
  bool any = false;
  while ((c = fgetc(f_)) != EOF) {
    any = true;
    if (c == '\n') break;
    if (c == '\r') {  // "\r\n" or lone "\r"
      int d = fgetc(f_);
      if (d != '\n' && d != EOF) ungetc(d, f_);
      break;
    }
    line.push_back(static_cast<char>(c));
  }
  if (any) line_no_++;
  return any;
}

bool ResumeReader::next(ResumeRecord &rec) {
  std::string line;
  while (read_line(line)) {
    if (line.empty() || line[0] == '#') continue;
    ResumeRecord r;
    r.line_number = line_no_;
    if (parse_line(line, r)) {
      rec = std::move(r);
      return true;
    }
  }
  return false;
}

bool ResumeReader::parse_line(const std::string &line, ResumeRecord &rec) {
  bool have_method = false, have_x = false, have_n = false, have_b1 = false,
       have_z = false, have_qx = false, have_checksum = false;
  unsigned long saved_checksum = 0;
  Z z;

  size_t pos = 0;
  while (pos < line.size()) {
    while (pos < line.size() && line[pos] == ' ') pos++;
    if (pos >= line.size()) break;
    size_t eq = line.find('=', pos);
    if (eq == std::string::npos) {
      printf("Resume warning, skipping line with no '=' after: %s\n", line.substr(pos, 15).c_str());
      return false;
    }
    std::string tag = line.substr(pos, eq - pos);
    if (tag.size() > 15) tag.resize(15);
    size_t semi = line.find(';', eq + 1);
    bool had_semi = semi != std::string::npos;
    std::string val = line.substr(eq + 1, (had_semi ? semi : line.size()) - eq - 1);
    pos = had_semi ? semi + 1 : line.size();

    bool ok = true;
    if (tag == "METHOD") {
      std::string m = trim(val);
      if (m == "ECM") rec.method = METHOD_ECM;
      else if (m == "P-1") rec.method = METHOD_PM1;
      else if (m == "P+1") rec.method = METHOD_PP1;
      else ok = false;
      have_method = ok;
    } else if (tag == "X") { ok = parse_mpz(rec.x, val); have_x = ok; }
    else if (tag == "Y") { ok = parse_mpz(rec.y, val); rec.have_y = ok; }
    else if (tag == "Z") { ok = parse_mpz(z, val); have_z = ok; }
    else if (tag == "QX") { ok = parse_mpz(rec.x, val); have_qx = ok; }
    else if (tag == "X0") ok = parse_mpz(rec.x0, val);
    else if (tag == "Y0") ok = parse_mpz(rec.y0, val);
    else if (tag == "CHECKSUM") {
      char *end; std::string t = trim(val);
      saved_checksum = std::strtoul(t.c_str(), &end, 10);
      ok = !t.empty() && *end == 0;
      have_checksum = ok;
    } else if (tag == "COMMENT") rec.comment = val;
    else if (tag == "N") {
      rec.n_text = trim(val);
      std::string err;
      ok = eval_expression(rec.n.get(), rec.n_text, &err);
      if (!ok) fprintf(stderr, "Cannot evaluate N=%s: %s\n", rec.n_text.c_str(), err.c_str());
      have_n = ok;
    } else if (tag == "SIGMA") { ok = parse_mpz(rec.sigma, val); rec.have_sigma = ok; }
    else if (tag == "PARAM") { char *end; std::string t = trim(val); rec.param = (int)std::strtol(t.c_str(), &end, 10); ok = !t.empty() && *end == 0; }
    else if (tag == "ETYPE") { char *end; std::string t = trim(val); rec.etype = (int)std::strtol(t.c_str(), &end, 10); ok = !t.empty() && *end == 0; }
    else if (tag == "A") { ok = parse_mpz(rec.A, val); rec.have_A = ok; }
    else if (tag == "B1") {
      char *end; std::string t = trim(val);
      rec.B1 = std::strtod(t.c_str(), &end);
      ok = !t.empty() && *end == 0 && std::isfinite(rec.B1) && rec.B1 >= 0;
      have_b1 = ok;
    }
    else if (tag == "PROGRAM") rec.program = val;
    else if (tag == "WHO") rec.who = val;
    else if (tag == "TIME") rec.time = val;
    else {
      printf("Save file line has unknown tag: %s\n", tag.c_str());
      return false;
    }
    if (!ok) {
      fprintf(stderr, "Save file line has an invalid %s value, skipping\n", tag.c_str());
      return false;
    }
    /* Prime95 lines have no semicolon after SIGMA */
    if (!had_semi && !(have_qx && have_n && rec.have_sigma)) {
      printf("%s field not followed by semicolon\n", tag.c_str());
      return false;
    }
  }

  /* Prime95 v22 lines: no METHOD field, QX= instead of X= */
  if (have_qx) {
    if (have_n && rec.have_sigma) {
      if (mpz_cmp_ui(rec.n.get(), 1) <= 0) {
        fprintf(stderr, "Save file line has N <= 1, skipping\n");
        return false;
      }
      rec.method = METHOD_ECM;
      rec.program = "Prime95";
      mpz_mod(rec.x.get(), rec.x.get(), rec.n.get());
      return true;
    }
    return false;
  }

  if (!have_method || !have_x || !have_n || !have_b1 ||
      (rec.method == METHOD_ECM && !rec.have_sigma && !rec.have_A)) {
    fprintf(stderr, "Save file line lacks fields\n");
    return false;
  }

  if (mpz_cmp_ui(rec.n.get(), 1) <= 0) {
    fprintf(stderr, "Save file line has N <= 1, skipping\n");
    return false;
  }
  if (rec.have_sigma && rec.method == METHOD_ECM && (rec.param < 0 || rec.param > 3)) {
    fprintf(stderr, "Save file line has PARAM=%d; only parametrizations 0-3 are supported, skipping\n", rec.param);
    return false;
  }

  if (have_checksum) {
    Z c;
    mpz_set_d(c.get(), rec.B1);
    if (rec.have_sigma) mpz_mul_ui(c.get(), c.get(), mpz_fdiv_ui(rec.sigma.get(), CHKSUMMOD));
    if (rec.have_A) mpz_mul_ui(c.get(), c.get(), mpz_fdiv_ui(rec.A.get(), CHKSUMMOD));
    mpz_mul_ui(c.get(), c.get(), mpz_fdiv_ui(rec.n.get(), CHKSUMMOD));
    mpz_mul_ui(c.get(), c.get(), mpz_fdiv_ui(rec.x.get(), CHKSUMMOD));
    if (have_z) mpz_mul_ui(c.get(), c.get(), mpz_fdiv_ui(z.get(), CHKSUMMOD));
    mpz_mul_ui(c.get(), c.get(), (unsigned long)(rec.param + 1) % CHKSUMMOD);
    unsigned long expect = mpz_fdiv_ui(c.get(), CHKSUMMOD);
    if (expect != saved_checksum) {
      fprintf(stderr, "Resume file line has bad checksum %lu, expected %lu\n", saved_checksum, expect);
      return false;
    }
  }

  mpz_mod(rec.x.get(), rec.x.get(), rec.n.get());
  if (rec.have_y) mpz_mod(rec.y.get(), rec.y.get(), rec.n.get());
  if (have_z) {
    if (!mpz_invert(z.get(), z.get(), rec.n.get()))
      printf("Oops, factor found while reading from save file.\n");
    mpz_mul(z.get(), z.get(), rec.x.get());
    mpz_mod(rec.x.get(), z.get(), rec.n.get());
  }
  return true;
}
