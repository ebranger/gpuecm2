/* Copyright 2026 Erik Branger.
   Parsing rules follow read_resumefile_line() in GMP-ECM's resume.c
   (Copyright 2001-2023 Paul Zimmermann, Alexander Kruppa and Cyril Bouvier,
   MIT license).
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#pragma once
#include <cstdio>
#include <string>

#include "host/mpz_class_lite.h"

enum { METHOD_ECM = 0, METHOD_PM1 = 1, METHOD_PP1 = 2 };

/* GMP-ECM parametrizations and curve types (values as in ecm.h) */
enum {
  PARAM_SUYAMA = 0, PARAM_BATCH_SQUARE = 1, PARAM_BATCH_2 = 2, PARAM_BATCH_32BITS_D = 3
};
enum { ETYPE_MONTGOMERY = 1, ETYPE_WEIERSTRASS = 2 };

struct ResumeRecord {
  int method = METHOD_ECM;
  int param = PARAM_SUYAMA;  // default for old save files
  int etype = ETYPE_MONTGOMERY;
  bool have_sigma = false, have_A = false, have_y = false;
  Z sigma, A, x, y, x0, y0;
  double B1 = 0;
  std::string n_text;  // N as written in the file (may be an expression)
  Z n;
  std::string program, who, time, comment;
  unsigned long line_number = 0;
};

/* Reads save lines one at a time. Malformed lines are skipped with a message
   on stderr, exactly like GMP-ECM does. */
class ResumeReader {
 public:
  explicit ResumeReader(FILE *f) : f_(f) {}
  /* Returns false at end of file. */
  bool next(ResumeRecord &rec);

 private:
  FILE *f_;
  unsigned long line_no_ = 0;
  bool read_line(std::string &line);
  bool parse_line(const std::string &line, ResumeRecord &rec);
};
