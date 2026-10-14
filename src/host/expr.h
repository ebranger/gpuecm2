/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#pragma once
#include <string>
#include <gmp.h>

/* Evaluates an integer expression in the syntax GMP-ECM accepts for input
   numbers (and writes to save files):
     + - * / %   (. is accepted for *), unary minus, ^ (right associative),
     n!  n!m (multifactorial), n#  n#m (primorial of primes in [m, n]),
     grouping with () [] {}, functions Phi(n,x) and GCD(a,b) (case-insensitive).
   Returns true on success; on failure returns false and sets *err.
   Not (yet) supported: U(), primU(), PhiL(), PhiM(). */
bool eval_expression(mpz_t result, const std::string &text, std::string *err);
