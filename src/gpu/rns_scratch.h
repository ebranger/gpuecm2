/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#pragma once
#include <cstddef>

/* GPU scratch memory of one RNS -> limbs conversion for large moduli (the
   two-pass kernels of rns.cu, from 576 limbs on), at most; counted in
   stage2_gpu_bytes. (A header of its own: rns.h is included by every CGBN
   size's translation unit.) */
constexpr size_t RNS_TO_LIMBS_SCRATCH = (size_t)128 << 20;
/* ... and of the import / export of RNS storage (RnsContext::import_limbs) */
constexpr size_t RNS_IMEX_SCRATCH = (size_t)64 << 20;
