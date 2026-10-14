/* Copyright 2026 Erik Branger.
   This file is part of gpuecm2, distributed under the GNU Lesser General Public
   License, version 3 or later. See COPYING.LESSER and COPYING. */
#pragma once
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    cudaError_t err_ = (call);                                                      \
    if (err_ != cudaSuccess) {                                                      \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),       \
              __FILE__, __LINE__, cudaGetErrorString(err_));                         \
      exit(3);                                                                      \
    }                                                                               \
  } while (0)
