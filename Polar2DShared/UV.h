#pragma once

#ifdef CUDA
#include "cuda_runtime.h"
#endif

#ifdef CUDA
__device__
#endif
void calcPoint(double* uOutput, double* vOutput, double* uInput, double* vInput, double ro, int r, int f);

#ifdef CUDA
__device__
#endif
void calcBoundary(double* u, double* v, int thrn = 0, int thread_n = 1);