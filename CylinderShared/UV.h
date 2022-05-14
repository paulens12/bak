#pragma once

#ifdef CUDA
#include "cuda_runtime.h"
#endif

#ifdef CUDA
__device__
#endif
void calcPoint(double* uOutput, double* vOutput, double* oOutput, double* uInput, double* vInput, double* oInput, double ro, int r, int f, int z);

#ifdef CUDA
__device__
#endif
void calcBoundary(double* u, double* v, double* o, int thrn_z = 0, int thread_n_z = 1, int thrn_f = 0, int thread_n_f = 1);

#ifdef CUDA
__device__
#endif
double calcBoundaryO(double* oInput, double u, double ro, int r, int f);