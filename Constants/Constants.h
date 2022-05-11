#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

#ifndef __device__
#define __device__ /**/
#endif

__device__ const double Du = 0.1;
__device__ const double chi = 8.3;
__device__ const double au = 1;
__device__ const double Bv = 0.73;
__device__ const double dt = 0.00005;
__device__ const double gamma_o = 0.025;
__device__ const double Do = 0.2;
__device__ const double o0 = 1;

// cartesian
__device__ const int X = 80;
__device__ const int Y = 80;
__device__ const int Z = 80;
__device__ const int XY = X * Y;
__device__ const double dx2 = 0.075 * 0.075;
__device__ const double dy2 = 0.075 * 0.075;
__device__ const double dz2 = 0.075 * 0.075;

// polar
const int R = 40;
const int F = 200;
const double dr = 9 / (M_PI * R); /* 2pi(R*dr) = 360*dx, dx = 0.05 */
const double hdr = dr / 2;
const double dr2 = dr * dr;
//#define dr (0.04774648292756860073066512901175) /* 3/(20pi) */
//#define dr2 (0.00227972663195259985748728792222) /* dr * dr */
const double BASE_df = 2 * M_PI / F;