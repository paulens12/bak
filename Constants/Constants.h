#pragma once

#define _USE_MATH_DEFINES
#include <math.h>

const double Du = 0.1;
const double chi = 8.3;
const double au = 1;
const double Bv = 0.73;
const double dt = 0.00005;
const double gamma_o = 0.025;
const double Do = 0.2;
const double o0 = 1;

// cartesian
const int X = 80;
const int Y = 80;
const int Z = 80;
const int XY = X * Y;
const double dx2 = 0.05 * 0.05;
const double dy2 = 0.05 * 0.05;
const double dz2 = 0.05 * 0.05;

// polar
const int R = 40;
const int F = 200;
const double dr = 9 / (M_PI * R); /* 2pi(R*dr) = 360*dx, dx = 0.05 */
const double hdr = dr / 2;
const double dr2 = dr * dr;
//#define dr (0.04774648292756860073066512901175) /* 3/(20pi) */
//#define dr2 (0.00227972663195259985748728792222) /* dr * dr */
const double BASE_df = 2 * M_PI / F;