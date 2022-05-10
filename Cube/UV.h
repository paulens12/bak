#pragma once

#define GET(arr, x, y, z) (arr[(x) + (y) * X + (z) * XY])

void calcPoint(double* uOutput, double* vOutput, double* oOutput, double* uInput, double* vInput, double* oInput, int x, int y, int z);
void calcBoundary(double* u, double* v, double* o, int thrn = 0, int thread_n = 1);