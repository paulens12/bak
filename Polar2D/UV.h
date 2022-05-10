#pragma once

void calcPoint(double* uOutput, double* vOutput, double* uInput, double* vInput, double ro, int r, int f);
void calcBoundary(double* u, double* v, int thrn = 0, int thread_n = 1);