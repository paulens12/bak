#pragma once
#include <iostream>

#ifndef CUDA
#define __constant__ /**/
#define __device__ /**/
#else
#include "cuda_runtime.h"
#endif

__constant__ const double Du = 0.1;
__constant__ const double chi = 8.3;
__constant__ const double au = 1;
__constant__ const double Bv = 0.73;
__constant__ const double dt = 0.00005;
__constant__ const double gamma_o = 0.025;
__constant__ const double Do = 0.2;
__constant__ const double o0 = 1;

// cartesian
__constant__ const int X = 64;
__constant__ const int Y = 64;
__constant__ const int Z_cube = 64;
__constant__ const int XY = X * Y;
__constant__ const double dx2 = 0.125 * 0.125;
__constant__ const double dy2 = 0.125 * 0.125;
__constant__ const double dz2_cube = 0.125 * 0.125;

// polar
#define R_abs 4
#define Z_abs 8
#define R 40  /* %4 == 0 */
#define F 224 /* %8 == 0 */
#define Z 64
#ifdef CUDA
	__constant__ double dr = 0.0;
	__constant__ double hdr = 0.0;
	__constant__ double dr2 = 0.0;
	__constant__ double BASE_df = 0.0;
	__constant__ double dz2 = 0.0;

	void initializeCudaConstants() {
		double l_dr = (double)R_abs / R;
		std::cout << l_dr << std::endl;
		double l_hdr = l_dr / 2;
		double l_dr2 = l_dr * l_dr;
		double df = 2 * 3.14159265358979323846 / F;
		cudaMemcpyToSymbol(dr, &l_dr, sizeof(double));
		cudaMemcpyToSymbol(hdr, &l_hdr, sizeof(double));
		cudaMemcpyToSymbol(dr2, &l_dr2, sizeof(double));
		cudaMemcpyToSymbol(BASE_df, &df, sizeof(double));

		double dz = (double)Z_abs / Z;
		double l_dz2 = dz * dz;
		cudaMemcpyToSymbol(dz2, &l_dz2, sizeof(double));

		l_dr = 0;
		std::cout << l_dr << std::endl;
		cudaMemcpyFromSymbol(&l_dr, dr, sizeof(double));
		std::cout << l_dr << std::endl;
	}
#else
	const double dr = (double)R_abs / R;
	const double hdr = dr / 2;
	const double dr2 = dr * dr;
	const double BASE_df = 2 * 3.14159265358979323846 / F;
	const double dz = (double)Z_abs / Z;
	const double dz2 = dz * dz;
#endif
