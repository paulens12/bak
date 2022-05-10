﻿
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

#define Du 0.2
#define chi 8
#define au 1
#define Bv 0.73
#define R 40
#define F 160
#define dt 0.00005
const double H_dr = 9 / (M_PI * R); /* 2pi(R*dr) = 360*dx (cartesian sim) */
const double H_dr2 = H_dr * H_dr;
const double H_BASE_df = 2 * M_PI / F;

__constant__ double dr;
__constant__ double hdr;
__constant__ double dr2;
__constant__ double BASE_df;

#define FRAME_DURATION 6
//#define H 128
//#define L 3456000
#define L 72000
#define SNAPSHOT_STEP 3600

// fp - f+1
// fm - f-1
// rp - r+1
// rm - r-1
__device__
inline double getNextU(
	double u, double urp, double urm, double ufp, double ufm,
	double v, double vrp, double vrm, double vfp, double vfm,
	int r, double df
)
{
	double urp2 = (u + urp) / 2;
	double urm2 = (u + urm) / 2;
	double ufp2 = (u + ufp) / 2;
	double ufm2 = (u + ufm) / 2;
	//double ro = r * dr + dr;
	//double rp2 = r * dr + 3 * dr / 2;
	//double rm2 = r * dr + dr / 2;
	double ro = r * dr;
	double rp2 = ro + hdr;
	double rm2 = ro - hdr;

	return dt * (
		Du * (
			(rp2 * (urp - u) - rm2 * (u - urm)) / (ro * dr2)
			+ (ufp - 2 * u + ufm) / (ro * ro * df * df)
			) - chi * (
				(rp2 * urp2 * (vrp - v) - rm2 * urm2 * (v - vrm)) / (ro * dr2)
				+ (ufp2 * (vfp - v) - ufm2 * (v - vfm)) / (ro * ro * df * df)
				) + au * u * (1 - u)
		) + u;
}

__device__
inline double getNextV(
	double u,
	double v, double vrp, double vrm, double vfp, double vfm,
	int r, double df
)
{
	//double ro = r * dr + dr;
	//double rp2 = r * dr + 3 * dr / 2;
	//double rm2 = r * dr + dr / 2;
	double ro = r * dr;
	double rp2 = ro + hdr;
	double rm2 = ro - hdr;

	return dt * (
		(rp2 * (vrp - v) - rm2 * (v - vrm)) / (ro * dr2)
		+ (vfp - 2 * v + vfm) / (ro * ro * df * df)
		+ u / (1 + Bv * u) - v
		) + v;
}

__global__
void boundaryKernel(double* u, double* v)
{
	int fOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int fStride = blockDim.x * gridDim.x;

	for (int f = fOffset; f < F; f += fStride)
	{ // no-flux boundary condition
		u[3 * R * F / 16 + (R / 2 - 1) * F + f] = (4 * u[3 * R * F / 16 + (R / 2 - 2) * F + f] - u[3 * R * F / 16 + (R / 2 - 3) * F + f]) / 3;
		v[3 * R * F / 16 + (R / 2 - 1) * F + f] = (4 * v[3 * R * F / 16 + (R / 2 - 2) * F + f] - v[3 * R * F / 16 + (R / 2 - 3) * F + f]) / 3;
	}


	for (int f = fOffset; f < F / 8; f += fStride)
	{ // central symmetry
		u[F / 8 + f] = u[f] = (u[F / 4 + f] + u[3 * F / 8 + f]) / 2;
		v[F / 8 + f] = v[f] = (v[F / 4 + f] + v[3 * F / 8 + f]) / 2;
	}
}

__global__
void calcKernel(double* uOutput, double* vOutput, double* uInput, double* vInput)
{
	int rOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int rStride = blockDim.x * gridDim.x;
	int fOffset = threadIdx.y + blockIdx.y * blockDim.y;
	int fStride = blockDim.y * gridDim.y;

	double urp, urm, vrp, vrm, ufp, ufm, vfp, vfm, u, v;
	int fp, fm; // offsets
	double df;

	for (int r = rOffset; r < R / 4; r += rStride)
	{
		// central symmetry boundary condition - skip r = 0
		if(r==0)
			continue;
		df = BASE_df * 4;
		for (int f = fOffset; f < F / 4; f += fStride)
		{
			u = uInput[r * F / 4 + f];
			v = vInput[r * F / 4 + f];

			if (f == 0) fm = r * F / 4 + F / 4 - 1;
			else fm = r * F / 4 + f - 1;
			if (f == F / 4 - 1) fp = r * F / 4;
			else fp = r * F / 4 + f + 1;

			ufp = uInput[fp];
			vfp = vInput[fp];
			ufm = uInput[fm];
			vfm = vInput[fm];

			urm = uInput[(r - 1) * F / 4 + f];
			vrm = vInput[(r - 1) * F / 4 + f];

			if (r == R / 4 - 1)
			{
				urp = (uInput[R * F / 16 + 2 * f] + uInput[R * F / 16 + 2 * f + 1]) / 2;
				vrp = (vInput[R * F / 16 + 2 * f] + vInput[R * F / 16 + 2 * f + 1]) / 2;
			}
			else
			{
				urp = uInput[(r + 1) * F / 4 + f];
				vrp = vInput[(r + 1) * F / 4 + f];
			}

			uOutput[r * F / 4 + f] = getNextU(u, urp, urm, ufp, ufm, v, vrp, vrm, vfp, vfm, r, df);
			vOutput[r * F / 4 + f] = getNextV(u, v, vrp, vrm, vfp, vfm, r, df);
		}
	}

	for (int r = rOffset + R / 4; r < R / 2; r += rStride)
	{
		df = BASE_df * 2;
		for (int f = fOffset; f < F / 2; f += fStride)
		{
			u = uInput[R * F / 16 + (r - R / 4) * F / 2 + f];
			v = vInput[R * F / 16 + (r - R / 4) * F / 2 + f];

			if (f == 0) fm = R * F / 16 + (r - R / 4 + 1) * F / 2 - 1;
			else fm = R * F / 16 + (r - R / 4) * F / 2 + f - 1;
			if (f == F / 2 - 1) fp = R * F / 16 + (r - R / 4) * F / 2;
			else fp = R * F / 16 + (r - R / 4) * F / 2 + f + 1;

			ufp = uInput[fp];
			vfp = vInput[fp];
			ufm = uInput[fm];
			vfm = vInput[fm];

			if (r == R / 4)
			{
				urm = uInput[(r - 1) * F / 4 + f / 2] / 2;
				vrm = vInput[(r - 1) * F / 4 + f / 2] / 2;
			}
			else
			{
				urm = uInput[R * F / 16 + (r - 1 - R / 4) * F / 2 + f];
				vrm = vInput[R * F / 16 + (r - 1 - R / 4) * F / 2 + f];
			}

			if (r == R / 2 - 1)
			{
				urp = (uInput[3 * R * F / 16 + 2 * f] + uInput[3 * R * F / 16 + 2 * f + 1]) / 2;
				vrp = (vInput[3 * R * F / 16 + 2 * f] + vInput[3 * R * F / 16 + 2 * f + 1]) / 2;
			}
			else
			{
				urp = uInput[R * F / 16 + (r + 1 - R / 4) * F / 2 + f];
				vrp = vInput[R * F / 16 + (r + 1 - R / 4) * F / 2 + f];
			}

			uOutput[R * F / 16 + (r - R / 4) * F / 2 + f] = getNextU(u, urp, urm, ufp, ufm, v, vrp, vrm, vfp, vfm, r, df);
			vOutput[R * F / 16 + (r - R / 4) * F / 2 + f] = getNextV(u, v, vrp, vrm, vfp, vfm, r, df);
		}
	}

	for (int r = rOffset + R / 2; r < R - 1; r += rStride)
	{
		// no-flux boundary condition - skip r = R - 1
		for (int f = fOffset; f < F; f += fStride)
		{

			u = uInput[3 * R * F / 16 + (r - R / 2) * F + f];
			v = vInput[3 * R * F / 16 + (r - R / 2) * F + f];

			if (f == 0) fm = 3 * R * F / 16 + (r - R / 2 + 1) * F - 1;
			else fm = 3 * R * F / 16 + (r - R / 2) * F + f - 1;
			if (f == F - 1) fp = 3 * R * F / 16 + (r - R / 2) * F;
			else fp = 3 * R * F / 16 + (r - R / 2) * F + f + 1;

			ufp = uInput[fp];
			vfp = vInput[fp];
			ufm = uInput[fm];
			vfm = vInput[fm];

			if (r == R / 2)
			{
				urm = uInput[3 * R * F / 16 - F / 2 + f / 2] / 2;
				vrm = vInput[3 * R * F / 16 - F / 2 + f / 2] / 2;
			}
			else
			{
				urm = uInput[3 * R * F / 16 + (r - 1 - R / 2) * F + f];
				vrm = vInput[3 * R * F / 16 + (r - 1 - R / 2) * F + f];
			}

			urp = uInput[3 * R * F / 16 + (r + 1 - R / 2) * F + f];
			vrp = vInput[3 * R * F / 16 + (r + 1 - R / 2) * F + f];

			uOutput[3 * R * F / 16 + (r - R / 2) * F + f] = getNextU(u, urp, urm, ufp, ufm, v, vrp, vrm, vfp, vfm, r, BASE_df);
			vOutput[3 * R * F / 16 + (r - R / 2) * F + f] = getNextV(u, v, vrp, vrm, vfp, vfm, r, BASE_df);
		}
	}
}

int main()
{
	double* matrixU1, * matrixU2, * matrixV1, * matrixV2;
	int bufferlength = 11 * R * F / 16;
	int size = bufferlength * sizeof(double);
	double H_hdr = H_dr / 2;
	cudaError_t cudaErr;

	cudaErr = cudaMemcpyToSymbol(dr, &H_dr, sizeof(double));
	if (cudaErr != cudaSuccess)
		cout << "cudaMemcpyToSymbol: " << cudaGetErrorString(cudaErr) << endl;
	cudaErr = cudaMemcpyToSymbol(dr2, &H_dr2, sizeof(double));
	if (cudaErr != cudaSuccess)
		cout << "cudaMemcpyToSymbol: " << cudaGetErrorString(cudaErr) << endl;
	cudaErr = cudaMemcpyToSymbol(BASE_df, &H_BASE_df, sizeof(double));
	if (cudaErr != cudaSuccess)
		cout << "cudaMemcpyToSymbol: " << cudaGetErrorString(cudaErr) << endl;
	cudaErr = cudaMemcpyToSymbol(hdr, &H_hdr, sizeof(double));
	if (cudaErr != cudaSuccess)
		cout << "cudaMemcpyToSymbol: " << cudaGetErrorString(cudaErr) << endl;

	double* matrixUInit = new double[bufferlength];
	double* matrixVInit = new double[bufferlength];

	cudaErr = cudaMalloc(&matrixU1, size);
	if (cudaErr != cudaSuccess)
		cout << "cudaMalloc: " << cudaGetErrorString(cudaErr) << endl;
	cudaErr = cudaMalloc(&matrixU2, size);
	if (cudaErr != cudaSuccess)
		cout << "cudaMalloc: " << cudaGetErrorString(cudaErr) << endl;
	cudaErr = cudaMalloc(&matrixV1, size);
	if (cudaErr != cudaSuccess)
		cout << "cudaMalloc: " << cudaGetErrorString(cudaErr) << endl;
	cudaErr = cudaMalloc(&matrixV2, size);
	if (cudaErr != cudaSuccess)
		cout << "cudaMalloc: " << cudaGetErrorString(cudaErr) << endl;

	int deviceId;
	cudaErr = cudaGetDevice(&deviceId);
	if (cudaErr != cudaSuccess)
		cout << "cudaGetDevice: " << cudaGetErrorString(cudaErr) << endl;

	cudaErr = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	if (cudaErr != cudaSuccess)
		cout << "cudaDeviceSetCacheConfig: " << cudaGetErrorString(cudaErr) << endl;

	cout << size * (L / SNAPSHOT_STEP + 1) << endl;
	double* matrixU = (double*)malloc(size * (L / SNAPSHOT_STEP + 1)); // for gif output
	perror("malloc error");
	double* matrixV = (double*)malloc(size * (L / SNAPSHOT_STEP + 1));
	perror("malloc error");
	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<double> distr(0, 0.1);
	default_random_engine re(1);

	for (int i = 0; i < bufferlength; i++)
	{
		matrixU[i] = matrixUInit[i] = distr(re) + 1.0;
		matrixV[i] = matrixVInit[i] = 0;
	}

	cudaMemcpy(matrixU1, matrixUInit, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	cudaMemcpy(matrixV1, matrixVInit, size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	delete matrixUInit;
	delete matrixVInit;

	dim3 blocks(1, 10);
	dim3 threads(16, 10);

	cudaGetLastError();
	auto start = clock();
	double* temp = NULL;
	for (int i = 0; i < L; ++i)
	{
		if (i % SNAPSHOT_STEP == SNAPSHOT_STEP - 1)
		{
			cudaDeviceSynchronize();
			int step = i / SNAPSHOT_STEP + 1;
			//save frame
			cudaMemcpy(matrixU + step * bufferlength, matrixU1, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			cudaMemcpy(matrixV + step * bufferlength, matrixV1, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			double elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;
			cout << "step " << step << ", time elapsed: " << elapsed << ", avg: " << elapsed / step << endl;
		}

		calcKernel <<< blocks, threads >>> (matrixU2, matrixV2, matrixU1, matrixV1);
		cudaErr = cudaGetLastError();
		if (cudaErr != cudaSuccess)
			cout << "calcKernel: " << cudaGetErrorString(cudaErr) << endl;


		boundaryKernel <<< 10, 64 >>> (matrixU2, matrixV2);
		cudaErr = cudaGetLastError();
		if (cudaErr != cudaSuccess)
			cout << "boundaryKernel: " << cudaGetErrorString(cudaErr) << endl;


		// pointer swap
		temp = matrixU1;
		matrixU1 = matrixU2;
		matrixU2 = temp;

		temp = matrixV1;
		matrixV1 = matrixV2;
		matrixV2 = temp;
	}
	auto duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "duration: " << duration << endl;

	ofstream datustream;
	datustream.open("u.dat", ios::binary | ios::out);
	if (datustream.is_open())
	{
		cout << "stream open!" << endl;
		datustream.write((char*)matrixU, size * (L / SNAPSHOT_STEP + 1));
		datustream.close();
	}

	ofstream datvstream;
	datvstream.open("v.dat", ios::binary | ios::out);
	if (datvstream.is_open())
	{
		cout << "stream open!" << endl;
		datvstream.write((char*)matrixV, size * (L / SNAPSHOT_STEP + 1));
		datvstream.close();
	}

	FILE* csvu, * csvv;
	errno_t err = fopen_s(&csvu, "u.csv", "w");
	if (err) return err;
	err = fopen_s(&csvv, "v.csv", "w");
	if (err) return err;
	for (int j = 0; j <= L / SNAPSHOT_STEP; j++)
	{
		for (int i = 0; i < bufferlength; i++)
		{
			fprintf(csvu, "%f;", matrixU[j * bufferlength + i]);
			fprintf(csvv, "%f;", matrixV[j * bufferlength + i]);
		}
		fprintf(csvu, "\n");
		fprintf(csvv, "\n");
	}

	fclose(csvu);
	fclose(csvv);
}