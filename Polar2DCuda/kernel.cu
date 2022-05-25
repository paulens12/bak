
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <gd.h>
#include <random>
#include <chrono>
#include "Constants.h"
#include "../RectPNG/RectPNG.h"
#include "argh.h"
#include "../Polar2DShared/UV.cpp"
#include "../PolarPNG/PolarPNG.h"

#define GET(arr, x, y, z) (arr[(x) + (y) * X + (z) * XY])

using namespace std;

#define L 300000
#define SNAPSHOT_STEP 100000

__global__
void calcKernel(double* uOutput, double* vOutput, double* uInput, double* vInput)
{
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int xStride = blockDim.x * gridDim.x;

	for (int r = 1; r < R / 4; ++r) {
		double ro = r * dr;
		for (int f = xOffset * 4; f < F; f += xStride * 4) {
			calcPoint(uOutput, vOutput, uInput, vInput, ro, r, f);
		}
	}

	for (int r = R / 4; r < R / 2; ++r) {
		double ro = r * dr;
		for (int f = xOffset * 2; f < F; f += xStride * 2) {
			calcPoint(uOutput, vOutput, uInput, vInput, ro, r, f);
		}
	}

	for (int r = R / 2; r < R - 1; ++r) {
		double ro = r * dr;
		for (int f = xOffset; f < F; f += xStride) {
			calcPoint(uOutput, vOutput, uInput, vInput, ro, r, f);
		}
	}
}

// apply boundary conditions
__global__
void boundaryKernel(double* u, double* v)
{
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int xStride = blockDim.x * gridDim.x;
	
	calcBoundary(u, v, xOffset, xStride);
}

int main(int argc, char* argv[])
{
	initializeCudaConstants();
	cudaError_t err;

	int bufferlength = R * F;
	double* matrixU1, * matrixU2, * matrixV1, * matrixV2;
	int size = bufferlength * sizeof(double);
	err = cudaMalloc(&matrixU1, size);
	if (err != cudaSuccess)
		cout << "cudaMalloc: " << cudaGetErrorString(err) << endl;
	err = cudaMalloc(&matrixU2, size);
	if (err != cudaSuccess)
		cout << "cudaMalloc: " << cudaGetErrorString(err) << endl;
	err = cudaMalloc(&matrixV1, size);
	if (err != cudaSuccess)
		cout << "cudaMalloc: " << cudaGetErrorString(err) << endl;
	err = cudaMalloc(&matrixV2, size);
	if (err != cudaSuccess)
		cout << "cudaMalloc: " << cudaGetErrorString(err) << endl;
	int deviceId;
	err = cudaGetDevice(&deviceId);
	if (err != cudaSuccess)
		cout << "cudaGetDevice: " << cudaGetErrorString(err) << endl;

	double* matrixU = new double[bufferlength];
	double* matrixV = new double[bufferlength];

	cout << size * (L / SNAPSHOT_STEP + 1) << endl;
	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<double> distr(0, 0.1);
	cout << seed << endl;
	default_random_engine re(1);
	for (int i = 0; i < bufferlength; i++)
	{
		if (i < bufferlength / 2 && i % 2 != 0 || i < bufferlength / 4 && i % 4 != 0)
			matrixU[i] = 0.0;
		else
			matrixU[i] = distr(re) + 1.0;
		matrixV[i] = 0.0;
	}

	cudaMemcpy(matrixU1, matrixU, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixV1, matrixV, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixU2, matrixU, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixV2, matrixV, size, cudaMemcpyHostToDevice);
	boundaryKernel <<< 28, 8 >>> (matrixU1, matrixV1);

	PolarPNG uPng(R, 3, F, 4.5);
	PolarPNG vPng(R, 3, F, 0.6);

	uPng.savePNG(matrixU, "u_step0.png");
	vPng.savePNG(matrixV, "v_step0.png");

	ofstream datustream;
	datustream.open("u.dat", ios::binary | ios::out);
	ofstream datvstream;
	datvstream.open("v.dat", ios::binary | ios::out);

	if (datustream.is_open())
		datustream.write((char*)matrixU, size);
	if (datvstream.is_open())
		datvstream.write((char*)matrixV, size);

#ifdef _DEBUG
	int warpSize, multiProcessorCount;
	cudaDeviceProp props;
	cudaDeviceGetAttribute(&warpSize, cudaDeviceAttr::cudaDevAttrWarpSize, deviceId);
	cudaDeviceGetAttribute(&multiProcessorCount, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, deviceId);
	cudaGetDeviceProperties(&props, deviceId);
#endif

	auto start = clock();
	auto start_current = start;
	double* temp = nullptr;
	for (int i = 0; i < L; i++)
	{
		cudaGetLastError(); // flush previous errors
		calcKernel <<< 7, 32 >>> (matrixU2, matrixV2, matrixU1, matrixV1);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << "calcKernel: " << cudaGetErrorString(err) << endl;

		boundaryKernel <<< 7, 32 >>> (matrixU2, matrixV2);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << "boundaryKernel: " << cudaGetErrorString(err) << endl;

		if (i % SNAPSHOT_STEP == SNAPSHOT_STEP - 1)
		{
			cudaDeviceSynchronize();
			int step = i / SNAPSHOT_STEP + 1;

			// save frame
			clock_t saveframe = clock();

			cudaGetLastError(); // flush previous errors
			cudaMemcpy(matrixU, matrixU2, size, cudaMemcpyDeviceToHost);
			err = cudaGetLastError();
			if (err != cudaSuccess)
				cout << "cudaMemcpy: " << cudaGetErrorString(err) << endl;
			cudaMemcpy(matrixV, matrixV2, size, cudaMemcpyDeviceToHost);
			err = cudaGetLastError();
			if (err != cudaSuccess)
				cout << "cudaMemcpy: " << cudaGetErrorString(err) << endl;

			if (datustream.is_open())
				datustream.write((char*)matrixU, size);
			if (datvstream.is_open())
				datvstream.write((char*)matrixV, size);

			uPng.savePNG(matrixU, "u_step" + to_string(step) + ".png");
			vPng.savePNG(matrixV, "v_step" + to_string(step) + ".png");
			double done = clock();

			double processedIn = (saveframe - start_current) / (double)CLOCKS_PER_SEC;
			double outputIn = (done - saveframe) / (double)CLOCKS_PER_SEC;
			double totalTime = (done - start) / (double)CLOCKS_PER_SEC;
			cout << "step " << step << ", done processing in: " << processedIn << ", saved snapshot in: " << outputIn << ", total: " << totalTime << ", avg: " << totalTime / step << endl;
			start_current = clock();
		}

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

	delete[] matrixU, matrixV;
	cudaFree(matrixU1);
	cudaFree(matrixU2);
	cudaFree(matrixV1);
	cudaFree(matrixV2);
}
