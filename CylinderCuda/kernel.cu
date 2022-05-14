
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <gd.h>
#include <random>
#include <chrono>
#include "Constants.h"
#include "argh.h"
#include "../CylinderShared/UV.cpp"
#include "../PolarPNG/PolarPNG.h"

#define GET(arr, x, y, z) (arr[(x) + (y) * X + (z) * XY])

using namespace std;

#define L 6000000
#define SNAPSHOT_STEP 1000

__global__
void calcKernel(double* uOutput, double* vOutput, double* oOutput, double* uInput, double* vInput, double* oInput)
{
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int xStride = blockDim.x * gridDim.x;
	int yOffset = threadIdx.y + blockIdx.y * blockDim.y;
	int yStride = blockDim.y * gridDim.y;

	for (int z = 1 + xOffset; z < Z - 1; z += xStride) {
		for (int r = 1; r < R / 4; ++r) {
			double ro = r * dr;
			for (int f = yOffset * 4; f < F; f += yStride * 4) {
				calcPoint(uOutput, vOutput, oOutput, uInput, vInput, oInput, ro, r, f, z);
			}
		}

		for (int r = R / 4; r < R / 2; ++r) {
			double ro = r * dr;
			for (int f = yOffset * 2; f < F; f += yStride * 2) {
				calcPoint(uOutput, vOutput, oOutput, uInput, vInput, oInput, ro, r, f, z);
			}
		}

		for (int r = R / 2; r < R; ++r) {
			double ro = r * dr;
			for (int f = yOffset; f < F; f += yStride) {
				if (Z == Z - 1)
					calcBoundaryO(oInput, uInput[R * F * (Z - 1) + F * r + f], ro, r, f);
				else
					calcPoint(uOutput, vOutput, oOutput, uInput, vInput, oInput, ro, r, f, z);
			}
		}
	}
}

// apply boundary conditions
__global__
void boundaryKernel(double* uOutput, double* vOutput, double* oOutput)
{
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int xStride = blockDim.x * gridDim.x;
	int yOffset = threadIdx.y + blockIdx.y * blockDim.y;
	int yStride = blockDim.y * gridDim.y;

	calcBoundary(uOutput, vOutput, oOutput, xOffset, xStride, yOffset, yStride);
}

int main(int argc, char* argv[])
{
	initializeCudaConstants();

	cudaError_t err;

	int bufferlength = R * F * Z;
	double* matrixU1, * matrixU2, * matrixV1, * matrixV2, * matrixO1, * matrixO2;
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
	err = cudaMalloc(&matrixO1, size);
	if (err != cudaSuccess)
		cout << "cudaMalloc: " << cudaGetErrorString(err) << endl;
	err = cudaMalloc(&matrixO2, size);
	if (err != cudaSuccess)
		cout << "cudaMalloc: " << cudaGetErrorString(err) << endl;
	int deviceId;
	err = cudaGetDevice(&deviceId);
	if (err != cudaSuccess)
		cout << "cudaGetDevice: " << cudaGetErrorString(err) << endl;

	double* matrixU = new double[bufferlength];
	double* matrixV = new double[bufferlength];
	double* matrixO = new double[bufferlength];

	cout << size * (L / SNAPSHOT_STEP + 1) << endl;
	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<double> distr(0, 0.1);
	cout << seed << endl;
	default_random_engine re(1);

	int layerSize = R * F;
	for (int z = 0; z < Z; z++) {
		int offset = layerSize * z;
		for (int i = 0; i < layerSize; i++)
		{
			if (i < layerSize / 2 && i % 2 != 0 || i < layerSize / 4 && i % 4 != 0)
				matrixU[offset + i] = 0.0;
			else
				matrixU[offset + i] = distr(re) + 1.0;
			matrixV[offset + i] = 0.0;
			matrixO[offset + i] = o0;
		}
	}


	cout << "o0:" << o0 << endl;

	cudaMemcpy(matrixU1, matrixU, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixV1, matrixV, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixO1, matrixO, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixU2, matrixU, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixV2, matrixV, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixO2, matrixO, size, cudaMemcpyHostToDevice);

	dim3 boundaryBlocks(1, 1);
	dim3 boundaryThreads(1, 32);
	dim3 calcBlocks(2, 14);
	dim3 calcThreads(32, 1);
	boundaryKernel <<< boundaryBlocks, boundaryThreads >> > (matrixU1, matrixV1, matrixO1);

	PolarPNG uPng(R, 3, F, 4.0);
	PolarPNG vPng(R, 3, F, 2.0);
	PolarPNG oPng(R, 3, F, 2.0);

	for (int z : { 0, Z / 4, Z / 2, 3 * Z / 4, Z - 1 }) {
		uPng.savePNG(matrixU + R * F * z, "u_step0_Z" + to_string(z) + ".png");
		vPng.savePNG(matrixV + R * F * z, "v_step0_Z" + to_string(z) + ".png");
		oPng.savePNG(matrixO + R * F * z, "o_step0_Z" + to_string(z) + ".png");
	}

	ofstream datustream;
	datustream.open("u.dat", ios::binary | ios::out);
	ofstream datvstream;
	datvstream.open("v.dat", ios::binary | ios::out);
	ofstream datostream;
	datostream.open("o.dat", ios::binary | ios::out);

	if (datustream.is_open())
		datustream.write((char*)matrixU, size);
	if (datvstream.is_open())
		datvstream.write((char*)matrixV, size);
	if (datostream.is_open())
		datostream.write((char*)matrixO, size);

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
		calcKernel <<< calcBlocks, calcThreads >>> (matrixU2, matrixV2, matrixO2, matrixU1, matrixV1, matrixO1);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << "calcKernel: " << cudaGetErrorString(err) << endl;

		boundaryKernel <<< boundaryBlocks, boundaryThreads >>> (matrixU2, matrixV2, matrixO2);
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
			cudaMemcpy(matrixO, matrixO2, size, cudaMemcpyDeviceToHost);
			err = cudaGetLastError();
			if (err != cudaSuccess)
				cout << "cudaMemcpy: " << cudaGetErrorString(err) << endl;

			if (datustream.is_open())
				datustream.write((char*)matrixU, size);
			if (datvstream.is_open())
				datvstream.write((char*)matrixV, size);
			if (datostream.is_open())
				datostream.write((char*)matrixO, size);

			for (int z : { 0, Z / 4, Z / 2, 3 * Z / 4, Z - 1 }) {
				uPng.savePNG(matrixU + R * F * z, "u_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				vPng.savePNG(matrixV + R * F * z, "v_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				oPng.savePNG(matrixO + R * F * z, "o_step" + to_string(step) + "_Z" + to_string(z) + ".png");
			}
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
