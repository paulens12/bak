
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

#define GET(arr, x, y, z) (arr[(x) + (y) * X + (z) * XY])

using namespace std;

#define L 5000000
#define SNAPSHOT_STEP 10000

__device__
double getNextU(double* u, double* v, double* o, int x, int y, int z)
{
	// u_(i_x+0.5,i_y,i_z)
	double uxp2 = (GET(u, x + 1, y, z) + GET(u, z, y, z)) / 2;
	// u_(i_x-0.5,i_y,i_z)
	double uxm2 = (GET(u, x - 1, y, z) + GET(u, z, y, z)) / 2;
	// ...
	double uyp2 = (GET(u, x, y + 1, z) + GET(u, z, y, z)) / 2;
	double uym2 = (GET(u, x, y - 1, z) + GET(u, z, y, z)) / 2;
	double uzp2 = (GET(u, x, y, z + 1) + GET(u, z, y, z)) / 2;
	double uzm2 = (GET(u, x, y, z - 1) + GET(u, z, y, z)) / 2;

	double DuMember =
		(GET(u, x + 1, y, z) - 2 * GET(u, x, y, z) + GET(u, x - 1, y, z)) / dx2 +
		(GET(u, x, y + 1, z) - 2 * GET(u, x, y, z) + GET(u, x, y - 1, z)) / dy2 +
		(GET(u, x, y, z + 1) - 2 * GET(u, x, y, z) + GET(u, x, y, z - 1)) / dz2;
	double ChiMember =
		(uxp2 * (GET(v, x + 1, y, z) - GET(v, x, y, z)) - uxm2 * (GET(v, x, y, z) - GET(v, x - 1, y, z))) / dx2 +
		(uyp2 * (GET(v, x, y + 1, z) - GET(v, x, y, z)) - uym2 * (GET(v, x, y, z) - GET(v, x, y - 1, z))) / dy2 +
		(uzp2 * (GET(v, x, y, z + 1) - GET(v, x, y, z)) - uzm2 * (GET(v, x, y, z) - GET(v, x, y, z - 1))) / dz2;
	double aMember = GET(u, x, y, z) * (1 - GET(u, x, y, z) / GET(o, x, y, z));

	return (Du * DuMember - chi * ChiMember + au * aMember) * dt + GET(u, x, y, z);
}

__device__
double getNextV(double* u, double* v, int x, int y, int z)
{
	return (
		(GET(v, x + 1, y, z) - 2 * GET(v, x, y, z) + GET(v, x - 1, y, z)) / dx2 +
		(GET(v, x, y + 1, z) - 2 * GET(v, x, y, z) + GET(v, x, y - 1, z)) / dy2 +
		(GET(v, x, y, z + 1) - 2 * GET(v, x, y, z) + GET(v, x, y, z - 1)) / dz2 +
		GET(u, x, y, z) / (1 + Bv * GET(u, x, y, z)) - GET(v, x, y, z)
		) * dt + GET(v, x, y, z);
}

__device__
double getNextO(double* u, double* o, double o_z_plus_one, int x, int y, int z)
{
	return (
		Do * (
			(GET(o, x + 1, y, z) - 2 * GET(o, x, y, z) + GET(o, x - 1, y, z)) / dx2 +
			(GET(o, x, y + 1, z) - 2 * GET(o, x, y, z) + GET(o, x, y - 1, z)) / dy2 +
			(o_z_plus_one - 2 * GET(o, x, y, z) + GET(o, x, y, z - 1)) / dz2
			) -
		gamma_o * GET(u, x, y, z)
		) * dt + GET(o, x, y, z);
}

__device__
void calcPoint(double* uOutput, double* vOutput, double* oOutput, double* uInput, double* vInput, double* oInput, int x, int y, int z) {
	GET(uOutput, x, y, z) = getNextU(uInput, vInput, oInput, x, y, z);
	GET(vOutput, x, y, z) = getNextV(uInput, vInput, x, y, z);
	GET(oOutput, x, y, z) = getNextO(uInput, oInput, GET(oInput, x, y, z + 1), x, y, z);
}

__global__
void calcKernel(double* uOutput, double* vOutput, double* oOutput, double* uInput, double* vInput, double* oInput)
{
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int xStride = blockDim.x * gridDim.x;
	int yOffset = threadIdx.y + blockIdx.y * blockDim.y;
	int yStride = blockDim.y * gridDim.y;
	int zOffset = threadIdx.z + blockIdx.z * blockDim.z;
	int zStride = blockDim.z * gridDim.z;

	for (int z = zOffset + 1; z < Z - 1; z += zStride)
	{
		for (int y = yOffset + 1; y < Y - 1; y += yStride)
		{
			for (int x = xOffset + 1; x < X - 1; x += xStride)
			{
				calcPoint(uOutput, vOutput, oOutput, uInput, vInput, oInput, x, y, z);
			}
		}
	}
}

// apply boundary conditions
__global__
void boundaryKernel(double* u, double* v, double* o)
{
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int xStride = blockDim.x * gridDim.x;
	int yOffset = threadIdx.y + blockIdx.y * blockDim.y;
	int yStride = blockDim.y * gridDim.y;

	for (int y = yOffset + 1; y < Y - 1; y += yStride) {
		for (int x = xOffset + 1; x < X - 1; x += xStride) {
			GET(u, x, y, 0) = fmax((4 * GET(u, x, y, 1) - GET(u, x, y, 2)) / 3.0, 0.0);
			GET(v, x, y, 0) = fmax((4 * GET(v, x, y, 1) - GET(v, x, y, 2)) / 3.0, 0.0);
			GET(o, x, y, 0) = fmax((4 * GET(o, x, y, 1) - GET(o, x, y, 2)) / 3.0, 0.0);
			GET(u, x, y, Z - 1) = fmax((4 * GET(u, x, y, Z - 2) - GET(u, x, y, Z - 3)) / 3.0, 0.0);
			GET(v, x, y, Z - 1) = fmax((4 * GET(v, x, y, Z - 2) - GET(v, x, y, Z - 3)) / 3.0, 0.0);
			GET(o, x, y, Z - 1) = getNextO(u, o, o0, x, y, Z - 1);
		}
		for (int z = xOffset + 1; z < Z - 1; z += xStride) {
			GET(u, 0, y, z) = fmax((4 * GET(u, 1, y, z) - GET(u, 2, y, z)) / 3.0, 0.0);
			GET(v, 0, y, z) = fmax((4 * GET(v, 1, y, z) - GET(v, 2, y, z)) / 3.0, 0.0);
			GET(o, 0, y, z) = fmax((4 * GET(o, 1, y, z) - GET(o, 2, y, z)) / 3.0, 0.0);
			GET(u, X - 1, y, z) = fmax((4 * GET(u, X - 2, y, z) - GET(u, X - 3, y, z)) / 3.0, 0.0);
			GET(v, X - 1, y, z) = fmax((4 * GET(v, X - 2, y, z) - GET(v, X - 3, y, z)) / 3.0, 0.0);
			GET(o, X - 1, y, z) = fmax((4 * GET(o, X - 2, y, z) - GET(o, X - 3, y, z)) / 3.0, 0.0);
		}
	}
	for (int z = yOffset + 1; z < Z - 1; z += yStride) {
		for (int x = xOffset + 1; x < X - 1; x += xStride) {
			GET(u, x, 0, z) = fmax((4 * GET(u, x, 1, z) - GET(u, x, 2, z)) / 3.0, 0.0);
			GET(v, x, 0, z) = fmax((4 * GET(v, x, 1, z) - GET(v, x, 2, z)) / 3.0, 0.0);
			GET(o, x, 0, z) = fmax((4 * GET(o, x, 1, z) - GET(o, x, 2, z)) / 3.0, 0.0);
			GET(u, x, Y - 1, z) = fmax((4 * GET(u, x, Y - 2, z) - GET(u, x, Y - 3, z)) / 3.0, 0.0);
			GET(v, x, Y - 1, z) = fmax((4 * GET(v, x, Y - 2, z) - GET(v, x, Y - 3, z)) / 3.0, 0.0);
			GET(o, x, Y - 1, z) = fmax((4 * GET(o, x, Y - 2, z) - GET(o, x, Y - 3, z)) / 3.0, 0.0);
		}
	}
}

int main(int argc, char* argv[])
{
	cudaError_t err;
	/*
	double tmp;
	argh::parser cmdl(argc, argv);
	err = cudaMemcpyFromSymbol(&tmp, Du, sizeof(double), 0, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		cout << "cudaMemcpyFromSymbol: " << cudaGetErrorString(err) << endl;
	cmdl("Du", tmp) >> tmp;
	cout << "Du " << tmp;
	err = cudaMemcpyToSymbol(Du, &tmp, sizeof(double), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		cout << "cudaMemcpyToSymbol: " << cudaGetErrorString(err) << endl;

	cudaMemcpyFromSymbol(&tmp, chi, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cmdl("chi", tmp) >> tmp;
	cout << " chi " << tmp;
	err = cudaMemcpyToSymbol(chi, &tmp, sizeof(double), 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
		cout << "cudaMemcpyToSymbol: " << cudaGetErrorString(err) << endl;

	cudaMemcpyFromSymbol(&tmp, au, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cmdl("au", tmp) >> tmp;
	cout << " au " << tmp;
	cudaMemcpyToSymbol(au, &tmp, sizeof(double), 0, cudaMemcpyHostToDevice);

	cudaMemcpyFromSymbol(&tmp, Bv, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cmdl("Bv", tmp) >> tmp;
	cout << " Bv " << tmp;
	cudaMemcpyToSymbol(Bv, &tmp, sizeof(double), 0, cudaMemcpyHostToDevice);

	cudaMemcpyFromSymbol(&tmp, gamma_o, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cmdl("go", tmp) >> tmp;
	cout << " gamma_o " << tmp;
	cudaMemcpyToSymbol(gamma_o, &tmp, sizeof(double), 0, cudaMemcpyHostToDevice);

	cudaMemcpyFromSymbol(&tmp, Do, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cmdl("Do", tmp) >> tmp;
	cout << " Do " << tmp;
	cudaMemcpyToSymbol(Do, &tmp, sizeof(double), 0, cudaMemcpyHostToDevice);

	cudaMemcpyFromSymbol(&tmp, dx2, sizeof(double), 0, cudaMemcpyDeviceToHost);
	cmdl("dx", sqrt(tmp)) >> tmp; tmp *= tmp;
	cout << " dx2 " << tmp;
	cudaMemcpyToSymbol(dx2, &tmp, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dy2, &tmp, sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dz2, &tmp, sizeof(double), 0, cudaMemcpyHostToDevice);

	cout << endl;
	*/

	int bufferlength = X * Y * Z;
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
	for (int i = 0; i < bufferlength; i++)
	{
		matrixU[i] = distr(re) + 1.0;
		matrixV[i] = 0.0;
		matrixO[i] = o0;
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
	for (int z : { 0, Z / 4, Z / 2, 3 * Z / 4, Z - 1 }) {
		savePNG(X, Y, &GET(matrixU, 0, 0, z), 2, "u_step0_Z" + to_string(z) + ".png");
		savePNG(X, Y, &GET(matrixV, 0, 0, z), 1, "v_step0_Z" + to_string(z) + ".png");
		savePNG(X, Y, &GET(matrixO, 0, 0, z), 2, "o_step0_Z" + to_string(z) + ".png");
	}

	cudaMemcpy(matrixU1, matrixU, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixV1, matrixV, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixO1, matrixO, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixU2, matrixU, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixV2, matrixV, size, cudaMemcpyHostToDevice);
	cudaMemcpy(matrixO2, matrixO, size, cudaMemcpyHostToDevice);

	int warpSize, multiProcessorCount;
	cudaDeviceProp props;
	cudaDeviceGetAttribute(&warpSize, cudaDeviceAttr::cudaDevAttrWarpSize, deviceId);
	cudaDeviceGetAttribute(&multiProcessorCount, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, deviceId);
	cudaGetDeviceProperties(&props, deviceId);

	// int threads_per_block = 8 * warpSize;
	// int number_of_blocks = 32 * multiProcessorCount;
	dim3 blocks(5, 10, 10);
	dim3 threads(16, 8, 8);

	auto start = clock();
	auto start_current = start;
	double* temp = nullptr;
	for (int i = 0; i < L; i++)
	{
		cudaGetLastError(); // flush previous errors
		calcKernel <<< blocks, threads >>> (matrixU2, matrixV2, matrixO2, matrixU1, matrixV1, matrixO1);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << "calcKernel: " << cudaGetErrorString(err) << endl;

		boundaryKernel <<< dim3(10, 4), dim3(8, 4) >>> (matrixU2, matrixV2, matrixO2);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << "boundaryKernel: " << cudaGetErrorString(err) << endl;

		bool dbg = false; // (i > 530000 && i < 530100);
		if (i % SNAPSHOT_STEP == SNAPSHOT_STEP - 1 || dbg)
		{
			cudaDeviceSynchronize();
			int step = i / SNAPSHOT_STEP + 1;
			if (dbg)
				step = i;

			// save frame
			clock_t saveframe = clock();

			cudaMemcpy(matrixU, matrixU2, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(matrixV, matrixV2, size, cudaMemcpyDeviceToHost);
			cudaMemcpy(matrixO, matrixO2, size, cudaMemcpyDeviceToHost);

			if (datustream.is_open())
				datustream.write((char*)matrixU, size);
			if (datvstream.is_open())
				datvstream.write((char*)matrixV, size);
			if (datostream.is_open())
				datostream.write((char*)matrixO, size);

			for (int z : { 0, Z / 4, Z / 2, 3 * Z / 4, Z - 1 }) {
				savePNG(X, Y, &GET(matrixU, 0, 0, z), 2, "u_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				savePNG(X, Y, &GET(matrixV, 0, 0, z), 1, "v_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				savePNG(X, Y, &GET(matrixO, 0, 0, z), 2, "o_step" + to_string(step) + "_Z" + to_string(z) + ".png");
			}

			if (dbg) {
				for (int z = 0; z < Z; z++) {
					savePNG(X, Y, &GET(matrixU, 0, 0, z), 2, "u_step" + to_string(step) + "_Z" + to_string(z) + ".png");
					savePNG(X, Y, &GET(matrixV, 0, 0, z), 1, "v_step" + to_string(step) + "_Z" + to_string(z) + ".png");
					savePNG(X, Y, &GET(matrixO, 0, 0, z), 2, "o_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				}
			}
			double done = clock();

			double processedIn = (saveframe - start_current) / (double)CLOCKS_PER_SEC;
			double outputIn = (done - saveframe) / (double)CLOCKS_PER_SEC;
			double totalTime = (done - start) / (double)CLOCKS_PER_SEC;
			cout << "step " << step << ", done processing in: " << processedIn << ", saved snapshot in:" << outputIn << ", total: " << totalTime << ", avg: " << totalTime / step << endl;
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
}
