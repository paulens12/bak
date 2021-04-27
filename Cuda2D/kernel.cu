
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "gif.h"
#include <random>
#include <chrono>

using namespace std;

#define Du (0.1)
#define chi (8.3)
#define au (1)
#define Bv (0.73)
#define dx2 (0.075 * 0.075)
#define dy2 (0.075 * 0.075)
#define dt (0.00005)

#define FRAME_DURATION 6
#define W 360
#define H 128
#define L 720000
#define SNAPSHOT_STEP 1200

#define BOUNDARY_BOTTOM(X) s X

__global__
void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__device__
inline double getNextU(double u, double ul, double ur, double uu, double ud, double v, double vl, double vr, double vu, double vd)
{
	double ul2 = (u + ul) / 2;
	double ur2 = (u + ur) / 2;
	double uu2 = (u + uu) / 2;
	double ud2 = (u + ud) / 2;

	return dt * (
		Du * (
			(ur - 2 * u + ul) / dx2
			+ (uu - 2 * u + ud) / dy2
			)
		- chi * (
			(ur2 * (vr - v) - ul2 * (v - vl)) / dx2
			+ (uu2 * (vu - v) - ud2 * (v - vd)) / dy2
			)
		+ au * u * (1 - u)
		) + u;
}

__device__
inline double getNextV(double u, double v, double vl, double vr, double vu, double vd)
{
	return dt * (
		(vr - 2 * v + vl) / dx2
		+ (vu - 2 * v + vd) / dy2
		+ u / (1 + Bv * u)
		- v
		) + v;
}

__global__
void calcKernel(double* uOutput, double* vOutput, double* uInput, double* vInput)
{
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int xStride = blockDim.x * gridDim.x;
	int yOffset = threadIdx.y + blockIdx.y * blockDim.y;
	int yStride = blockDim.y * gridDim.y;

	for (int w = xOffset; w < W; w += xStride)
	{
		for (int h = yOffset; h < H - 1; h += yStride)
		{
			// no-flux boundary condition - skip h = H - 1, h = 0
			if (h == 0) continue;

			double *uLeft, *uRight, *vLeft, *vRight, *uUp, *uDown, *vUp, *vDown;
			if (w == 0)
			{
				uLeft = uInput + W * h + W - 1; // cyclic boundary condition
				vLeft = vInput + W * h + W - 1; // cyclic boundary condition
			}
			else
			{
				uLeft = uInput + W * h + w - 1;
				vLeft = vInput + W * h + w - 1;
			}
			if (w == W - 1)
			{
				uRight = uInput + W * h; // cyclic boundary condition
				vRight = vInput + W * h; // cyclic boundary condition
			}
			else
			{
				uRight = uInput + W * h + w + 1;
				vRight = vInput + W * h + w + 1;
			}

			uUp = uInput + W * h + w + W;
			vUp = vInput + W * h + w + W;
			uDown = uInput + W * h + w - W;
			vDown = vInput + W * h + w - W;

			uOutput[W * h + w] = getNextU(uInput[W * h + w], *uLeft, *uRight, *uUp, *uDown, vInput[W * h + w], *vLeft, *vRight, *vUp, *vDown);
			vOutput[W * h + w] = getNextV(uInput[W * h + w], vInput[W * h + w], *vLeft, *vRight, *vUp, *vDown);
		}
	}
}

int main()
{
	double *matrixU1, *matrixU2, *matrixV1, *matrixV2;
	int size = W * H * sizeof(double);
	cudaMallocManaged(&matrixU1, size);
	cudaMallocManaged(&matrixU2, size);
	cudaMallocManaged(&matrixV1, size);
	cudaMallocManaged(&matrixV2, size);
	int deviceId;
	cudaGetDevice(&deviceId);

	cout << size / SNAPSHOT_STEP * L << endl;
	double* matrixU = (double*)malloc(size / SNAPSHOT_STEP * L); // for gif output
	double* matrixV = (double*)malloc(size / SNAPSHOT_STEP * L);
	perror("malloc");
	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<double> distr(0, 0.1);
	default_random_engine re(1);
	for (int i = 0; i < H*W; i++)
	{
		matrixU1[i] = distr(re) + 1.0;
		matrixV1[i] = 0;
	}

	auto err = cudaMemPrefetchAsync(matrixU1, size, deviceId);
	if (err != cudaSuccess)
		cout << cudaGetErrorString(err) << endl;
	err = cudaMemPrefetchAsync(matrixU2, size, deviceId);
	if (err != cudaSuccess)
		cout << cudaGetErrorString(err) << endl;
	err = cudaMemPrefetchAsync(matrixV1, size, deviceId);
	if (err != cudaSuccess)
		cout << cudaGetErrorString(err) << endl;
	err = cudaMemPrefetchAsync(matrixV2, size, deviceId);
	if (err != cudaSuccess)
		cout << cudaGetErrorString(err) << endl;
	int warpSize, multiProcessorCount;
	cudaDeviceGetAttribute(&warpSize, cudaDeviceAttr::cudaDevAttrWarpSize, deviceId);
	cudaDeviceGetAttribute(&multiProcessorCount, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, deviceId);
	int threads_per_block = 8 * warpSize;
	int number_of_blocks = 32 * multiProcessorCount;

	auto start = clock();
	for (int i = 0; i < L; i++)
	{
		if (i % SNAPSHOT_STEP == 0)
		{
			int step = i / SNAPSHOT_STEP;
			//save frame
			cout << "step " << step << ", time elapsed: " << (clock() - start) / (double)CLOCKS_PER_SEC << endl;
			for(int w = 0; w < W; w++)
				for (int h = 0; h < H; h++)
				{
					matrixU[step * H * W + W * h + w] = matrixU1[W * h + w];
					matrixV[step * H * W + W * h + w] = matrixV1[W * h + w];
				}
		}

		dim3 blocks(2, multiProcessorCount / 2);
		dim3 threads(32, 32);
		calcKernel <<< number_of_blocks, threads_per_block >>> (matrixU2, matrixV2, matrixU1, matrixV1);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << cudaGetErrorString(err) << endl;
		cudaDeviceSynchronize();

		for (int w = 0; w < W; w++)
		{ // no-flux boundary condition
			matrixU2[w] = (4 * matrixU2[W + w] - matrixU2[2 * W + w]) / 3;
			matrixV2[w] = (4 * matrixV2[W + w] - matrixV2[2 * W + w]) / 3;
			matrixU2[W * (H - 1) + w] = (4 * matrixU2[W * (H - 2) + w] - matrixU2[W * (H - 3) + w]) / 3;
			matrixV2[W * (H - 1) + w] = (4 * matrixV2[W * (H - 2) + w] - matrixV2[W * (H - 3) + w]) / 3;
		}

		// pointer swap
		double* temp = matrixU1;
		matrixU1 = matrixU2;
		matrixU2 = temp;

		temp = matrixV1;
		matrixV1 = matrixV2;
		matrixV2 = temp;
	}
	auto duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "duration: " << duration << endl;

	double maxU = 3.5;
	double maxV = 0.7;
	double multiU = 255 / maxU;
	double multiV = 255 / maxV;

	uint8_t frameU[W * H * 4];
	uint8_t frameV[W * H * 4];

	GifWriter gu;
	GifWriter gv;
	GifBegin(&gu, "U.gif", W, H, FRAME_DURATION);
	GifBegin(&gv, "V.gif", W, H, FRAME_DURATION);

	for (int i = 0; i < H * W; i++)
	{
		frameU[i * 4 + 3] = 0;
		frameV[i * 4 + 3] = 0;
	}
	for (int l = 0; l < L; l++)
	{
		for (int h = 0; h < H; h++)
		{
			for (int w = 0; w < W; w++)
			{
				uint8_t colorU = min((int)(multiU * matrixU[l * W * H + h * W + w]), 255);
				uint8_t colorV = min((int)(multiV * matrixV[l * W * H + h * W + w]), 255);
				frameU[4 * (W * h + w)] = colorU;
				frameU[4 * (W * h + w) + 1] = colorU;
				frameU[4 * (W * h + w) + 2] = colorU;
				frameV[4 * (W * h + w)] = colorV;
				frameV[4 * (W * h + w) + 1] = colorV;
				frameV[4 * (W * h + w) + 2] = colorV;
			}
		}
		GifWriteFrame(&gu, frameU, W, H, FRAME_DURATION);
		GifWriteFrame(&gv, frameV, W, H, FRAME_DURATION);
	}
	GifEnd(&gu);
	GifEnd(&gv);
	cout << "U multiplier: " << multiU << endl << "V multiplier: " << multiV << endl << "U max: " << maxU << endl << "V max: " << maxV;
}
