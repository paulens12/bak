
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "gd.h"
#include <random>
#include <chrono>

using namespace std;

#define Du (0.1)
#define chi (8.3)
#define au (1)
#define Bv (0.73)
#define dx2 (0.05 * 0.05)
#define dy2 (0.05 * 0.05)
#define dt (0.00005)

#define FRAME_DURATION 6
#define W 360
#define H 128
#define L 3456000
#define SNAPSHOT_STEP 3600

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
void boundaryKernel(double* u, double* v)
{
	int xOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int xStride = blockDim.x * gridDim.x;
	for (int w = xOffset; w < W; w += xStride)
	{ // no-flux boundary condition
		u[w] = (4 * u[W + w] - u[2 * W + w]) / 3;
		v[w] = (4 * v[W + w] - v[2 * W + w]) / 3;
		u[W * (H - 1) + w] = (4 * u[W * (H - 2) + w] - u[W * (H - 3) + w]) / 3;
		v[W * (H - 1) + w] = (4 * v[W * (H - 2) + w] - v[W * (H - 3) + w]) / 3;
	}
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
	cudaError_t err;
	err = cudaMallocManaged(&matrixU1, size);
	if (err != cudaSuccess)
		cout << "cudaMallocManaged: " << cudaGetErrorString(err) << endl;
	err = cudaMallocManaged(&matrixU2, size);
	if (err != cudaSuccess)
		cout << "cudaMallocManaged: " << cudaGetErrorString(err) << endl;
	err = cudaMallocManaged(&matrixV1, size);
	if (err != cudaSuccess)
		cout << "cudaMallocManaged: " << cudaGetErrorString(err) << endl;
	err = cudaMallocManaged(&matrixV2, size);
	if (err != cudaSuccess)
		cout << "cudaMallocManaged: " << cudaGetErrorString(err) << endl;
	int deviceId;
	err = cudaGetDevice(&deviceId);
	if (err != cudaSuccess)
		cout << "cudaGetDevice: " << cudaGetErrorString(err) << endl;

	cout << size * (L / SNAPSHOT_STEP + 1) << endl;
	double* matrixU = (double*)malloc(size * (L / SNAPSHOT_STEP + 1)); // for gif output
	perror("malloc error");
	double* matrixV = (double*)malloc(size * (L / SNAPSHOT_STEP + 1));
	perror("malloc error");
	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<double> distr(0, 0.1);
	default_random_engine re(1);
	for (int i = 0; i < H*W; i++)
	{
		matrixU[i] = matrixU1[i] = distr(re) + 1.0;
		matrixV[i] = matrixV1[i] = 0;
	}

	//err = cudaMemPrefetchAsync(matrixU1, size, deviceId);
	//if (err != cudaSuccess)
	//	cout << "cudaMemPrefetchAsync: " << cudaGetErrorString(err) << endl;
	//err = cudaMemPrefetchAsync(matrixU2, size, deviceId);
	//if (err != cudaSuccess)
	//	cout << "cudaMemPrefetchAsync: " << cudaGetErrorString(err) << endl;
	//err = cudaMemPrefetchAsync(matrixV1, size, deviceId);
	//if (err != cudaSuccess)
	//	cout << "cudaMemPrefetchAsync: " << cudaGetErrorString(err) << endl;
	//err = cudaMemPrefetchAsync(matrixV2, size, deviceId);
	//if (err != cudaSuccess)
	//	cout << "cudaMemPrefetchAsync: " << cudaGetErrorString(err) << endl;

	int warpSize, multiProcessorCount;
	cudaDeviceGetAttribute(&warpSize, cudaDeviceAttr::cudaDevAttrWarpSize, deviceId);
	cudaDeviceGetAttribute(&multiProcessorCount, cudaDeviceAttr::cudaDevAttrMultiProcessorCount, deviceId);

	// int threads_per_block = 8 * warpSize;
	// int number_of_blocks = 32 * multiProcessorCount;
	dim3 blocks(10, 1);
	dim3 threads(4, 128);

	auto start = clock();
	double* temp = NULL;
	for (int i = 0; i < L; i++)
	{
		if (i % SNAPSHOT_STEP == SNAPSHOT_STEP - 1)
		{
			cudaDeviceSynchronize();
			int step = i / SNAPSHOT_STEP + 1;
			//save frame
			double elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;
			cout << "step " << step << ", time elapsed: " << elapsed << ", avg: " << elapsed / step << endl;
			for (int w = 0; w < W; w++)
				for (int h = 0; h < H; h++)
				{
					matrixU[step * H * W + W * h + w] = matrixU1[W * h + w];
					matrixV[step * H * W + W * h + w] = matrixV1[W * h + w];
				}
		}
		cudaGetLastError();
		calcKernel <<< blocks, threads >>> (matrixU2, matrixV2, matrixU1, matrixV1);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << "calcKernel: " << cudaGetErrorString(err) << endl;


		boundaryKernel <<< 1, warpSize >>> (matrixU2, matrixV2);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << "boundaryKernel: " << cudaGetErrorString(err) << endl;

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

	double maxU = 3.5;
	double maxV = 0.7;
	double multiU = 255 / maxU;
	double multiV = 255 / maxV;

	uint8_t frameU[W * H * 4];
	uint8_t frameV[W * H * 4];

	gdImagePtr imu, imv, imup, imvp;
	imup = imvp = NULL;
	// Open output file in binary mode
	FILE* outu = fopen("U.gif", "wb");
	FILE* outv = fopen("V.gif", "wb");

	// Create the image
	imu = gdImageCreate(W, H);
	imv = gdImageCreate(W, H);
	gdImageGifAnimBegin(imu, outu, 1, -1);
	gdImageGifAnimBegin(imv, outv, 1, -1);

	// Allocate all grayscale colors
	int colors[256];
	for (int i = 0; i < 256; i++) {
		colors[i] = gdImageColorAllocate(imu, i, i, i);
	}
	gdImagePaletteCopy(imv, imu);

	// Insert frames
	for (int l = 0; l <= L / SNAPSHOT_STEP; l++)
	{
		for (int h = 0; h < H; h++)
		{
			for (int w = 0; w < W; w++)
			{
				uint8_t colorU = min((int)(multiU * matrixU[l * W * H + h * W + w]), 255);
				uint8_t colorV = min((int)(multiV * matrixV[l * W * H + h * W + w]), 255);
				gdImageSetPixel(imu, w, h, colorU);
				gdImageSetPixel(imv, w, h, colorV);
			}
		}
		gdImageGifAnimAdd(imu, outu, 0, 0, 0, FRAME_DURATION, gdDisposalNone, imup);
		gdImageGifAnimAdd(imv, outv, 0, 0, 0, FRAME_DURATION, gdDisposalNone, imvp);

		imvp = imv;
		imup = imu;
		imu = gdImageCreate(W, H);
		imv = gdImageCreate(W, H);
		gdImagePaletteCopy(imu, imup);
		gdImagePaletteCopy(imv, imvp);
	}

	gdImageGifAnimEnd(outu);
	gdImageGifAnimEnd(outv);

	// Free memory
	gdImageDestroy(imu);
	gdImageDestroy(imv);

	fclose(outu);
	fclose(outv);
	cout << "U multiplier: " << multiU << endl << "V multiplier: " << multiV << endl << "U max: " << maxU << endl << "V max: " << maxV;
	
	FILE* csvu = fopen("u.csv", "w");
	FILE* csvv = fopen("v.csv", "w");
	for (int j = 0; j <= L / SNAPSHOT_STEP; j++)
	{
		for (int i = 0; i < W*H; i++)
		{
			fprintf(csvu, "%f;", matrixU[j * W * H + i]);
			fprintf(csvv, "%f;", matrixV[j * W * H + i]);
		}
		fprintf(csvu, "\n");
		fprintf(csvv, "\n");
	}

	fclose(csvu);
	fclose(csvv);
}
