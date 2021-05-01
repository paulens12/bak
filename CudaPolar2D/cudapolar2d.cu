
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>

using namespace std;

#define Du (0.1)
#define chi (8.3)
#define au (1)
#define Bv (0.73)
#define dr (0.05)
#define dr2 (0.0025) /* 0.05 * 0.05 */
#define BASE_df (0.05)
#define dt (0.00005)

#define FRAME_DURATION 6
#define F 360
//#define H 128
#define R 128
#define L 3456000
#define SNAPSHOT_STEP 3600

// fp - f+1
// fm - f-1
// rp - r+1
// rm - r-1
__device__
inline double getNextU(
	double u, double urp, double urm, double ufp, double ufm,
	double v, double vrp, double vrm, double vfp, double vfm,
	double r, double f, double df
)
{
	double ufp2 = (u + ufp) / 2;
	double ufm2 = (u + ufm) / 2;

	return dt * (
		Du * (
			(urp - urm) / (2 * dr * r)
			+ (urp - 2 * u + urm) / dr2
			+ (ufp - 2 * u + ufm) / (df * df * r * r)
			)
		- chi * (
			(u * (vrp - vrm) / (2 * r)
				+ (urp - urm) * (vrp - vrm) / (4 * dr)
				+ u * (vrp - 2 * v + vrm) / dr) / dr
			+ (ufp2 * (vfp - v) - ufm2 * (v - vfm)) / (df * df * r * r) / dr2)
		+ au * u * (1 - u)
		) + u;
}

__device__
inline double getNextV(
	double u,
	double v, double vrp, double vrm, double vfp, double vfm,
	double r, double f, double df
)
{
	return dt * (
		(vrp - vrm) / (2 * dr * r)
		+ (vrp - 2 * v + vrm) / dr2
		+ (vfp - 2 * v + vfm) / (df * df * r * r)
		+ u / (1 + Bv * u)
		- v
		) + u;
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
}

__global__
void calcKernel(double* uOutput, double* vOutput, double* uInput, double* vInput)
{
	int rOffset = threadIdx.x + blockIdx.x * blockDim.x;
	int rStride = blockDim.x * gridDim.x;
	int fOffset = threadIdx.y + blockIdx.y * blockDim.y;
	int fStride = blockDim.y * gridDim.y;

	for (int r = rOffset; r < R - 1; r += rStride)
	{
		double urp, urm, vrp, vrm, ufp, ufm, vfp, vfm, u, v;
		int rm, fp, fm; // offsets
		int df;

		// no-flux boundary condition - skip r = R - 1

		if (r < R / 4)
		{
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

				if (r == 0)
				{
					if (f >= F / 8) rm = r * F / 4 + f - F / 8;
					else rm = r * F / 4 + f + F / 8;
				}
				else
				{
					rm = (r - 1) * F / 4 + f;
				}

				urm = uInput[rm];
				vrm = vInput[rm];

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

				uOutput[r * F / 4 + f] = getNextU(u, urp, urm, ufp, ufm, v, vrp, vrm, vfp, vfm, r, f, df);
				vOutput[r * F / 4 + f] = getNextV(u, v, vrp, vrm, vfp, vfm, r, f, df);
			}
		}
		else if (r < R / 2)
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

				uOutput[R * F / 16 + (r - R / 4) * F / 2 + f] = getNextU(u, urp, urm, ufp, ufm, v, vrp, vrm, vfp, vfm, r, f, df);
				vOutput[R * F / 16 + (r - R / 4) * F / 2 + f] = getNextV(u, v, vrp, vrm, vfp, vfm, r, f, df);
			}
		} 
		else
		{
			for (int f = fOffset; f < F; f += fStride)
			{

				u = uInput[3 * R * F / 16 + (r - R / 2) * F + f];
				v = vInput[3 * R * F / 16 + (r - R / 2) * F + f];

				if (f == 0) fm = 3 * R * F / 16 + (r - R / 2 + 1) * F - 1;
				else fm = 3 * R * F / 16 + (r - R / 2) * F + f - 1;
				if (f == F / 2 - 1) fp = 3 * R * F / 16 + (r - R / 2) * F;
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

				uOutput[3 * R * F / 16 + (r - R / 2) * F + f] = getNextU(u, urp, urm, ufp, ufm, v, vrp, vrm, vfp, vfm, r, f, df);
				vOutput[3 * R * F / 16 + (r - R / 2) * F + f] = getNextV(u, v, vrp, vrm, vfp, vfm, r, f, df);
			}
		}
	}
}

int main()
{
	double* matrixU1, * matrixU2, * matrixV1, * matrixV2;
	int bufferlength = 11 * R * F / 16;
	int size = bufferlength * sizeof(double);
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

	for (int i = 0; i < bufferlength; i++)
	{
		matrixU[i] = matrixU1[i] = distr(re) + 1.0;
		matrixV[i] = matrixV1[i] = 0;
	}

	dim3 blocks(1, 10);
	dim3 threads(32, 12);

	auto start = clock();
	double* temp = NULL;
	for (int i = 0; i < L; ++i)
	{
		if (i % SNAPSHOT_STEP == SNAPSHOT_STEP - 1)
		{
			cudaDeviceSynchronize();
			int step = i / SNAPSHOT_STEP + 1;
			//save frame
			double elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;
			cout << "step " << step << ", time elapsed: " << elapsed << ", avg: " << elapsed / step << endl;
			cudaMemcpy(matrixU + step * bufferlength, matrixU1 + step * bufferlength, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			cudaMemcpy(matrixV + step * bufferlength, matrixV1 + step * bufferlength, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);
			//for (int j = 0; j < bufferlength; ++j)
			//{
			//	matrixU[step * bufferlength + j] = matrixU1[j];
			//	matrixV[step * bufferlength + j] = matrixV1[j];
			//}
		}

		cudaGetLastError();
		calcKernel <<< blocks, threads >>> (matrixU2, matrixV2, matrixU1, matrixV1);
		err = cudaGetLastError();
		if (err != cudaSuccess)
			cout << "calcKernel: " << cudaGetErrorString(err) << endl;


		boundaryKernel <<< 10, 64 >>> (matrixU2, matrixV2);
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

	//double maxU = 3.5;
	//double maxV = 0.7;
	//double multiU = 255 / maxU;
	//double multiV = 255 / maxV;

	FILE* csvu = fopen("u.csv", "w");
	FILE* csvv = fopen("v.csv", "w");
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