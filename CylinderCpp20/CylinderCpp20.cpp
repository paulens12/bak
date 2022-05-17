#define THREAD_N 2

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <thread>
#include <UV.h>
#include <Constants.h>
#include <barrier>
#include "../PolarPNG/PolarPNG.h"

using namespace std;

//#define H 128
//#define L 60000
//#define SNAPSHOT_STEP 10000

#define L 6000000
#define SNAPSHOT_STEP 10000

// perform one iteration of simulation
void iterate(double* matrixU2, double* matrixV2, double* matrixO2, double* matrixU1, double* matrixV1, double* matrixO1, int size)
{
	PolarPNG uPng(R, 3, F, 4.0);
	PolarPNG vPng(R, 3, F, 2.0);
	PolarPNG oPng(R, 3, F, 2.0);

	for (int z : { 0, Z / 4, Z / 2, 3 * Z / 4, Z - 1 }) {
		uPng.savePNG(matrixU1 + R * F * z, "u_step0_Z" + to_string(z) + ".png");
		vPng.savePNG(matrixV1 + R * F * z, "v_step0_Z" + to_string(z) + ".png");
		oPng.savePNG(matrixO1 + R * F * z, "o_step0_Z" + to_string(z) + ".png");
	}

	ofstream datustream;
	datustream.open("u.dat", ios::binary | ios::out);
	ofstream datvstream;
	datvstream.open("v.dat", ios::binary | ios::out);
	ofstream datostream;
	datostream.open("o.dat", ios::binary | ios::out);

	if (datustream.is_open())
		datustream.write((char*)matrixU1, size);
	if (datvstream.is_open())
		datvstream.write((char*)matrixV1, size);
	if (datostream.is_open())
		datostream.write((char*)matrixO1, size);

	auto start = clock();
	auto start_current = start;
	int i = 0;
	double* temp = nullptr;
	barrier sync_point(THREAD_N, [&]() noexcept {
		if (i % SNAPSHOT_STEP == SNAPSHOT_STEP - 1)
		{
			int step = i / SNAPSHOT_STEP + 1;
			// save frame
			clock_t saveframe = clock();


			if (datustream.is_open())
				datustream.write((char*)matrixU2, size);
			if (datvstream.is_open())
				datvstream.write((char*)matrixV2, size);
			if (datostream.is_open())
				datostream.write((char*)matrixO2, size);

			for (int z : { 0, Z / 4, Z / 2, 3 * Z / 4, Z - 1 }) {
				uPng.savePNG(matrixU2 + R * F * z, "u_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				vPng.savePNG(matrixV2 + R * F * z, "v_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				oPng.savePNG(matrixO2 + R * F * z, "o_step" + to_string(step) + "_Z" + to_string(z) + ".png");
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

		temp = matrixO1;
		matrixO1 = matrixO2;
		matrixO2 = temp;

		++i;
		});

	auto work = [&](int thrn) {
		double ro;
		while (i < L) {
			for(int z = thrn + 1; z < Z; z += THREAD_N) {
				for (int r = 1; r < R - 1; ++r) {
					ro = r * dr;
					for (int f = 0; f < F; ++f) {
						if (z == Z - 1)
							matrixO2[R * F * z + F * r + f] = calcBoundaryO(matrixO1, matrixU1[R * F * z + F * r + f], ro, r, f);
						else
							calcPoint(matrixU2, matrixV2, matrixO2, matrixU1, matrixV1, matrixO1, ro, r, f, z);
					}
				}
			}
			calcBoundary(matrixU2, matrixV2, matrixO2, thrn, THREAD_N);
			// end iteration
			sync_point.arrive_and_wait();
		}
	};

	thread* threads[THREAD_N]{};

	for (int i = 0; i < THREAD_N; ++i) {
		threads[i] = new thread(work, i);
	}

	for (thread* thr : threads) {
		thr->join();
		delete thr;
	}

	auto duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "duration: " << duration << endl;
}

int main()
{
	double* matrixU1, * matrixU2, * matrixV1, * matrixV2, * matrixO1, * matrixO2;
	int bufferlength = R * F * Z;
	int size = bufferlength * sizeof(double);
	matrixU1 = new double[bufferlength];
	matrixV1 = new double[bufferlength];
	matrixO1 = new double[bufferlength];
	matrixU2 = new double[bufferlength];
	matrixV2 = new double[bufferlength];
	matrixO2 = new double[bufferlength];

	errno_t err;

	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<double> distr(0, 0.1);
	default_random_engine re(1);

	int layerSize = R * F;
	for (int z = 0; z < Z; z++) {
		int offset = layerSize * z;
		for (int i = 0; i < layerSize; i++)
		{
			if (i < layerSize / 2 && i % 2 != 0 || i < layerSize / 4 && i % 4 != 0)
				matrixU1[offset + i] = matrixU2[offset + i] = 0.0;
			else
				matrixU1[offset + i] = matrixU2[offset + i] = distr(re) + 1.0;
			matrixV1[offset + i] = matrixV2[offset + i] = 0.0;
			matrixO1[offset + i] = matrixO2[offset + i] = o0;
		}
	}
	calcBoundary(matrixU1, matrixV1, matrixO1);
	calcBoundary(matrixU2, matrixV2, matrixO2);

	iterate(matrixU2, matrixV2, matrixO2, matrixU1, matrixV1, matrixO1, size);

	delete[] matrixU1, matrixU2, matrixV1, matrixV2;
}