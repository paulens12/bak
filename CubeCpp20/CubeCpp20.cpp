#define THREAD_N 6

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <thread>
#include <Constants.h>
#include <UV.h>
#include <barrier>
#include "../RectPNG/RectPNG.h"

using namespace std;

//#define H 128
//#define L 60000
//#define SNAPSHOT_STEP 10000

#define L 20000
#define SNAPSHOT_STEP 1000

// perform one iteration of simulation
void iterate(double* matrixU2, double* matrixV2, double* matrixO2, double* matrixU1, double* matrixV1, double* matrixO1, int bufferlength, int size)
{
	ofstream datustream;
	datustream.open("u.dat", ios::binary | ios::out);
	ofstream datvstream;
	datvstream.open("v.dat", ios::binary | ios::out);
	ofstream datostream;
	datostream.open("o.dat", ios::binary | ios::out);

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
				savePNG(X, Y, &GET(matrixU2, 0, 0, z), 2, "u_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				savePNG(X, Y, &GET(matrixV2, 0, 0, z), 1, "v_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				savePNG(X, Y, &GET(matrixO2, 0, 0, z), 2, "o_step" + to_string(step) + "_Z" + to_string(z) + ".png");
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

		++i;
		});

	auto work = [&](int thrn) {
		while (i < L) {
			for (int z = 1; z < Z - 1; z++)
			{
				for (int y = 1; y < Y - 1; y++)
				{
					for (int x = thrn + 1; x < X - 1; x += THREAD_N)
					{
						calcPoint(matrixU2, matrixV2, matrixO2, matrixU1, matrixV1, matrixO1, x, y, z);
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

	datustream.close();
	datvstream.close();
	datostream.close();
}

int main()
{
	int bufferlength = X * Y * Z;
	int size = bufferlength * sizeof(double);
	double* matrixU1 = new double[bufferlength];
	double* matrixV1 = new double[bufferlength];
	double* matrixO1 = new double[bufferlength];
	double* matrixU2 = new double[bufferlength];
	double* matrixV2 = new double[bufferlength];
	double* matrixO2 = new double[bufferlength];

	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<double> distr(0, 0.1);
	default_random_engine re(1);

	for (int i = 0; i < bufferlength; ++i)
	{
		matrixU1[i] = distr(re) + 1.0;
		matrixV1[i] = matrixV2[i] = 0.0;
		matrixO1[i] = matrixO2[i] = o0;
	}
	calcBoundary(matrixU1, matrixV1, matrixO1);
	calcBoundary(matrixU2, matrixV2, matrixO2);

	iterate(matrixU2, matrixV2, matrixO2, matrixU1, matrixV1, matrixO1, bufferlength, size);

	//double maxU = 3.5;
	//double maxV = 0.7;
}