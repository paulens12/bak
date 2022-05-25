
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <Constants.h>
#include <UV.h>
#include "../PolarPNG/PolarPNG.h"

using namespace std;

//#define H 128
//#define L 60000
//#define SNAPSHOT_STEP 10000

#define L 300000
#define SNAPSHOT_STEP 100000


// perform one iteration of simulation
void calcMatrix(double* uOutput, double* vOutput, double* uInput, double* vInput)
{
	double ro;
	for (int r = 1; r < R - 1; ++r)
	{
		ro = r * dr;
		for (int f = 0; f < F; ++f)
		{
			calcPoint(uOutput, vOutput, uInput, vInput, ro, r, f);
		}
	}
}

int main()
{
	double* matrixU1, * matrixU2, * matrixV1, * matrixV2;
	int bufferlength = R * F;
	int size = bufferlength * sizeof(double);
	matrixU1 = new double[bufferlength];
	matrixV1 = new double[bufferlength];
	matrixU2 = new double[bufferlength];
	matrixV2 = new double[bufferlength];

	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<double> distr(0, 0.1);
	default_random_engine re(1);

	for (int i = 0; i < bufferlength; ++i)
	{
		if (i < bufferlength / 2 && i % 2 != 0 || i < bufferlength / 4 && i % 4 != 0)
			matrixU1[i] = matrixU2[i] = 0;
		else
			matrixU2[i] = matrixU1[i] = distr(re) + 1.0;

		matrixV1[i] = matrixV2[i] = 0;
	}
	calcBoundary(matrixU1, matrixV1);
	calcBoundary(matrixU2, matrixV2);

	PolarPNG uPng(R, 3, F, 4.5);
	PolarPNG vPng(R, 3, F, 0.6);

	uPng.savePNG(matrixU1, "u_step0.png");
	vPng.savePNG(matrixV1, "v_step0.png");

	ofstream datustream;
	datustream.open("u.dat", ios::binary | ios::out);
	ofstream datvstream;
	datvstream.open("v.dat", ios::binary | ios::out);

	if (datustream.is_open())
		datustream.write((char*)matrixU1, size);
	if (datvstream.is_open())
		datvstream.write((char*)matrixV1, size);

	auto start = clock();
	auto start_current = start;
	double* temp = nullptr;
	for (int i = 0; i < L; ++i)
	{

		calcMatrix(matrixU2, matrixV2, matrixU1, matrixV1);

		calcBoundary(matrixU2, matrixV2);

		if (i % SNAPSHOT_STEP == SNAPSHOT_STEP - 1)
		{
			int step = i / SNAPSHOT_STEP + 1;
			// save frame
			clock_t saveframe = clock();

			if (datustream.is_open())
				datustream.write((char*)matrixU2, size);
			if (datvstream.is_open())
				datvstream.write((char*)matrixV2, size);

			uPng.savePNG(matrixU2, "u_step" + to_string(step) + ".png");
			vPng.savePNG(matrixV2, "v_step" + to_string(step) + ".png");
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
	datustream.close();
	datvstream.close();
}