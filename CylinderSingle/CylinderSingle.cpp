
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

#define L 6000000
#define SNAPSHOT_STEP 10000


// perform one iteration of simulation
void calcMatrix(double* uOutput, double* vOutput, double* oOutput, double* uInput, double* vInput, double* oInput)
{
	double ro;
	for (int z = 1; z < Z; ++z)
	{
		for (int r = 1; r < R - 1; ++r)
		{
			ro = r * dr;
			for (int f = 0; f < F; ++f)
			{
				if (z == Z - 1) {
					if (r < R - 1)
						oOutput[R * F * (Z - 1) + F * r + f] = calcBoundaryO(oInput, uInput[R * F * (Z - 1) + F * r + f], ro, r, f);
				}
				else
					calcPoint(uOutput, vOutput, oOutput, uInput, vInput, oInput, ro, r, f, z);
			}
		}
	}
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
	double* temp = NULL;
	for (int i = 0; i < L; ++i)
	{

		calcMatrix(matrixU2, matrixV2, matrixO2, matrixU1, matrixV1, matrixO1);

		calcBoundary(matrixU2, matrixV2, matrixO2);

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
	}
	auto duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "duration: " << duration << endl;
}