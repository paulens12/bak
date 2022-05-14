
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <Constants.h>
#include <UV.h>
#include "../RectPNG/RectPNG.h"

using namespace std;

//#define H 128
//#define L 60000
//#define SNAPSHOT_STEP 10000

#define L 60000
#define SNAPSHOT_STEP 1000


// perform one iteration of simulation
void calcMatrix(double* uOutput, double* vOutput, double* oOutput, double* uInput, double* vInput, double* oInput)
{
	for (int z = 1; z < Z_cube - 1; z++)
	{
		for (int y = 1; y < Y - 1; y++)
		{
			for (int x = 1; x < X - 1; x++)
			{
				calcPoint(uOutput, vOutput, oOutput, uInput, vInput, oInput, x, y, z);
			}
		}
	}
}

int main()
{
	int bufferlength = X * Y * Z_cube;
	int size = bufferlength * sizeof(double);
	double* matrixU1 = new double[bufferlength];
	double* matrixV1 = new double[bufferlength];
	double* matrixO1 = new double[bufferlength];
	double* matrixU2 = new double[bufferlength];
	double* matrixV2 = new double[bufferlength];
	double* matrixO2 = new double[bufferlength];

	errno_t err;

	double* matrixU = new double[bufferlength * (L / SNAPSHOT_STEP + 1)]; // for gif output
	double* matrixV = new double[bufferlength * (L / SNAPSHOT_STEP + 1)]; // for gif output
	double* matrixO = new double[bufferlength * (L / SNAPSHOT_STEP + 1)]; // for gif output
	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<double> distr(0, 0.1);
	default_random_engine re(1);

	for (int i = 0; i < bufferlength; ++i)
	{
		matrixU[i] = matrixU1[i] = distr(re) + 1.0;
		matrixV[i] = matrixV1[i] = matrixV2[i] = 0.0;
		matrixO[i] = matrixO1[i] = matrixO2[i] = o0;
	}
	calcBoundary(matrixU1, matrixV1, matrixO1);
	calcBoundary(matrixU, matrixV, matrixO);


	ofstream datustream;
	datustream.open("u.dat", ios::binary | ios::out);
	ofstream datvstream;
	datvstream.open("v.dat", ios::binary | ios::out);
	ofstream datostream;
	datostream.open("o.dat", ios::binary | ios::out);

	clock_t start = clock();
	clock_t start_current = start;
	double* temp = NULL;
	for (int i = 0; i < L; ++i)
	{
		calcMatrix(matrixU2, matrixV2, matrixO2, matrixU1, matrixV1, matrixO1);

		calcBoundary(matrixU2, matrixV2, matrixO2);

		if (i % SNAPSHOT_STEP == SNAPSHOT_STEP - 1)
		{
			int step = i / SNAPSHOT_STEP + 1;
			//save frame
			clock_t saveframe = clock();

			if (datustream.is_open())
				datustream.write((char*)matrixU2, size);
			if (datvstream.is_open())
				datvstream.write((char*)matrixV2, size);
			if (datostream.is_open())
				datostream.write((char*)matrixO2, size);

			for (int z : { 0, Z_cube / 2, Z_cube - 1 }) {
				savePNG(X, Y, &GET(matrixU2, 0, 0, z), 2, "u_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				savePNG(X, Y, &GET(matrixV2, 0, 0, z), 1, "v_step" + to_string(step) + "_Z" + to_string(z) + ".png");
				savePNG(X, Y, &GET(matrixO2, 0, 0, z), 1, "o_step" + to_string(step) + "_Z" + to_string(z) + ".png");
			}
			double done = clock();

			double processedIn = (saveframe - start_current) / (double)CLOCKS_PER_SEC;
			double outputIn = (done - saveframe) / (double)CLOCKS_PER_SEC;
			double totalTime = (done - start) / (double)CLOCKS_PER_SEC;
			cout << "step " << step << ", done processing in: " << processedIn << ", saved snapshot in:" << outputIn << ", total: " << totalTime << endl;
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

	//double maxU = 3.5;
	//double maxV = 0.7;
	//double multiU = 255 / maxU;
	//double multiV = 255 / maxV;
	//ofstream datustream;
	//datustream.open("u.dat", ios::binary | ios::out);
	//if (datustream.is_open())
	//{
	//	datustream.write((char*)matrixU, size * (L / SNAPSHOT_STEP + 1));
	//	datustream.close();
	//}
	//ofstream datvstream;
	//datvstream.open("v.dat", ios::binary | ios::out);
	//if (datvstream.is_open())
	//{
	//	datvstream.write((char*)matrixV, size * (L / SNAPSHOT_STEP + 1));
	//	datvstream.close();
	//}
	//ofstream datostream;
	//datostream.open("o.dat", ios::binary | ios::out);
	//if (datostream.is_open())
	//{
	//	datostream.write((char*)matrixO, size * (L / SNAPSHOT_STEP + 1));
	//	datostream.close();
	//}

	datustream.close();
	datvstream.close();
	datostream.close();

	cout << "start csv" << endl;
	FILE* csvu, * csvv, * csvo;
	err = fopen_s(&csvu, "u.csv", "w");
	if (err) return err;
	err = fopen_s(&csvv, "v.csv", "w");
	if (err) return err;
	err = fopen_s(&csvo, "o.csv", "w");
	if (err) return err;
	for (int j = 0; j <= L / SNAPSHOT_STEP; j++)
	{
		double uMax = matrixU[j * bufferlength];
		double vMax = matrixV[j * bufferlength];
		double oMax = matrixO[j * bufferlength];
		double uMin = matrixU[j * bufferlength];
		double vMin = matrixV[j * bufferlength];
		double oMin = matrixO[j * bufferlength];
		for (int i = 1; i < bufferlength; i++)
		{
			if (matrixU[j * bufferlength + i] < uMin) uMin = matrixU[j * bufferlength + i];
			if (matrixU[j * bufferlength + i] > uMax) uMax = matrixU[j * bufferlength + i];
			if (matrixV[j * bufferlength + i] < vMin) vMin = matrixV[j * bufferlength + i];
			if (matrixV[j * bufferlength + i] > vMax) vMax = matrixV[j * bufferlength + i];
			if (matrixO[j * bufferlength + i] < oMin) oMin = matrixO[j * bufferlength + i];
			if (matrixO[j * bufferlength + i] > oMax) oMax = matrixO[j * bufferlength + i];

			if (j > L / SNAPSHOT_STEP - 2) {
				fprintf(csvu, "%f;", matrixU[j * bufferlength + i]);
				fprintf(csvv, "%f;", matrixV[j * bufferlength + i]);
				fprintf(csvo, "%f;", matrixO[j * bufferlength + i]);
			}
		}
		fprintf(csvu, "%f;%f\n", uMin, uMax);
		fprintf(csvv, "%f;%f\n", vMin, vMax);
		fprintf(csvo, "%f;%f\n", oMin, oMax);
	}

	fclose(csvu);
	fclose(csvv);
	fclose(csvo);
}