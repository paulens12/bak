
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <Constants.h>
#include <UV.h>

using namespace std;

//#define H 128
//#define L 60000
//#define SNAPSHOT_STEP 10000

#define L 600
#define SNAPSHOT_STEP 1


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

	errno_t err;

	double* matrixU = new double[bufferlength * (L / SNAPSHOT_STEP + 1)]; // for gif output
	double* matrixV = new double[bufferlength * (L / SNAPSHOT_STEP + 1)]; // for gif output
	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<double> distr(0, 0.1);
	default_random_engine re(1);

	for (int i = 0; i < bufferlength; ++i)
	{
		if (i < bufferlength / 2 && i % 2 != 0 || i < bufferlength / 4 && i % 4 != 0)
			matrixU[i] = matrixU1[i] = matrixU2[i] = 0;
		else
			matrixU[i] = matrixU1[i] = distr(re) + 1.0;

		matrixV[i] = matrixV1[i] = matrixV2[i] = 0;
	}
	calcBoundary(matrixU1, matrixV1);
	calcBoundary(matrixU, matrixV);

	auto start = clock();
	double* temp = NULL;
	for (int i = 0; i < L; ++i)
	{

		calcMatrix(matrixU2, matrixV2, matrixU1, matrixV1);

		calcBoundary(matrixU2, matrixV2);

		if (i % SNAPSHOT_STEP == SNAPSHOT_STEP - 1)
		{
			int step = i / SNAPSHOT_STEP + 1;
			//save frame
			double elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;
			cout << "step " << step << ", time elapsed: " << elapsed << ", avg: " << elapsed / step << endl;
			err = memcpy_s(matrixU + step * bufferlength, size, matrixU2, size);
			if (err)
				cout << "error: " << err << endl;
			err = memcpy_s(matrixV + step * bufferlength, size, matrixV2, size);
			if (err)
				cout << "error: " << err << endl;
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

	//double maxU = 3.5;
	//double maxV = 0.7;
	//double multiU = 255 / maxU;
	//double multiV = 255 / maxV;
	ofstream datustream;
	datustream.open("u.dat", ios::binary | ios::out);
	if (datustream.is_open())
	{
		datustream.write((char*)matrixU, size * (L / SNAPSHOT_STEP + 1));
		datustream.close();
	}
	ofstream datvstream;
	datvstream.open("v.dat", ios::binary | ios::out);
	if (datvstream.is_open())
	{
		datvstream.write((char*)matrixV, size * (L / SNAPSHOT_STEP + 1));
		datvstream.close();
	}

	cout << "start csv" << endl;
	FILE* csvu, * csvv;
	err = fopen_s(&csvu, "u.csv", "w");
	if (err) return err;
	err = fopen_s(&csvv, "v.csv", "w");
	if (err) return err;
	for (int j = 0; j <= L / SNAPSHOT_STEP; j++)
	{
		for (int i = 0; i < bufferlength; i += 1)
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