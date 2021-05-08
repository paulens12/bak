
#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

const double Du = 0.1;
const double chi = 8.3;
const double au = 1;
const double Bv = 0.73;
const int R = 60; /* 2pi(R*dr) = 360*dx (cartesian sim) */
const int F = 200;
const double dt = 0.00005;
const double dr = 9 / (M_PI * R);
const double dr2 = dr * dr;
//#define dr (0.04774648292756860073066512901175) /* 3/(20pi) */
//#define dr2 (0.00227972663195259985748728792222) /* dr * dr */
const double BASE_df = 2 * M_PI / F;

#define FRAME_DURATION 6
//#define H 128
//#define L 3456000
#define L 360000
#define SNAPSHOT_STEP 1000


bool arnan(double x) {
	return isnan(x);
}

// fp - f+1
// fm - f-1
// rp - r+1
// rm - r-1
inline double getNextU(
	double u, double urp, double urm, double ufp, double ufm,
	double v, double vrp, double vrm, double vfp, double vfm,
	int r, double df
)
{
	double urp2 = (u + urp) / 2;
	double urm2 = (u + urm) / 2;
	double ufp2 = (u + ufp) / 2;
	double ufm2 = (u + ufm) / 2;
	double ro = r * dr + dr;
	double rp2 = r * dr + 3 * dr / 2;
	double rm2 = r * dr + dr / 2;
	//double dr1 = (r == 0) ? (dr * 2) : dr;
	//double ro = r * dr;
	//double rp2 = r * dr + dr/2;
	//double rm2 = r * dr - dr/2;

	double result = dt * (
		Du * ((
			(rp2 * (urp - u) - rm2 * (u - urm))
				/ (ro * dr2)
			) + (
				(ufp - 2 * u + ufm) / (ro * ro * df * df)
				)
			) - chi * (
				(rp2 * urp2 * (vrp - v) - rm2 * urm2 * (v - vrm))
					/ (ro * dr2)
				+ (ufp2 * (vfp - v) - ufm2 * (v - vfm)) / (ro * ro * df * df)
				) + au * u * (1 - u)
		) + u;
	if (!isfinite(result) || result < 0)
	{
		if(_fpclass(result) != _FPCLASS_NINF && _fpclass(result) != _FPCLASS_PINF)
			cout << "error " << _fpclass(result) << endl;
	}
	return result;
}

inline double getNextV(
	double u,
	double v, double vrp, double vrm, double vfp, double vfm,
	int r, double df
)
{
	//double ro = r * dr + dr / 2;
	//double rp2 = r * dr + dr;
	//double rm2 = r * dr;

	double ro = r * dr + dr;
	double rp2 = r * dr + 3 * dr / 2;
	double rm2 = r * dr + dr / 2;

	//double ro = r * dr + dr;
	//double rp2 = r * dr + 3 * dr / 2;
	//double rm2 = r * dr + dr / 2;
	//double dr1 = (r == 0) ? (dr * 2) : dr;

	double result = dt * (
		((rp2 * (vrp - v) - rm2 * (v - vrm)) / dr2
			+
			(vfp - 2 * v + vfm) / (ro * df * df)
			) / ro + u / (1 + Bv * u) - v
		) + v;
	if (!isfinite(result) || result < 0)
	{
		if (_fpclass(result) != _FPCLASS_NINF && _fpclass(result) != _FPCLASS_PINF)
			cout << "error " << _fpclass(result) << endl;
	}
	return result;
}

void boundaryKernel(double* u, double* v)
{
	for (int f = 0; f < F; ++f)
	{ // no-flux boundary condition
		u[3 * R * F / 16 + (R / 2 - 1) * F + f] = (4 * u[3 * R * F / 16 + (R / 2 - 2) * F + f] - u[3 * R * F / 16 + (R / 2 - 3) * F + f]) / 3;
		v[3 * R * F / 16 + (R / 2 - 1) * F + f] = (4 * v[3 * R * F / 16 + (R / 2 - 2) * F + f] - v[3 * R * F / 16 + (R / 2 - 3) * F + f]) / 3;
	}


	for (int f = 0; f < F / 8; ++f)
	{ // central symmetry
		u[F / 8 + f] = u[f] = (u[F / 4 + f] + u[3 * F / 8 + f]) / 2;
		v[F / 8 + f] = v[f] = (v[F / 4 + f] + v[3 * F / 8 + f]) / 2;
	}
}

void calcKernel(double* uOutput, double* vOutput, double* uInput, double* vInput)
{
	double urp, urm, vrp, vrm, ufp, ufm, vfp, vfm, u, v;
	int fp, fm; // offsets
	double df;

	for (int r = 1; r < R / 4; ++r)
	{
		df = BASE_df * 4;
		for (int f = 0; f < F / 4; ++f)
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

			urm = uInput[(r - 1) * F / 4 + f];
			vrm = vInput[(r - 1) * F / 4 + f];

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

			uOutput[r * F / 4 + f] = getNextU(u, urp, urm, ufp, ufm, v, vrp, vrm, vfp, vfm, r, df);
			vOutput[r * F / 4 + f] = getNextV(u, v, vrp, vrm, vfp, vfm, r, df);
		}
	}

	for (int r = R / 4; r < R / 2; ++r)
	{
		df = BASE_df * 2;
		for (int f = 0; f < F / 2; ++f)
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

			uOutput[R * F / 16 + (r - R / 4) * F / 2 + f] = getNextU(u, urp, urm, ufp, ufm, v, vrp, vrm, vfp, vfm, r, df);
			vOutput[R * F / 16 + (r - R / 4) * F / 2 + f] = getNextV(u, v, vrp, vrm, vfp, vfm, r, df);
		}
	}

	for (int r = R / 2; r < R - 1; ++r)
	{
		for (int f = 0; f < F; ++f)
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

			uOutput[3 * R * F / 16 + (r - R / 2) * F + f] = getNextU(u, urp, urm, ufp, ufm, v, vrp, vrm, vfp, vfm, r, BASE_df);
			vOutput[3 * R * F / 16 + (r - R / 2) * F + f] = getNextV(u, v, vrp, vrm, vfp, vfm, r, BASE_df);
		}
	}
}

int main()
{
	double* matrixU1, * matrixU2, * matrixV1, * matrixV2;
	int bufferlength = 11 * R * F / 16;
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

	for (int i = 0; i < bufferlength; i++)
	{
		matrixU[i] = matrixU1[i] = distr(re) + 1.0;
		matrixV[i] = matrixV1[i] = 0;
	}

	auto start = clock();
	double* temp = NULL;
	for (int i = 0; i < L; ++i)
	{
		if (i % SNAPSHOT_STEP == SNAPSHOT_STEP - 1)
		{
			int step = i / SNAPSHOT_STEP + 1;
			//save frame
			double elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;
			cout << "step " << step << ", time elapsed: " << elapsed << ", avg: " << elapsed / step << endl;
			err = memcpy_s(matrixU + step * bufferlength, size, matrixU1, size);
			if (err)
				cout << "error: " << err << endl;
			err = memcpy_s(matrixV + step * bufferlength, size, matrixV1, size);
			if (err)
				cout << "error: " << err << endl;
			//for (int j = 0; j < bufferlength; ++j)
			//{
			//	matrixU[step * bufferlength + j] = matrixU1[j];
			//	matrixV[step * bufferlength + j] = matrixV1[j];
			//}
		}

		calcKernel(matrixU2, matrixV2, matrixU1, matrixV1);

		boundaryKernel(matrixU2, matrixV2);


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
	FILE* datu, * datv;
	err = fopen_s(&datu, "u.dat", "w");
	if (!err)
	{
		fwrite(matrixU, 1, size * (L / SNAPSHOT_STEP + 1), datu);
		fclose(datu);
	}
	err = fopen_s(&datv, "v.dat", "w");
	if (!err)
	{
		fwrite(matrixV, 1, size * (L / SNAPSHOT_STEP + 1), datv);
		fclose(datv);
	}

	FILE* csvu, * csvv;
	err = fopen_s(&csvu, "u.csv", "w");
	if (err) return err;
	err = fopen_s(&csvv, "v.csv", "w");
	if (err) return err;
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