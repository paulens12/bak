// pirmas_bandymas.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "gif.h"
#include "png++/png.hpp"
#include <random>
#include <chrono>
#include <ppl.h>
#include <string>

#define SKIP_FRAMES 1
#define FRAME_DURATION 1
#define W 224
#define H 64
#define TOP 5
#define L 601
#define LMult 10000
//#define LMult2 2
#define THREADS 4
//THREADS turi dalinti W

long double Du = 0.1;
long double chi = 8.3;
long double au = 1;
long double Bv = 0.73;
long double dx = (8.0 * 3.14159265358979323846) / W;
long double dx2 = dx * dx;
long double dy2 = 0.125*0.125;
long double dt = 0.00005;

int LMult2 = 1;

using namespace std;

inline long double getNextU(long double u, long double ul, long double ur, long double uu, long double ud, long double v, long double vl, long double vr, long double vu, long double vd);
inline long double getNextV(long double u, long double v, long double vl, long double vr, long double vu, long double vd);

long double bufferU1[H*W];
long double bufferV1[H*W];
long double bufferU2[H*W];
long double bufferV2[H*W];

int main()
{
	long double* tempU1 = bufferU1;
	long double* tempV1 = bufferV1;
	long double* tempU2 = bufferU2;
	long double* tempV2 = bufferV2;

	long double* temp;
	auto seed = chrono::system_clock::now().time_since_epoch().count();
	normal_distribution<long double> distr(0, 0.1);
	default_random_engine re(1);
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			tempU1[i * W + j] = distr(re) + 1.0;
			tempV1[i * W + j] = 0;
		}
	}

	//cout << "Iveskite Du: ";
	//cin >> Du;
	//cout << Du << endl << "Iveskite chi: ";
	//cin >> chi;
	//cout << chi << endl << "Iveskite daugikli: ";
	//cin >> LMult2;
	//cout << LMult2 << endl;
	//if (W % THREADS != 0) cout << "DEMESIO!!! THREADS nedalija W!";

	long double maxU = 4.5;
	long double maxV = 0.6;
	long double multiU = 256 / maxU;
	long double multiV = 256 / maxV;

	uint8_t frameU[W * H * 4];
	uint8_t frameV[W * H * 4];

	string vars = "D" + to_string(Du) + "_chi" + to_string(chi) + "_M" + to_string(LMult2) + "dx0.05";

	GifWriter gu;
	GifWriter gv;
	GifBegin(&gu, ("U2_" + vars + ".gif").c_str(), W, H, FRAME_DURATION);
	GifBegin(&gv, ("V2_" + vars + ".gif").c_str(), W, H, FRAME_DURATION);

	png::image<png::rgb_pixel> imgU(W, L);
	png::image<png::rgb_pixel> imgV(W, L);

	for (int i = 0; i < H * W; i++)
	{
		frameU[i * 4 + 3] = 0;
		frameV[i * 4 + 3] = 0;
	}
	long double sumU, sumV;
	uint8_t colorU, colorV;

	for (int j = 0; j < W; j++)
	{
		sumU = 0.0;
		sumV = 0.0;
		for (int h = 0; h < H; h++)
		{
			if (h < TOP)
			{
				sumU += tempU1[h * W + j];
				sumV += tempV1[h * W + j];
			}
			colorU = min((int)(multiU * tempU1[h * W + j]), 255);
			colorV = min((int)(multiV * tempV1[h * W + j]), 255);
			frameU[4 * (W * h + j)] = colorU;
			frameU[4 * (W * h + j) + 1] = colorU;
			frameU[4 * (W * h + j) + 2] = colorU;
			frameV[4 * (W * h + j)] = colorV;
			frameV[4 * (W * h + j) + 1] = colorV;
			frameV[4 * (W * h + j) + 2] = colorV;
		}
		colorU = min((int)(multiU * sumU / TOP), 255);
		colorV = min((int)(multiV * sumV / TOP), 255);
		imgU[L - 1][j] = png::rgb_pixel(colorU, colorU, colorU);
		imgV[L - 1][j] = png::rgb_pixel(colorV, colorV, colorV);
	}
	GifWriteFrame(&gu, frameU, W, H, FRAME_DURATION);
	GifWriteFrame(&gv, frameV, W, H, FRAME_DURATION);


	atomic<int> done = 0;
	bool cont[THREADS];
	std::fill(cont, cont + THREADS, true);

	int width = W / THREADS;
	auto start = clock();
	concurrency::parallel_for(0, THREADS, [&](int thn)
		{
			int rangeBegin = thn * width;
			for (int i = 0; i < L - 1; i++)
			{
				for (int ii = 1; ii <= LMult; ii++)
				{
					while (!cont[thn]) {}
					cont[thn] = false;
					for (int w = rangeBegin; w < rangeBegin + width; w++)
					{
						for (int h = 1; h < H - 1; h++)
						{
							int l, r, u, d;

							if (w == 0)
								l = W - 1;
							else
								l = w - 1;
							if (w == W - 1)
								r = 0;
							else
								r = w + 1;
							d = h - 1;
							u = h + 1;

							tempU2[h * W + w] = getNextU(tempU1[h * W + w], tempU1[h * W + l], tempU1[h * W + r], tempU1[u * W + w], tempU1[d * W + w], tempV1[h * W + w], tempV1[h * W + l], tempV1[h * W + r], tempV1[u * W + w], tempV1[d * W + w]);
							tempV2[h * W + w] = getNextV(tempU1[h * W + w], tempV1[h * W + w], tempV1[h * W + l], tempV1[h * W + r], tempV1[u * W + w], tempV1[d * W + w]);
						}
						tempU2[w] = (4.0 * tempU2[1 * W + w] - tempU2[2 * W + w]) / 3;
						tempV2[w] = (4.0 * tempV2[1 * W + w] - tempV2[2 * W + w]) / 3;
						tempU2[(H - 1) * W + w] = (4.0 * tempU2[(H - 2) * W + w] - tempU2[(H - 3) * W + w]) / 3;
						tempV2[(H - 1) * W + w] = (4.0 * tempV2[(H - 2) * W + w] - tempV2[(H - 3) * W + w]) / 3;
					}
					done++;

					if (thn == 0)
					{
						while (done != THREADS) {}
						done = 0;

						if (ii == LMult)
						{
							for (int j = 0; j < W; j++)
							{
								sumU = 0.0;
								sumV = 0.0;
								for (int h = 0; h < H; h++)
								{
									if (h < TOP)
									{
										sumU += tempU2[h * W + j];
										sumV += tempV2[h * W + j];
									}
									colorU = min((int)(multiU * tempU2[h * W + j]), 255);
									colorV = min((int)(multiV * tempV2[h * W + j]), 255);
									frameU[4 * (W * h + j)] = colorU;
									frameU[4 * (W * h + j) + 1] = colorU;
									frameU[4 * (W * h + j) + 2] = colorU;
									frameV[4 * (W * h + j)] = colorV;
									frameV[4 * (W * h + j) + 1] = colorV;
									frameV[4 * (W * h + j) + 2] = colorV;
								}
								colorU = min((int)(multiU * sumU / TOP), 255);
								colorV = min((int)(multiV * sumV / TOP), 255);
								imgU[L - i - 2][j] = png::rgb_pixel(colorU, colorU, colorU);
								imgV[L - i - 2][j] = png::rgb_pixel(colorV, colorV, colorV);
							}
							GifWriteFrame(&gu, frameU, W, H, FRAME_DURATION);
							GifWriteFrame(&gv, frameV, W, H, FRAME_DURATION);

							double elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;
							cout << "step " << i << ", time elapsed: " << elapsed << ", avg: " << elapsed / (i + 1) << endl;
							//if (i % 10 == 0)
							//	cout << i << endl;
						}

						temp = tempU1;
						tempU1 = tempU2;
						tempU2 = temp;

						temp = tempV1;
						tempV1 = tempV2;
						tempV2 = temp;

						for (int k = 0; k < THREADS; k++)
							cont[k] = true;
					}
				}
			}
		});
	auto duration = (clock() - start) / (long double)CLOCKS_PER_SEC;
	cout << "gijos: " << THREADS << " trukme: " << duration << endl;

	GifEnd(&gu);
	GifEnd(&gv);

	imgU.write("U2_" + vars + ".png");
	imgV.write("V2_" + vars + ".png");

	//cout << "U multiplier: " << multiU << endl << "V multiplier: " << multiV << endl << "U max: " << maxU << endl << "V max: " << maxV;
}

inline long double getNextU(long double u, long double ul, long double ur, long double uu, long double ud, long double v, long double vl, long double vr, long double vu, long double vd)
{
	long double ul2 = (u + ul) / 2;
	long double ur2 = (u + ur) / 2;
	long double uu2 = (u + uu) / 2;
	long double ud2 = (u + ud) / 2;

	double result = dt * (
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
	if (!isfinite(result) || result < 0)
	{
		if (_fpclass(result) != _FPCLASS_NINF && _fpclass(result) != _FPCLASS_PINF)
			cout << "error " << _fpclass(result) << endl;
	}
	return result;
}

inline long double getNextV(long double u, long double v, long double vl, long double vr, long double vu, long double vd)
{
	double result = dt * (
		(vr - 2 * v + vl) / dx2
		+ (vu - 2 * v + vd) / dy2
		+ u / (1 + Bv * u)
		- v
		) + v;
	if (!isfinite(result) || result < 0)
	{
		if (_fpclass(result) != _FPCLASS_NINF && _fpclass(result) != _FPCLASS_PINF)
			cout << "error " << _fpclass(result) << endl;
	}
	return result;
}


// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
