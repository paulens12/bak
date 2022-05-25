// antras_bandymas.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "png++/png.hpp"
#include <random>
#include <chrono>
#include <ppl.h>

#define W 224
#define L 601
#define LMult 10000
#define THREADS 4
//THREADS turi dalinti W

long double Du = 0.1;
long double chi = 8.3;
long double au = 1;
long double Bv = 0.73;
long double dx = (8.0 * 3.14159265358979323846) / W;
long double dx2 = dx*dx;
long double dt = 0.00005;

int LMult2 = 1;

using namespace std;

long double getNextU(long double prevU, long double prevUL, long double prevUR, long double prevV, long double prevVL, long double prevVR);
long double getNextV(long double prevU, long double prevV, long double prevVL, long double prevVR);

long double matrixU[L][W]; // [eilute][stulpelis]
long double matrixV[L][W]; // [eilute][stulpelis]

long double tempU[LMult+1][W];
long double tempV[LMult+1][W];

int main()
{
	normal_distribution<long double> distr(0, 0.1);
	default_random_engine re(1);
	for (int i = 0; i < W; i++)
	{
		matrixU[0][i] = distr(re) + 1.0;
		matrixV[0][i] = 0;
	}

	//cout << "Iveskite Du: ";
	//cin >> Du;
	//cout << Du << endl << "Iveskite chi: ";
	//cin >> chi;
	//cout << chi << endl << "Iveskite daugikli: ";
	//cin >> LMult2;
	//cout << LMult2 << endl;
	//if (W % THREADS != 0) cout << "DEMESIO!!! THREADS nedalija W!";

	atomic<int> done = 0;
	bool cont[THREADS];
	std::fill(cont, cont + THREADS, true);

	memcpy(tempU[0], matrixU[0], W * sizeof(long double));
	memcpy(tempV[0], matrixV[0], W * sizeof(long double));

	int width = W / THREADS;
	auto start = clock();
	concurrency::parallel_for(0, THREADS, [&](int thn)
	{
		int rangeBegin = thn * width;
		for (int i = 0; i < L - 1; i++)
		{
			for (int jj = 0; jj < LMult2; jj++)
			{
				for (int ii = 1; ii <= LMult; ii++)
				{
					while (!cont[thn]) {}
					cont[thn] = false;
					if (jj == 0 && ii == 1)
					{
						memcpy(tempU[0] + rangeBegin, matrixU[i] + rangeBegin, width * sizeof(long double));
						memcpy(tempV[0] + rangeBegin, matrixV[i] + rangeBegin, width * sizeof(long double));
					}
					else if (ii == 1)
					{
						memcpy(tempU[0] + rangeBegin, tempU[LMult] + rangeBegin, width * sizeof(long double));
						memcpy(tempV[0] + rangeBegin, tempV[LMult] + rangeBegin, width * sizeof(long double));
					}
					for (int j = rangeBegin; j < rangeBegin + width; j++)
					{
						int prev;

						if (j == 0)
							prev = W - 1;
						else
							prev = j - 1;

						int next;
						if (j == W - 1)
							next = 0;
						else
							next = j + 1;

						tempU[ii][j] = getNextU(tempU[ii - 1][j], tempU[ii - 1][prev], tempU[ii - 1][next], tempV[ii - 1][j], tempV[ii - 1][prev], tempV[ii - 1][next]);
						tempV[ii][j] = getNextV(tempU[ii - 1][j], tempV[ii - 1][j], tempV[ii - 1][prev], tempV[ii - 1][next]);
					}
					done++;

					if (thn == 0)
					{
						while (done != THREADS) {}
						done = 0;
						if (ii == LMult)
						{
							memcpy(matrixU[i + 1], tempU[ii], W * sizeof(long double));
							memcpy(matrixV[i + 1], tempV[ii], W * sizeof(long double));
						}
						for (int k = 0; k < THREADS; k++)
						{
							cont[k] = true;
						}
					}
				}
			}
		}
	});
	auto duration = (clock() - start) / (long double)CLOCKS_PER_SEC;
	cout << "gijos: " << THREADS << " trukme: " << duration << endl;

	png::image<png::rgb_pixel> imgU(W, L);
	png::image<png::rgb_pixel> imgV(W, L);

	long double maxU = 0;
	long double maxV = 0;
	for (int i = 0; i < L; i++)
	{
		for (int j = 0; j < W; j++)
		{
			if (matrixU[i][j] > maxU)
				maxU = matrixU[i][j];
			if (matrixV[i][j] > maxV)
				maxV = matrixV[i][j];
		}
	}
	maxU = 4.5;
	maxV = 0.6;
	long double multiU = 255 / maxU;
	long double multiV = 255 / maxV;

	for (int i = 1; i <= L; i++)
	{
		for (int j = 0; j < W; j++)
		{
			int color = min((int)(matrixU[i - 1][j] * multiU), 255);
			imgU[L - i][j] = png::rgb_pixel(color, color, color);
			color = min((int)(matrixV[i - 1][j] * multiV), 255);
			imgV[L - i][j] = png::rgb_pixel(color, color, color);
		}
	}

	string vars = "D" + to_string(Du) + "_chi" + to_string(chi) + "_M" + to_string(LMult2);
	imgU.write("U1_" + vars + ".png");
	imgV.write("V1_" + vars + ".png");
	//cout << "U multiplier: " << multiU << endl << "V multiplier: " << multiV << endl << "U max: " << maxU << endl << "V max: " << maxV << endl;
}

long double getNextU(long double prevU, long double prevUL, long double prevUR, long double prevV, long double prevVL, long double prevVR)
{
	long double prevUL2 = (prevU + prevUL) / 2;
	long double prevUR2 = (prevU + prevUR) / 2;
	return	(
		(prevUR - 2 * prevU + prevUL) * Du / dx2
		- (
			prevUR2 * (prevVR - prevV)
			- prevUL2 * (prevV - prevVL)
			) * chi / dx2
		+ au * prevU * (1 - prevU)
		) * dt + prevU;
}

long double getNextV(long double prevU, long double prevV, long double prevVL, long double prevVR)
{
	return	(
		(prevVR - 2 * prevV + prevVL) / dx2
		+ prevU / (1 + Bv * prevU)
		- prevV
		) * dt + prevV;
}