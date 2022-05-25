// pirmas_bandymas.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <iostream>
#include "png++/png.hpp"
#include <random>
#include <chrono>
#include <ppl.h>

#define W 360
#define H 600
#define LMult 1200

using namespace std;

double getNextU(double prevU, double prevUL, double prevUR, double prevV, double prevVL, double prevVR);
double getNextV(double prevU, double prevUL, double prevUR, double prevV, double prevVL, double prevVR);

double matrixU[H][W]; // [eilute][stulpelis]
double matrixV[H][W]; // [eilute][stulpelis]
double tempU[LMult + 1][W];
double tempV[LMult + 1][W];

int main()
{
	normal_distribution<double> distr(0, 0.1);
	default_random_engine re(1);
	for (int i = 0; i < W; i++)
	{
		matrixU[0][i] = distr(re) + 1.0;
		matrixV[0][i] = 0;
	}

	auto start = clock();
	for (int i = 0; i < H - 1; i++)
	{
		memcpy(tempU[0], matrixU[i], W * sizeof(double));
		memcpy(tempV[0], matrixV[i], W * sizeof(double));
		for (int k = 1; k <= LMult; k++)
		{
			for (int j = 0; j < W; j++)
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

				tempU[k][j] = getNextU(
					tempU[k-1][j],
					tempU[k-1][prev],
					tempU[k-1][next],
					tempV[k - 1][j],
					tempV[k - 1][prev],
					tempV[k - 1][next]
				);
				tempV[k][j] = getNextV(
					tempU[k - 1][j],
					tempU[k - 1][prev],
					tempU[k - 1][next],
					tempV[k - 1][j],
					tempV[k - 1][prev],
					tempV[k - 1][next]
				);
			}
		}
		memcpy(matrixU[i + 1], tempU[LMult], W * sizeof(double));
		memcpy(matrixV[i + 1], tempV[LMult], W * sizeof(double));
	}
	auto duration = (clock() - start) / (double)CLOCKS_PER_SEC;
	cout << "duration: " << duration << endl;

	png::image<png::rgb_pixel> imgU(W, H);
	png::image<png::rgb_pixel> imgV(W, H);

	double maxU = 0;
	double maxV = 0;
	for (int i = 0; i < H; i++)
	{
		for (int j = 0; j < W; j++)
		{
			if (matrixU[i][j] > maxU)
				maxU = matrixU[i][j];
			if (matrixV[i][j] > maxV)
				maxV = matrixV[i][j];
		}
	}
	maxU = 3.5;
	maxV = 0.7;
	double multiU = 255 / maxU;
	double multiV = 255 / maxV;
	for (int i = 0; i < W; i++)
	{
		for (int j = 1; j <= H; j++)
		{
			int color = min((int)(matrixU[j - 1][i] * multiU), 255);
			imgU[H - j][i] = png::rgb_pixel(color, color, color);
			color = min((int)(matrixV[j - 1][i] * multiV), 255);
			imgV[H - j][i] = png::rgb_pixel(color, color, color);
		}
	}
	imgU.write("U1.png");
	imgV.write("V1.png");
	//cout << "U multiplier: " << multiU << endl << "V multiplier: " << multiV << endl << "U max: " << maxU << endl << "V max: " << maxV;
}

double Du = 0.1;
double chi = 8.3;
double au = 1;
double Bv = 0.73;
double dx2 = 0.075*0.075;
double dt = 0.00005;

inline double getNextU(double prevU, double prevUL, double prevUR, double prevV, double prevVL, double prevVR)
{
	double prevUL2 = (prevU + prevUL) / 2;
	double prevUR2 = (prevU + prevUR) / 2;
	return	(
				(prevUR - 2 * prevU + prevUL) * Du / dx2
				- (
					prevUR2 * (prevVR - prevV)
					- prevUL2 * (prevV - prevVL)
				) * chi / dx2
				+ au * prevU * (1 - prevU)
			) * dt + prevU;
}

inline double getNextV(double prevU, double prevUL, double prevUR, double prevV, double prevVL, double prevVR)
{
	return	(
				(prevVR - 2 * prevV + prevVL) / dx2
				+ prevU / (1 + Bv * prevU)
				- prevV
			) * dt + prevV;
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
