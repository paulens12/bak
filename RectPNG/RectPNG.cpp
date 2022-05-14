// RectPNG.cpp : Defines the functions for the static library.
//

#include "gd.h"
#include <string>
#include <stdio.h>
#include <algorithm>
#include <iostream>
using namespace std;

void savePNG(int maxX, int maxY, double* data, double maxValue, string filename) {
	gdImagePtr im = gdImageCreate(maxX, maxY);
	int colors[256]{};
	for (int i = 0; i < 256; i++) {
		colors[i] = gdImageColorAllocate(im, i, i, i);
	}

	/*
	double maxValue = 0.0;
	for (int y = 0; y < maxY; y++) {
		for (int x = 0; x < maxX; x++) {
			if (maxValue < data[x + y * maxX])
				maxValue = data[x + y * maxX];
		}
	}

	cout << "max: " << maxValue << endl;
	*/

	for (int y = 0; y < maxY; y++) {
		bool coutOnError = true;
		for (int x = 0; x < maxX; x++) {
			if (coutOnError && (!isfinite(data[x + y * maxX]) || data[x + y * maxX] < 0)) {
				cout << "X: " << x << " Y: " << y << " bad, file " << filename << "\n";
				//*(int*)0 = 0; // force crash
				coutOnError = false;
			}

			gdImageSetPixel(im, x, y, colors[std::clamp(int(data[x + y * maxX] * 256 / maxValue), 0, 255)]);
		}
	}

	FILE* pngout;
	errno_t err = fopen_s(&pngout, filename.c_str(), "wb");
	if (err) {
		cout << "error: " << err << endl;
		gdImageDestroy(im);
		return;
	}
	gdImagePng(im, pngout);
	fclose(pngout);
	gdImageDestroy(im);
}