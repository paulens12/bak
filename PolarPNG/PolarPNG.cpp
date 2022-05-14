// PolarPNG.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <algorithm>
#include <math.h>
#include "CartesianToPolar.h"
#include "PolarPNG.h"
#include "gd.h"

PolarPNG::PolarPNG(int maxR, int factor, int fSteps, double maxValue) :_ctp(maxR, factor, fSteps)
{
	_maxValue = maxValue;
	_maxR = maxR;
	_fSteps = fSteps;
}

double* PolarPNG::fillBlanks(double* data) {
	double* buffer = new double[_fSteps * _maxR];
	memcpy(buffer, data, sizeof(double) * _fSteps * _maxR);
	for (int r = 0; r < _maxR / 4; r++) {
		for (int f = 0; f < _fSteps - 3; f += 4) {
			double begin = buffer[r * _fSteps + f];
			double end = buffer[r * _fSteps + (f + 4) % _fSteps];
			double mid = buffer[r * _fSteps + f + 2] = (begin + end) / 2;
			buffer[r * _fSteps + f + 1] = (begin + mid) / 2;
			buffer[r * _fSteps + f + 3] = (end + mid) / 2;
		}
	}
	for (int r = _maxR / 4; r < _maxR / 2; r++) {
		for (int f = 0; f < _fSteps; f += 2) {
			buffer[r * _fSteps + f + 1] = (buffer[r * _fSteps + f] + buffer[r * _fSteps + (f + 2) % _fSteps]) / 2;
		}
	}
	return buffer;
}

void PolarPNG::savePNG(double* data, std::string filename)
{
	double* buffer = fillBlanks(data);

	int imgSize = _ctp.getWidth();

	gdImagePtr im = gdImageCreate(imgSize, imgSize);
	int colors[256]{};
	for (int i = 0; i < 256; i++) {
		colors[i] = gdImageColorAllocate(im, i, i, i);
	}
	for (int y = 0; y < imgSize; y++) {
		bool coutOnError = true;
		for (int x = 0; x < imgSize; x++) {
			double val = _ctp.getValue(buffer, x, y);
			if (coutOnError && (!isfinite(val) || val < 0)) {
				std::cout << "X: " << x << " Y: " << y << " bad, file " << filename << std::endl;
				//*(int*)0 = 0; // force crash
				coutOnError = false;
			}
			gdImageSetPixel(im, x, y, colors[std::clamp(int(val * 256 / _maxValue), 0, 255)]);
		}
	}

	delete[] buffer;

	FILE* pngout;
	errno_t err = fopen_s(&pngout, filename.c_str(), "wb");
	if (err) {
		std::cout << "error opening png: " << err << std::endl;
		gdImageDestroy(im);
		return;
	}
	gdImagePng(im, pngout);
	fclose(pngout);
	gdImageDestroy(im);
}

void PolarPNG::saveFlat(double* data, std::string filename)
{
	double* buffer = fillBlanks(data);

	delete[] buffer;
}
