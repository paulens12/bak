// PolarPNG.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <string>
#include <algorithm>
#include <math.h>
#include "CartesianToPolar.h"
#include "PolarPNG.h"
#include "gd.h"

PolarPNG::PolarPNG(int maxR, int factor, int fSteps, double maxValue, int height, std::string gifCircle, std::string gifSide) :_ctp(maxR, factor, fSteps)
{
	_maxValue = maxValue;
	_maxR = maxR;
	_fSteps = fSteps;
	_height = height;

	int imgSize = _ctp.getWidth();
	im = gdImageCreate(imgSize, imgSize);
	for (int i = 0; i < 256; i++) {
		colors[i] = gdImageColorAllocate(im, i, i, i);
	}

	imSide = gdImageCreate(_fSteps, _height);
	for (int i = 0; i < 256; i++) {
		colorsSide[i] = gdImageColorAllocate(imSide, i, i, i);
	}

	if (!gifCircle.empty()) {
		FILE* out;
		errno_t err = fopen_s(&out, gifCircle.c_str(), "wb");
		if (err) {
			std::cout << "failed to open circle gif file: " << err << std::endl;
		}
		else {
			circleGif = out;
			gdImageGifAnimBegin(im, circleGif, -1, -1);
		}
	}

	if (!gifSide.empty()) {
		FILE* out;
		errno_t err = fopen_s(&out, gifSide.c_str(), "wb");
		if (err) {
			std::cout << "failed to open side gif file: " << err << std::endl;
		}
		else {
			sideGif = out;
			gdImageGifAnimBegin(imSide, sideGif, -1, -1);
		}
	}
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

	gdImagePtr imPrev = nullptr;
	if (sideGif != nullptr)
		imPrev = gdImageClone(im);

	double max = 0;

	int imgSize = _ctp.getWidth();
	for (int y = 0; y < imgSize; y++) {
		bool coutOnError = true;
		for (int x = 0; x < imgSize; x++) {
			double val = _ctp.getValue(buffer, x, y);
			if (coutOnError && (!isfinite(val) || val < 0)) {
				std::cout << "X: " << x << " Y: " << y << " bad, file " << filename << std::endl;
				//*(int*)0 = 0; // force crash
				coutOnError = false;
			}
			if (val > max) max = val;
			gdImageSetPixel(im, x, y, colors[std::clamp(int(val * 256 / _maxValue), 0, 255)]);
		}
	}
	if(max > _maxValue)
		std::cout << "real max: " << max << std::endl;

	std::vector<int> row(_fSteps);

	for (int f = 0; f < _fSteps; f++) {
		double val = 0;
		for (int i = 1; i < 5; i++)
			val += buffer[_fSteps * (_maxR - i) + f];
		val /= 5;
		row[f] = std::clamp(int(val * 256 / _maxValue), 0, 255);
	}

	_flatFrames.push_back(row);

	delete[] buffer;

	if (circleGif != nullptr) {
		gdImageGifAnimAdd(im, circleGif, 0, 0, 0, 1, gdDisposalNone, imPrev);
		gdFree(imPrev);
	}

	FILE* pngout;
	errno_t err = fopen_s(&pngout, filename.c_str(), "wb");
	if (err) {
		std::cout << "error opening png: " << err << std::endl;
		return;
	}
	gdImagePng(im, pngout);
	fclose(pngout);
}

void PolarPNG::saveSide(double* data, std::string filename)
{
	int imgHeight = _height;
	int imgWidth = _fSteps;

	gdImagePtr imPrev = nullptr;
	if (sideGif != nullptr) 
		imPrev = gdImageClone(imSide);

	for (int y = 0; y < _height; y++) {
		for (int x = 0; x < _fSteps; x++) {
			double val = data[_fSteps * _maxR * (_height - y - 1) + _fSteps * (_maxR - 1) + x];
			gdImageSetPixel(imSide, x, y, colors[std::clamp(int(val * 256 / _maxValue), 0, 255)]);
		}
	}

	std::vector<int> row(_fSteps);

	for (int f = 0; f < _fSteps; f++) {
		double val = 0;
		for (int i = 1; i <= 5; i++)
			val += data[_fSteps * _maxR * (_height - i) + _fSteps * (_maxR - 1) + f];
		val /= 5;
		row[f] = std::clamp(int(val * 256 / _maxValue), 0, 255);
	}

	_sideFrames.push_back(row);

	if (sideGif != nullptr) {
		gdImageGifAnimAdd(imSide, sideGif, 0, 0, 0, 1, gdDisposalNone, imPrev);
		gdFree(imPrev);
	}

	FILE* pngout;
	errno_t err = fopen_s(&pngout, filename.c_str(), "wb");
	if (err) {
		std::cout << "error opening png: " << err << std::endl;
		return;
	}
	gdImagePng(imSide, pngout);
	fclose(pngout);
}

void PolarPNG::saveGif()
{
	if (sideGif != nullptr) {
		gdImageGifAnimEnd(sideGif);
		fclose(sideGif);
	}

	if (circleGif != nullptr) {
		gdImageGifAnimEnd(circleGif);
		fclose(circleGif);
	}
}

void PolarPNG::saveOverview(std::string filename) {
	int height = _flatFrames.size();

	gdImagePtr img = gdImageCreate(_fSteps, height);
	int colorsLocal[256];
	for (int i = 0; i < 256; i++) {
		colorsLocal[i] = gdImageColorAllocate(img, i, i, i);
	}

	for (int y = 0; y < height; y++) {
		auto row = _flatFrames[height - y - 1];
		for (int x = 0; x < _fSteps; x++) {
			gdImageSetPixel(img, x, y, colorsLocal[row[x]]);
		}
	}

	FILE* pngout;
	errno_t err = fopen_s(&pngout, filename.c_str(), "wb");
	if (err) {
		std::cout << "error opening png: " << err << std::endl;
		gdFree(img);
		return;
	}
	gdImagePng(img, pngout);
	fclose(pngout);
	gdFree(img);
}
void PolarPNG::saveSideOverview(std::string filename) {
	int height = _sideFrames.size();
	if (height == 0)
		return;

	gdImagePtr img = gdImageCreate(_fSteps, height);
	int colorsLocal[256];
	for (int i = 0; i < 256; i++) {
		colorsLocal[i] = gdImageColorAllocate(img, i, i, i);
	}

	for (int y = 0; y < height; y++) {
		auto row = _sideFrames[height - y - 1];
		for (int x = 0; x < _fSteps; x++) {
			gdImageSetPixel(img, x, y, colorsLocal[row[x]]);
		}
	}

	FILE* pngout;
	errno_t err = fopen_s(&pngout, filename.c_str(), "wb");
	if (err) {
		std::cout << "error opening png: " << err << std::endl;
		gdFree(img);
		return;
	}
	gdImagePng(img, pngout);
	fclose(pngout);
	gdFree(img);
}

PolarPNG::~PolarPNG()
{
	gdImageDestroy(im);
	gdImageDestroy(imSide);
}
