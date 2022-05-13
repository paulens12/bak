#include "CartesianToPolar.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

inline bool CartesianToPolar::getPolarCoords(double* r, double* f, double x, double y)
{
	*r = sqrt(x * x + y * y);
	if (*r >= _maxR - 1) return false;
	*f = atan2(y, x);
	return true;
}

CartesianToPolar::CartesianToPolar(int maxR, int factor, int fSteps)
{
	_maxR = maxR;
	_factor = factor;
	_height = _width = 2 * maxR * factor;
	_fSteps = fSteps;

	double r, fRad, f;
	double x, y;

	_lookUpTable = new TLUTInfo[_width * _height];
	for (int i = 0; i < _height; ++i)
	{
		y = maxR - (double)i / factor;
		for (int j = 0; j < _width; ++j)
		{
			x = (double)j / (double)factor - maxR;
			_lookUpTable[i * _width + j].valid = getPolarCoords(&r, &fRad, x, y);
			if (!_lookUpTable[i * _width + j].valid)
				continue;

			int rmin = _lookUpTable[i * _width + j].rmin = r;
			int rmax = _lookUpTable[i * _width + j].rmax = r + 1;
			_lookUpTable[i * _width + j].rOffset = r - rmin;

			if (fRad < 0)
				fRad += 2 * M_PI;

			f = fRad * fSteps / (2 * M_PI);

			int fmin = _lookUpTable[i * _width + j].fmin = f;
			_lookUpTable[i * _width + j].fOffset = f - fmin;
			_lookUpTable[i * _width + j].fmax = (fmin + 1) % fSteps;


			if (f < 0 || _lookUpTable[i * _width + j].fmax >= fSteps)
				std::cout << "ERROR";
		}
	}
}

int CartesianToPolar::getWidth()
{
	return _width;
}

inline double CartesianToPolar::getPoint(double data[], int r, int f) {
	return data[r * _fSteps + f];
}

double CartesianToPolar::getValue(double data[], int x, int y)
{
	TLUTInfo info = _lookUpTable[y * _width + x];
	if (!info.valid) return 0.0;

	return getPoint(data, info.rmin, info.fmin) * (1.0 - info.rOffset) * (1.0 - info.fOffset)
		+ getPoint(data, info.rmax, info.fmin) * info.rOffset * (1.0 - info.fOffset)
		+ getPoint(data, info.rmin, info.fmax) * (1.0 - info.rOffset) * info.fOffset
		+ getPoint(data, info.rmax, info.fmax) * info.rOffset * info.fOffset;
}
