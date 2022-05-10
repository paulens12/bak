#pragma once
#include <cstddef>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <functional>

class CartesianToPolar
{
	struct TLUTInfo
	{
		bool valid;
		double rOffset;
		double fOffset;
		int fmin;
		int fmax;
		int rmin;
		int rmax;
	};

private:
	TLUTInfo* _lookUpTable;
	int _height, _width;
	int _maxR, _factor;

	bool polarToCartesian(double* r, double* f, double x, double y)
	{
		*r = sqrt(x * x + y * y);
		if (*r >= _maxR - 1) return false;
		*f = atan2(y, x);
		return true;
	}

	double getF(int f, int r) {
		if (r < _maxR / 4)
			return f / 4;
		else if (r < _maxR / 2)
			return f / 2;
		else
			return f;
	}


public:
	CartesianToPolar(int maxR, int factor, int fSteps)
	{
		_maxR = maxR;
		_factor = factor;
		_height = _width = 2 * maxR * factor;

		double r, fRad, f;
		double x, y;

		_lookUpTable = (TLUTInfo*)std::malloc(_width * _height * sizeof(TLUTInfo));
		for (int i = 0; i < _height; ++i)
		{
			y = maxR - (double)i / factor;
			for (int j = 0; j < _width; ++j)
			{
				x = (double)j / (double)factor - maxR;
				_lookUpTable[i * _width + j].valid = polarToCartesian(&r, &fRad, x, y);
				if (!_lookUpTable[i * _width + j].valid)
					continue;
				if (j == 113 && i == 91)
					std::cout << "here!" << std::endl;

				double meh;
				int rmin = _lookUpTable[i * _width + j].rmin = r;
				int rmax = _lookUpTable[i * _width + j].rmax = r + 1;
				_lookUpTable[i * _width + j].rOffset = std::modf(r, &meh);

				if (fRad < 0)
					fRad += 2 * M_PI;

				f = fRad * fSteps / (2 * M_PI);

				_lookUpTable[i * _width + j].fmin = getF(f, rmin);
				if (f < 0)
					std::cout << "ERROR";
				_lookUpTable[i * _width + j].fOffset = f - _lookUpTable[i * _width + j].fmin;

				if (r < maxR / 4)
					_lookUpTable[i * _width + j].fmax = (_lookUpTable[i * _width + j].fmin + 1) % (fSteps / 4);
				else if (r < maxR / 2)
					_lookUpTable[i * _width + j].fmax = (_lookUpTable[i * _width + j].fmin + 1) % (fSteps / 2);
				else
					_lookUpTable[i * _width + j].fmax = (_lookUpTable[i * _width + j].fmin + 1) % fSteps;

				// fmax, fmin should be multiplied only when using with rmax -- getValue()
				if (rmax == maxR / 4 || rmax == maxR / 2)
					_lookUpTable[i * _width + j].fmax *= 2;

				if (f < 0 || _lookUpTable[i * _width + j].fmax >= fSteps)
					std::cout << "ERROR";
			}
		}
	}

	int getWidth()
	{
		return _width;
	}

	double getValue(int x, int y, std::function<double(int, int)> fn)
	{
		TLUTInfo info = _lookUpTable[y * _width + x];
		if (!info.valid) return 0.0;

		return fn(info.rmin, info.fmin) * (1.0 - info.rOffset) * (1.0 - info.fOffset)
			+ fn(info.rmax, info.fmin) * info.rOffset * (1.0 - info.fOffset)
			+ fn(info.rmin, info.fmax) * (1.0 - info.rOffset) * info.fOffset
			+ fn(info.rmax, info.fmax) * info.rOffset * info.fOffset;
	}
};

