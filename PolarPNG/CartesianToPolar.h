#pragma once

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
	int _maxR, _factor, _fSteps;

	double getPoint(double data[], int r, int f);

	bool getPolarCoords(double* r, double* f, double x, double y);

public:
	CartesianToPolar(int maxR, int factor, int fSteps);

	int getWidth();

	double getValue(double data[], int x, int y);
};

