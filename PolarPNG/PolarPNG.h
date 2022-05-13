#pragma once
#include <string>
#include "CartesianToPolar.h"

class PolarPNG
{
private:
	int _maxR, _fSteps;
	CartesianToPolar _ctp;
	double _maxValue;
	double* _buffer;
public:
	PolarPNG(int maxR, int factor, int fSteps, double maxValue);
	void savePNG(double* data, std::string filename);
	~PolarPNG();
};