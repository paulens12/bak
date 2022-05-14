#pragma once
#include <string>
#include "CartesianToPolar.h"

class PolarPNG
{
private:
	int _maxR, _fSteps;
	CartesianToPolar _ctp;
	double _maxValue;

	double* fillBlanks(double* data);
public:
	PolarPNG(int maxR, int factor, int fSteps, double maxValue);
	void savePNG(double* data, std::string filename);
	void saveFlat(double* data, std::string filename);
};