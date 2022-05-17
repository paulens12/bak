#pragma once
#include <string>
#include <vector>
#include "CartesianToPolar.h"
#include "gd.h"

class PolarPNG
{
private:
	gdImagePtr im;
	gdImagePtr imSide;
	int _maxR, _fSteps;
	CartesianToPolar _ctp;
	double _maxValue;
	int _height;
	int colors[256]{};
	int colorsSide[256]{};
	FILE* circleGif = nullptr;
	FILE* sideGif = nullptr;
	std::vector<std::vector<int>> _sideFrames;
	std::vector<std::vector<int>> _flatFrames;

	double* fillBlanks(double* data);
public:
	PolarPNG(int maxR, int factor, int fSteps, double maxValue, int height = 1, std::string gifFilename = "", std::string gifSideFilename = "");
	void savePNG(double* data, std::string filename);
	void saveSide(double* data, std::string filename);
	void saveGif();
	void saveOverview(std::string filename);
	void saveSideOverview(std::string filename);
	~PolarPNG();
};