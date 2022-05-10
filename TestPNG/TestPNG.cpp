// PolarPNG.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"
#include <argh.h>
#include <string>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>

int main(int argc, char** argv)
{
    argh::parser cmdl(argc, argv);

    std::string inFile = "u.dat";
    std::string outFile = "u";
    int R = 60;
    int F = 200;
    int mul = 3;
    double maxValue = 4.2;
    int minTimeStep = 0;
    int maxTimeStep = 960;
    int timeStepDelta = 20;

    double coef = 255.0 / maxValue;

    std::ifstream in;
    in.open(inFile.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if (!in.is_open())
        return 4;

    int size = in.tellg();
    if (size < (maxTimeStep + 1) * 11 * R * F / 16 * sizeof(double))
        return 3;

    auto memblock = new char[size];
    in.seekg(0, std::ios::beg);
    in.read(memblock, size);
    in.close();

    double* input = (double*)memblock;

    char* outBuffer = new char[R * F];

    for (int timeStep = minTimeStep; timeStep <= maxTimeStep; timeStep += timeStepDelta)
    {
        int timeOffset = timeStep * 11 * R * F / 16;

        for (int f = 0; f < F; ++f)
        {
            for (int r = 0; r < R / 4; ++r)
            {
                outBuffer[F * r + f] = (char)std::min(std::max((int)std::lround(input[timeOffset + r * F / 4 + f / 4] * coef), 0), 255);
            }
            for (int r = R / 4; r < R / 2; ++r)
            {
                outBuffer[F * r + f] = (char)std::min(std::max((int)std::lround(input[timeOffset + R * F / 16 + (r - R / 4) * F / 2 + f / 2] * coef), 0), 255);
            }
            for (int r = F / 2; r < R; ++r)
            {
                outBuffer[F * r + f] = (char)std::min(std::max((int)std::lround(input[timeOffset + 3 * R * F / 16 + (r - R / 2) * F + f] * coef), 0), 255);
            }
        }
        std::cout << timeStep << std::endl;
        stbi_write_png((outFile + "_" + std::to_string(timeStep) + ".png").c_str(), F, R, 1, outBuffer, F);
    }
    
}
