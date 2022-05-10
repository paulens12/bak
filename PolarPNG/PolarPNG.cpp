// PolarPNG.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"
#include <argh.h>
#include <string>
#include <vector>
//#include <libInterpolate/Interpolate.hpp>
#define _USE_MATH_DEFINES
#include <math.h>
#include "CartesianToPolar.h"

int getPos(int r, int f, int R, int F) {
    if (r < R / 4)
        return r * F / 4 + f;
    else if (r < R / 2)
        return R * F / 16 + (r - R / 4) * F / 2 + f;
    else
        return 3 * R * F / 16 + (r - R / 2) * F + f;
}

int main(int argc, char** argv)
{
    argh::parser cmdl(argc, argv);

    //--in=G:\programming\cuda\bak\Polar2DSingle\u.dat --out=u --radius=40 --circ=200 --mult=3 --max=4.2 --td=80 --tmin=0 --tmax=960
    std::string inFile = cmdl("--in").str();
    std::string outFile = cmdl("--out").str();
    int R = std::stoi(cmdl("--radius").str());
    int F = std::stoi(cmdl("--circ").str());
    int mul = std::stoi(cmdl("--mult").str());
    double maxValue = R; //std::stod(cmdl("--max").str());
    //int timeStepDelta = std::stoi(cmdl("--td").str());
    //int minTimeStep = std::stoi(cmdl("--tmin").str());
    //int maxTimeStep = std::stoi(cmdl("--tmax").str());
    int minTimeStep = 0;
    int maxTimeStep = 0;
    int timeStepDelta = 80;
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

    int idx = 0;
    for (int r = 0; r < R; ++r) {
        int fMax = r < R / 4 ? F / 4 : r < R / 2 ? F / 2 : F;
        std::cout << "r: " + std::to_string(r) + ", fMax: " + std::to_string(fMax) << std::endl;
        for (int f = 0; f < fMax / 4; ++f)
            input[idx++] = (double)r;
        for (int f = fMax / 4; f < fMax / 2; ++f)
            input[idx++] = (double)R - r;
        for (int f = fMax / 2; f < 3 * fMax / 4; ++f)
            input[idx++] = (double)r;
        for (int f = 3 * fMax / 4; f < fMax; ++f)
            input[idx++] = (double)R - r;
    }

    auto start = clock();

    CartesianToPolar ctp(R, mul, F);

    double elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;
    printf("constructor done - time elapsed: %f\n", elapsed);

    int imgSize = ctp.getWidth();
    char* outBuffer = new char[imgSize * imgSize];

    for (int timeStep = minTimeStep; timeStep <= maxTimeStep; timeStep += timeStepDelta)
    {
        int timeOffset = timeStep * 11 * R * F / 16;

        for (int h = 0; h < imgSize; ++h)
        {
            for (int w = 0; w < imgSize; ++w)
            {
                int rawValue = (int)std::lround(ctp.getValue(w, h, [=](int r, int f) {
                    if (r < R / 4)
                        return input[timeOffset + r * F / 4 + f];
                    else if (r < R / 2)
                        return input[timeOffset + R * F / 16 + (r - R / 4) * F / 2 + f];
                    else
                        return input[timeOffset + 3 * R * F / 16 + (r - R / 2) * F + f];
                    }) * coef);
                if (w < imgSize / 2 && h < imgSize / 2) {
                    int x = abs(w - imgSize / 2);
                    int y = abs(h - imgSize / 2);
                    int r = sqrt(x * x + y * y);
                    if (r < 35 && rawValue < 150) {
                        std::cout << "wat!";
                    }
                }
                    
                outBuffer[imgSize * h + w] = std::min(std::max(rawValue, 0), 255);
            }
        }
        elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;
        printf("loop done - time elapsed: %f\n", elapsed);

        stbi_write_png((outFile + "_" + std::to_string(timeStep) + ".png").c_str(), imgSize, imgSize, 1, outBuffer, imgSize);
    }


    elapsed = (clock() - start) / (double)CLOCKS_PER_SEC;
    printf("all done - time elapsed: %f\n", elapsed);

//    std::vector<double> x, y, z;
//    x.reserve(11 * R * F / 16);
//    y.reserve(11 * R * F / 16);
//    z.reserve(11 * R * F / 16);
//    for (int r = 0; r < R; ++r)
//    {
//        for (int f = 0; f < F; ++f)
//        {
//            double absf = (double)f * 2 * M_PI / (double)F;
//            x.push_back(r * cos(absf) * mul);
//            y.push_back(r * sin(absf) * mul);
//            if (r < R / 4)
//                z.push_back(input[timeOffset + r * F / 4 + f]);
//            else if (r < R / 2)
//                z.push_back(input[timeOffset + R * F / 16 + (r - R / 4) * F / 2 + f]);
//            else
//                z.push_back(input[timeOffset + 3 * R * F / 16 + (r - R / 2) * F + f]);
//        }
//    }
//
//    _2D::ThinPlateSplineInterpolator<double> interp;
//    interp.setData(x, y, z);
//
//    int diameter = 2 * R * mul;
//    char* outBuffer = (char*)malloc(diameter * diameter);
//    perror("malloc error");
//    if (outBuffer == NULL)
//        return 4;
//
//#pragma warning( push )
//#pragma warning( disable : 6386 )
//
//    for (int i = 0; i < diameter; i++)
//    {
//        for (int j = 0; j < diameter; j++)
//        {
//            int x = j - diameter / 2;
//            int y = diameter / 2 - i;
//            double r = sqrt(x * x + y * y);
//            if (r > R)
//                outBuffer[i * diameter + j] = 0;
//            else
//                outBuffer[i * diameter + j] = std::min(
//                    std::max(
//                        (int)std::lround(
//                            interp(x, y) * coef),
//                        0),
//                    255);
//        }
//    }
//
//#pragma warning( pop )
//
//    stbi_write_png(outFile.c_str(), diameter, diameter, 1, outBuffer, diameter);
}
