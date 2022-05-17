// PolarPNGApp.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <format>
#include <argh.h>
#include "../PolarPNG/PolarPNG.h"

struct pngOut {
    PolarPNG* pnglib;
    int z;
    ~pngOut() { delete pnglib; }
};

int main(int argc, char** argv)
{
    argh::parser cmdl(argc, argv);
    std::string inFile = cmdl("--in", "u.dat").str();
    std::string outFile = cmdl("--out", "u").str();
    double maxValue = std::stod(cmdl("--max", 4).str());
    int R = 40;
    int F = 224;
    int Z = 64;

    size_t frameSize = R * F * Z;
    std::ifstream in;
    in.open(inFile.c_str(), std::ios::in | std::ios::binary);
    if (!in.is_open())
        return 1;

    double* buffer = new double[frameSize];

    in.seekg(0, in.end);
    auto length = in.tellg();
    in.seekg(0, in.beg);

    pngOut outputs[] = {
        {new PolarPNG(R, 3, F, maxValue, 1, std::format("{}_Z{}.gif", outFile, 0)), 0},
        {new PolarPNG(R, 3, F, maxValue, 1, std::format("{}_Z{}.gif", outFile, Z / 2)), Z / 2},
        {new PolarPNG(R, 3, F, maxValue, 1, std::format("{}_Z{}.gif", outFile, (3 * Z) / 4)), (3 * Z) / 4},
        {new PolarPNG(R, 3, F, maxValue, 1, std::format("{}_Z{}.gif", outFile, (7 * Z) / 8)), (7 * Z) / 8},
        {new PolarPNG(R, 3, F, maxValue, 1, std::format("{}_Z{}.gif", outFile, Z - 1)), Z - 1},
    };
    PolarPNG* sidePng = new PolarPNG(R, 1, F, maxValue, Z, "", std::format("{}_side.gif", outFile));
    int step = 0;
    while (!in.eof()) {
        in.read((char*)buffer, frameSize * sizeof(double));
        std::streamsize dataSize = in.gcount();
        if (dataSize == 0)
            break;
        if (dataSize != frameSize * sizeof(double)) {
            std::cout << "Expected " << frameSize * sizeof(double) << ", got " << dataSize << " bytes." << std::endl;
        }

        for (pngOut& out : outputs) {
            out.pnglib->savePNG(buffer + R * F * out.z, std::format("{}_Z{}_step{}.png", outFile, out.z, step));
        }
        sidePng->saveSide(buffer, std::format("{}_side_step{}.png", outFile, step));
        step++;
    }

    for (pngOut& out : outputs) {
        out.pnglib->saveGif();
        out.pnglib->saveOverview(std::format("{}_overview_Z{}.png", outFile, out.z));
    }
    sidePng->saveGif();
    sidePng->saveSideOverview(std::format("{}_overview_top.png", outFile));

    delete sidePng;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
