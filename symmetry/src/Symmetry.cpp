// Copyright 2022 Caleb Magruder

#include <iostream>

#include "ColorMap.h"
#include "Image.hpp"
#include "FPI.hpp"

void print_help() {
    std::cerr << "symmetry run config/*.json images/*.pbm.\n"
        << "symmetry color images/*.pgm images/*.ppm\n";
    exit(1);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_help();
    }
    std::string command = std::string(argv[1]);
    if (command == "run") {
        std::string filenameConfig = std::string(argv[2]);
        std::string filenameImage = std::string(argv[3]);
        Param p(filenameConfig);
        FPI im(p, filenameConfig);
        im.run_fpi();
        im.write(filenameImage);
    } else if (command == "color") {
        std::string filenameColorMap = std::string(argv[2]);
        std::string filenamePGM = std::string(argv[3]);
        Image<uint16_t, 1> pgm(filenamePGM);
        ColorMap map(filenameColorMap);
        std::string filenamePPM(filenamePGM);
        filenamePPM[filenamePPM.size()-2] = 'p';
        Image<uint16_t, 3> ppm = map(pgm);
        ppm.write(filenamePPM);
    } else {
        print_help();
        exit(1);
    }

    return 0;
}
