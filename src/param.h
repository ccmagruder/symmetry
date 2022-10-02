#pragma once

#include <cstdint>
#include <iostream>

#include "INIReader.h"

class param
{
public:
    param(char* image_name);
//    ~param();
    double lambda;
    double alpha;
    double beta;
    double gamma;
    double omega;
    double n;
    double scale;
    uint64_t n_iter;
    int resx;
    int resy;
private:
    double getDouble(char* section, char* value);
    uint64_t getInteger(char* section, char* value);
    INIReader* reader;
};
