#pragma once

#include <complex>
#include <string>

#include "image.h"
#include "param.h"

class fpi : public image
{
private:
    param _param;
public:
    fpi(param p);
    ~fpi(){}
    std::complex<double> F(std::complex<double> z);
    void run_fpi();
};