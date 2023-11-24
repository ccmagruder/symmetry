// Copyright 2022 Caleb Magruder

#pragma once

#include <complex>
#include <string>

#include "Image.hpp"
#include "Param.h"

class FPI : public Image<uint64_t, 1>{
 public:
    explicit FPI(Param p, const std::string& label = " FPI ");

    ~FPI() {}

    std::complex<double> F(std::complex<double> z);

    void run_fpi(uint64_t niter);
    void run_fpi() { run_fpi(_param.n_iter); }

    void write(const std::string& filename) const;

    template <typename T>
    static int sgn(T val) {
        return (T(0) < val) - (val < T(0));
    }

 private:
    const Param _param;

    std::complex<double> _z;
    std::complex<double> _znm1;
    std::complex<double> _znew;

    const std::string _label;
};
