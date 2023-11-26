// Copyright 2022 Caleb Magruder

#pragma once

#include <complex>
#include <string>

#include "Image.hpp"
#include "Param.h"

class FPI : public Image<uint64_t, 1>{
 public:
    explicit FPI(Param p, const std::string& label = " FPI ")
        : Image(p.resy, p.resx),
          _param(p),
          _z(0, -0.12),
          _label(label) {
        // Initialize 1e3 transient beginning
        for (int i = 0; i < 1e3; i++) {
            _z = F(_z);
        }
    }

    ~FPI() {}

    std::complex<double> F(std::complex<double> z) {
        if (std::isnan(real(_z))) exit(1);

        // Compute z^{n-1} (znm1 equals 'z to the n minus 1')
        this->_znm1 = z;
        for (int i = 1; i < _param.n - 1; i++) {
            this->_znm1 *= z;
        }

        this->_znew = (
                _param.lambda
                + _param.alpha * abs(z) * abs(z)
                //+ _param.beta * real(pow(z, _param.n))
                + _param.beta * real(z*this->_znm1)
                + _param.omega * std::complex<double>(0, 1)
                + _param.delta * cos(arg(z) * _param.n * _param.p) * abs(z)
            ) * z                                           // NOLINT
            //+ _param.gamma * pow(conj(z), _param.n - 1);
            + _param.gamma * conj(this->_znm1);

        if (abs(this->_znew) > 8) {
            std::cerr << "Warning: abs(z)=" << abs(this->_znew)
                << " renormalized to 1\n";
            this->_znew /= abs(this->_znew) / static_cast<double>(3);
        }

        return this->_znew;
    }

    void run_fpi(uint64_t niter) {
        PBar i(niter, 8, this->_label);
        for (i = 0; i < niter; i++) {
            _z = F(_z);

            // For some figures cycles appear,add noise every 1000 iterations
            if (static_cast<int>(i) % 1000 == 0) {
                _z.real(_z.real() * 0.99 - 1e-2 * sgn(_z.real()));
                _z.imag(_z.imag() * 0.99 - 1e-2 * sgn(_z.imag()));
                for (int j = 0; j < 1e1; j++)
                    _z = F(_z);
            }

            // Imaginary axis sometimes closed under FPI
            // Kicks iterate out of loop, re-initializes
            if (std::abs(real(_z)) < 1e-15) {
                _z.real(0.001);
                for (int j = 0; j < 1e1; j++)
                    _z = F(_z);
            }

            if (std::abs(imag(_z)) < 1e-15) {
                _z.imag(0.001);
                for (int j = 0; j < 1e1; j++)
                    _z = F(_z);
            }

            int size = std::sqrt(_rows*_cols);
            int c = floor(_param.scale*size/2*real(_z) + _cols/2);
            int r = floor(_param.scale*size/2*imag(_z) + _rows/2);
            if (r >=0 && r < _rows && c >=0 && c < _cols) {
                ++(*this)[r][c];
            }
        }
    }

    void run_fpi() { run_fpi(_param.n_iter); }

    // Write to 16-bit *.pgm
    void write(const std::string& filename) const {
        // Rescale 64-bit image to rescaled 16-bit
        Image<uint16_t, 1> im(_rows, _cols);
        uint64_t max = this->max();
        for (int r = 0; r < _rows; r++) {
            for (int c = 0; c < _cols; c++) {
                im[r][c] = static_cast<uint16_t>(
                    static_cast<double>((*this)[r][c]) / max * __UINT16_MAX__);
            }
        }
        // Balance colors by taking logarithm
        im.logRescale();
        im.write(filename);
    }

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
