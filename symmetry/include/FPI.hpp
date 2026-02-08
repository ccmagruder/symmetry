// Copyright 2022 Caleb Magruder

#pragma once

#include <complex>
#include <string>

#include "Complex.hpp"
#include "Image.hpp"
#include "Param.h"
#include "PBar.h"

template <typename T = double>
class FPI : public Image<uint64_t, 1>{
    using Type = typename complex_traits<T>::value_type;

 public:
    explicit FPI(Param p, const std::string& label = " FPI ")
        : Image(p.resy, p.resx),
          _param(p),
          _z(0, -0.12),
          _label(label) {
        this->_alpha = static_cast<Type>(_param.alpha);
        this->_beta = static_cast<Type>(_param.beta);
        this->_delta = static_cast<Type>(_param.delta);
        this->_gamma = static_cast<Type>(_param.gamma);
        this->_n = static_cast<Type>(_param.n);
        this->_p = static_cast<Type>(_param.p);
        this->_lambda = std::complex<Type>(_param.lambda, 0);
        this->_omega = std::complex<Type>(0, _param.omega);
        // Initialize 1e3 transient beginning
        for (int i = 0; i < 1e3; i++) {
            _z = F(_z);
        }
    }

    ~FPI() {}

    std::complex<Type> F(std::complex<Type> z) {
        if (std::isnan(real(_z))) exit(1);

        // Compute z^{n-1} (znm1 equals 'z to the n minus 1')
        this->_znm1 = z;
        for (int i = 1; i < std::round(this->_n) - 1; i++) {
            this->_znm1 *= z;
        }

        this->_znew = (
                this->_lambda
                + this->_alpha * abs(z) * abs(z)
                //+ _param.beta * real(pow(z, _param.n))
                + this->_beta * real(z*this->_znm1)
                + this->_omega
                + std::complex<Type>(
                    this->_delta * cos(arg(z) * this->_n * this->_p) * abs(z),
                    0
                  )
            ) * z                                           // NOLINT
            //+ _param.gamma * pow(conj(z), _param.n - 1);
            + this->_gamma * conj(this->_znm1);

        if (abs(this->_znew) > 8) {
            std::cerr << "Warning: abs(z)=" << abs(this->_znew)
                << " renormalized to 1\n";
            this->_znew /= abs(this->_znew) / 3.0;
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

            double size = std::sqrt(_rows*_cols);
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

    template <typename S>
    static int sgn(S val) {
        return (S(0) < val) - (val < S(0));
    }

 private:
    const Param _param;

    std::complex<Type> _z;
    std::complex<Type> _znm1;
    std::complex<Type> _znew;
    std::complex<Type> _lambda;
    std::complex<Type> _omega;
    Type _alpha;
    Type _beta;
    Type _delta;
    Type _gamma;
    Type _n;
    Type _p;

    const std::string _label;
};
