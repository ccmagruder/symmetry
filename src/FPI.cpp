// Copyright 2022 Caleb Magruder

#include <fstream>

#include "Image.hpp"
#include "FPI.h"
#include "PBar.h"

FPI::FPI(Param p, const std::string& label)
        : Image(p.resy, p.resx),
          _param(p),
          _hash(Param::hash(p)),
          _z(0, -0.12),
          _label(label) {
    // Initialize 1e3 transient beginning
    for (int i = 0; i < 1e3; i++) {
        _z = F(_z);
    }
}

std::complex<double> FPI::F(std::complex<double> z) {
    // std::cerr << real(_z) << "," << imag(_z) << std::endl;
    if (std::isnan(real(_z))) exit(1);
    std::complex<double> znm1 = z;
    for (int i = 1; i < _param.n - 1; i++) {
        znm1 *= z;
    }

    std::complex<double> znew = (
            _param.lambda
            + _param.alpha * norm(z)
            //+ _param.beta * real(pow(z, _param.n))
            + _param.beta * real(z*znm1)
            + _param.omega * std::complex<double>(0, 1)
            + _param.delta * std::cos(arg(z) * _param.n * _param.p) * abs(z)
        ) * z                                           // NOLINT
        //+ _param.gamma * pow(conj(z), _param.n - 1);
        + _param.gamma * conj(znm1);

    if (abs(znew) > 8) {
        std::cerr << "Warning: abs(z)=" << abs(znew) << " renormalized to 1\n";
        znew = znew / abs(znew) / static_cast<double>(3);
    }

    return znew;
}

void FPI::run_fpi(uint64_t niter) {
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

// Write to 16-bit *.pgm
void FPI::write(const std::string& filename) const {
    // Rescale 64-bit image to rescaled 16-bit
    Image<uint16_t, 1> im(_rows, _cols);
    uint64_t max = this->max();
    for (int r = 0; r < _rows; r++) {
        for (int c = 0; c < _cols; c++) {
            im[r][c] = static_cast<uint16_t>(static_cast<double>((*this)[r][c])
                    / max * __UINT16_MAX__);
        }
    }
    // Balance colors by taking logarithm
    im.logRescale();
    im.write(filename);
}

std::string FPI::getHashFilename(const Param& p) {
    std::stringstream filename;
    size_t hash = Param::hash(p);
    filename << std::hex << hash << ".dat";
    return filename.str();
}

FPI FPI::load(const Param& p) {
    FPI fpi(p);

    std::string filename(FPI::getHashFilename(p));
    std::ifstream file(filename);
    // If the file does exist, create new object
    if (!file.good()) {
        return fpi;
    }

    size_t rows, cols;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    if (fpi.rows() != rows || fpi.cols() != cols) {
        throw std::runtime_error("FPI::load(Param) invalid rows or cols");
    }
    file.read(reinterpret_cast<char*>(fpi._data),
        rows * cols * fpi.colors() * sizeof(uint64_t));

    double real, imag;
    file.read(reinterpret_cast<char*>(&real), sizeof(double));
    file.read(reinterpret_cast<char*>(&imag), sizeof(double));
    fpi._z = {real, imag};
    return fpi;
}

void FPI::save() {
    std::stringstream ss;
    ss << std::hex << _hash << ".dat";

    std::ofstream file(ss.str(),
        std::ios::trunc | std::ios_base::binary);
    file.write(reinterpret_cast<char*>(&_rows), sizeof(_rows));
    file.write(reinterpret_cast<char*>(&_cols), sizeof(_cols));
    file.write(reinterpret_cast<char*>(_data),
        _rows * _cols * colors() * sizeof(*_data));
    double real = _z.real(), imag = _z.imag();
    file.write(reinterpret_cast<char*>(&real), sizeof(double));
    file.write(reinterpret_cast<char*>(&imag), sizeof(double));
    file.close();
}
