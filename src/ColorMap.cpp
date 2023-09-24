// Copyright 2022 Caleb Magruder

#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#include "Image.hpp"
#include "ColorMap.h"

ColorMap::ColorMap(std::string fileName) {
        std::ifstream file(fileName);
        if (file.fail()) {
            throw std::runtime_error("Missing " + fileName);
        }

        ptrdiff_t i = 0;
        char comma;
        double percent;
        while (!file.eof()) {
            _map.emplace_back(0, 0, 0);

            file >> percent >> comma >>
                _map[i][0] >> comma >>
                _map[i][1] >> comma >>
                _map[i][2];

            // Rescale percent to alpha in [0, 65535]
            _alpha.push_back(static_cast<uint16_t>(percent/100*__UINT16_MAX__));

            // Rescale RGB[0,255] -> RGB[0, 65535]
            // __UINT16_MAX__ / __UINT8_MAX__ == __UINT8_MAX__ + 2 = 257
            _map[i][0] = uint16_t(_map[i][0] * (__UINT8_MAX__ + 2));
            _map[i][1] = uint16_t(_map[i][1] * (__UINT8_MAX__ + 2));
            _map[i][2] = uint16_t(_map[i][2] * (__UINT8_MAX__ + 2));
            i++;
        }
        _numColors = _map.size();
}

void ColorMap::write(std::string fileName) {
    size_t dim = 256;
    Image<uint16_t, 3> im(dim, dim);
    const long double ratio = __UINT16_MAX__/dim;
    Color ctmp;
    ptrdiff_t r;
    r = 0;
    for (r=0; r < dim/2; r++) {
        for (size_t c=0; c < dim; c++) {
            ctmp = operator()(static_cast<uint16_t> (
                static_cast<long double>(c)*ratio));
            memcpy(im[r][3*c], ctmp.data(), 6);
        }
    }

    for (; r < dim; r++) {
        for (ptrdiff_t c=0; c < dim; c++) {
            uint16_t tmp = static_cast<uint16_t> (
                static_cast<long double>(c)*ratio);
            memcpy(im[r][3*c], &tmp, 2);
            memcpy(im[r][3*c+1], &tmp, 2);
            memcpy(im[r][3*c+2], &tmp, 2);
        }
    }

    im.write(fileName);
}

Image<uint16_t, 3> ColorMap::operator()(const Image<uint16_t, 1>& pgm) const {
    size_t rows = pgm.rows();
    size_t cols = pgm.cols();
    Image<uint16_t, 3> ppm(rows, cols);

    Color mycolor;
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            mycolor = operator()(pgm[r][c]);
            memcpy(ppm[r][c], mycolor.data(), 3*sizeof(uint16_t));
        }
    }
    return ppm;
}

Color ColorMap::operator()(const uint16_t alpha) const {
    ptrdiff_t i = 0;
    while (alpha > _alpha[i+1]) {
        i++;
    }
    double max = static_cast<double>(__UINT16_MAX__);
    double alphaL = max * (alpha - _alpha[i]) / (_alpha[i+1] - _alpha[i]);
    // Nonlinear Interpolation via squared values. For details, see:
    // https://sighack.com/post/averaging-rgb-colors-the-right-way
    const Color& a = _map[i];
    const Color& b = _map[i+1];
    Color c;
    for (ptrdiff_t i = 0; i < 3; i++) {
        c[i] = uint16_t(sqrt(((max-alphaL) * pow(a[i], 2)
            + alphaL * pow(b[i], 2)) / __UINT16_MAX__));
    }
    return c;
}
