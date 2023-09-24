// Copyright 2022 Caleb Magruder

#pragma once

#include <string>
#include <vector>

#include "Color.h"
#include "Image.hpp"

class ColorMap {
 public:
    explicit ColorMap(std::string fileName);

    void write(std::string fileName);

    Image<uint16_t, 3> operator()(const Image<uint16_t, 1>& pgm) const;
    Color operator()(const uint16_t alpha) const;

    std::vector<uint16_t> _alpha;
    std::vector<Color> _map;
    size_t _numColors;
};
