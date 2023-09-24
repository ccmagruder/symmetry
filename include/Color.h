// Copyright 2022 Caleb Magruder

#pragma once

#include <ostream>
#include <vector>

class Color : private std::vector<uint16_t> {
 public:
    explicit Color(uint16_t r = 0, uint16_t g = 0, uint16_t b = 0)
        : std::vector<uint16_t>{r, g, b} {}

    using std::vector<uint16_t>::data;
    using std::vector<uint16_t>::operator[];

    bool operator==(const Color& rhs) const {
        return *this == dynamic_cast<const std::vector<uint16_t>&>(rhs);
    }
};

std::ostream& operator<<(std::ostream& os, const Color& c);
