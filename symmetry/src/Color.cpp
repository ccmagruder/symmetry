// Copyright 2022 Caleb Magruder

#include <ostream>

#include "Color.h"

std::ostream& operator<<(std::ostream& os, const Color& c) {
    os << "(" << c[0] << ", " << c[1] << ", " << c[2] << ")";
    return os;
}
