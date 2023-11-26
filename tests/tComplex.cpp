// Copyright 2023 Caleb Magruder

#include <complex>

#include "gtest/gtest.h"

#include "Complex.hpp"

TEST(tComplex, Ctor) {
    Complex<std::complex<double>> vec(2);
    vec[0] = {3, 4};
    EXPECT_EQ(vec[0], std::complex<double>(3, 4));
}

TEST(tComplex, Multiplication) {
    Complex<std::complex<double>> vec1(1), vec2(1);
    vec1[0] = {0, 1};
    vec2[0] = {1, -1};
    vec2 *= vec1;
    EXPECT_EQ(vec2[0], std::complex<double>(1, 1));
}

TEST(tComplex, Assignment) {
    Complex<std::complex<double>> vec1(2);
    vec1[0] = {1, -1};
    vec1[1] = {0, 1};
    Complex<std::complex<double>> vec2(2);
    vec2 = vec1;
    EXPECT_EQ(vec2[1], std::complex<double>(0, 1));
}
