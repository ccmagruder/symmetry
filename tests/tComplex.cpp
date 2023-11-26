// Copyright 2023 Caleb Magruder

#include <complex>

#include "gtest/gtest.h"

#include "Complex.hpp"

TEST(tComplex, Ctor) {
    Complex<std::complex<double>> vec(2);
    vec[0] = {3, 4};
    EXPECT_EQ(vec[0], std::complex<double>(3, 4));
}

TEST(tComplex, Addition) {
    Complex<std::complex<double>> x(1), y(1), z(1);
    x[0] = {0, 1};
    y[0] = {1, -1};
    z = x + y;
    EXPECT_EQ(z[0], std::complex<double>(1, 0));
}

TEST(tComplex, Multiplication) {
    Complex<std::complex<double>> x(1), y(1), z(1);
    x[0] = {0, 1};
    y[0] = {1, -1};
    z = x * y;
    EXPECT_EQ(z[0], std::complex<double>(1, 1));
}

TEST(tComplex, Assignment) {
    Complex<std::complex<double>> vec1(2);
    vec1[0] = {1, -1};
    vec1[1] = {0, 1};
    Complex<std::complex<double>> vec2(2);
    vec2 = vec1;
    EXPECT_EQ(vec2[1], std::complex<double>(0, 1));
}

TEST(tComplex, ScalarMultiplication) {
    Complex<std::complex<double>> vec1(1), vec2(1);
    vec1[0] = {-1, 1};
    std::complex<double> a(0, 1);
    vec2 = a * vec1;
    EXPECT_EQ(vec2[0], std::complex<double>(-1, -1));
}

TEST(tComplex, Abs) {
    Complex<std::complex<double>> vec1(2), vec2(2);
    vec1[0] = {3, -4};
    vec1[1] = {0, -1};
    vec2 = abs(vec1);
    EXPECT_EQ(vec2[0], std::complex<double>(5, 0));
    EXPECT_EQ(vec2[1], std::complex<double>(1, 0));
}

TEST(tComplex, Arg) {
    Complex<std::complex<double>> vec1(2), vec2(2);
    vec1[0] = {-1, 0};
    vec1[1] = {0, -1};
    vec2 = arg(vec1);
    EXPECT_EQ(vec2[0], std::complex<double>(std::numbers::pi, 0));
    EXPECT_EQ(vec2[1], std::complex<double>(-std::numbers::pi/2, 0));
}

TEST(tComplex, Conj) {
    Complex<std::complex<double>> vec1(2), vec2(2);
    vec1[0] = {-1, 0};
    vec1[1] = {1, -2};
    vec2 = conj(vec1);
    EXPECT_EQ(vec2[0], std::complex<double>(-1, 0));
    EXPECT_EQ(vec2[1], std::complex<double>(1, 2));
}

TEST(tComplex, Cos) {
    Complex<std::complex<double>> vec1(2), vec2(2);
    vec1[0] = {std::numbers::pi, 4};
    vec1[1] = {std::numbers::pi/2, -2};
    vec2 = cos(vec1);
    EXPECT_NEAR(vec2[0].real(), -1, 1e-4);
    EXPECT_NEAR(vec2[1].real(), 0, 1e-4);
}