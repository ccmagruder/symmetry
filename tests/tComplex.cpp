// Copyright 2023 Caleb Magruder

#include <complex>

#include "gtest/gtest.h"

#include "Complex.hpp"

template <typename T>
class tComplex : public testing::Test {};

using Types = ::testing::Types <
    std::complex<float>,
    std::complex<double>
>;

TYPED_TEST_SUITE(tComplex, Types);

TYPED_TEST(tComplex, Ctor) {
    Complex<TypeParam> vec(2);
    vec[0] = {3, 4};
    EXPECT_EQ(vec[0], TypeParam(3, 4));
}

TYPED_TEST(tComplex, Addition) {
    Complex<TypeParam> x(1), y(1), z(1);
    x[0] = {0, 1};
    y[0] = {1, -1};
    z = x + y;
    EXPECT_EQ(z[0], TypeParam(1, 0));
}

TYPED_TEST(tComplex, Multiplication) {
    Complex<TypeParam> x(1), y(1), z(1);
    x[0] = {0, 1};
    y[0] = {1, -1};
    z = x * y;
    EXPECT_EQ(z[0], TypeParam(1, 1));
}

TYPED_TEST(tComplex, Assignment) {
    Complex<TypeParam> vec1(2);
    vec1[0] = {1, -1};
    vec1[1] = {0, 1};
    Complex<TypeParam> vec2(2);
    vec2 = vec1;
    EXPECT_EQ(vec2[1], TypeParam(0, 1));
}

TYPED_TEST(tComplex, ScalarMultiplication) {
    Complex<TypeParam> vec1(1), vec2(1);
    vec1[0] = {-1, 1};
    TypeParam a(0, 1);
    vec2 = a * vec1;
    EXPECT_EQ(vec2[0], TypeParam(-1, -1));
}

TYPED_TEST(tComplex, Abs) {
    Complex<TypeParam> vec1(2), vec2(2);
    vec1[0] = {3, -4};
    vec1[1] = {0, -1};
    vec2 = abs(vec1);
    EXPECT_EQ(vec2[0], TypeParam(5, 0));
    EXPECT_EQ(vec2[1], TypeParam(1, 0));
}

TYPED_TEST(tComplex, Arg) {
    Complex<TypeParam> vec1(2), vec2(2);
    vec1[0] = {-1, 0};
    vec1[1] = {0, -1};
    vec2 = arg(vec1);
    EXPECT_EQ(vec2[0], TypeParam(std::numbers::pi, 0));
    EXPECT_EQ(vec2[1], TypeParam(-std::numbers::pi/2, 0));
}

TYPED_TEST(tComplex, Conj) {
    Complex<TypeParam> vec1(2), vec2(2);
    vec1[0] = {-1, 0};
    vec1[1] = {1, -2};
    vec2 = conj(vec1);
    EXPECT_EQ(vec2[0], TypeParam(-1, 0));
    EXPECT_EQ(vec2[1], TypeParam(1, 2));
}

TYPED_TEST(tComplex, Cos) {
    Complex<TypeParam> vec1(2), vec2(2);
    vec1[0] = {std::numbers::pi, 4};
    vec1[1] = {std::numbers::pi/2, -2};
    vec2 = cos(vec1);
    EXPECT_NEAR(vec2[0].real(), -1, 1e-4);
    EXPECT_NEAR(vec2[1].real(), 0, 1e-4);
}
