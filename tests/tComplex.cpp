// Copyright 2023 Caleb Magruder

#include <complex>

#include "gtest/gtest.h"

#include "Complex.hpp"

template <typename T>
class tComplex : public testing::Test {};

using Types = ::testing::Types <
    float,
    double
#ifdef CMAKE_CUDA_COMPILER
    , gpuDouble
#endif
>;

TYPED_TEST_SUITE(tComplex, Types);

TYPED_TEST(tComplex, CtorInitializerList) {
    using Type = typename complex_traits<TypeParam>::value_type;
    using List = std::initializer_list<Type>;
    Complex<TypeParam> vec{3, 4, 1.4, -1};
    EXPECT_EQ(vec, List({3, 4, 1.4, -1}));
    EXPECT_NE(vec, List({3, 4, 1.5, -1}));
}

TYPED_TEST(tComplex, CopyCtor) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1{1, -1, 0, 1};
    Complex<TypeParam> vec2(vec1);
    EXPECT_EQ(vec2, std::initializer_list<Type>({1, -1, 0, 1}));
}

TYPED_TEST(tComplex, Addition) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> x{0, 1}, y{1.5, -1};
    std::initializer_list<Type> ans = {1.5, 0};
    EXPECT_EQ(x + y, ans);
}

TYPED_TEST(tComplex, Multiplication) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> x{0, 1}, y{1, -1};
    EXPECT_EQ(x * y, std::initializer_list<Type>({1, 1}));
}

TYPED_TEST(tComplex, ScalarMultiplication) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1{-1, 1};
    std::complex<Type> ad = {0, 1};
    Complex<TypeParam> vec2(ad * vec1);
    EXPECT_EQ(vec2, std::initializer_list<Type>({-1, -1}));
}

TYPED_TEST(tComplex, Abs) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1{3, -4, 0, -1};
    EXPECT_EQ(abs(vec1), std::initializer_list<Type>({5, 0, 1, 0}));
}

TYPED_TEST(tComplex, Arg) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1{-1, 0, 0, -1};
    Complex<TypeParam> vec2(arg(vec1));
    EXPECT_NEAR(vec2[0].real(), 3.14159, 1e-4);
    EXPECT_EQ(vec2[0].imag(), 0);
    EXPECT_NEAR(vec2[1].real(), -3.14159/2, 1e-4);
    EXPECT_EQ(vec2[1].imag(), 0);
}

TYPED_TEST(tComplex, Conj) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1{-1, 0, 1, -2};
    std::initializer_list<Type> ans = {-1, 0, 1, 2};
    EXPECT_EQ(conj(vec1), ans);
}

TYPED_TEST(tComplex, Cos) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1{3.14159, 4, 3.14159/2, -2};
    Complex<TypeParam> vec2(cos(vec1));
    EXPECT_NEAR(vec2[0].real(), -1, 1e-4);
    EXPECT_EQ(vec2[0].imag(), 4);
    EXPECT_NEAR(vec2[1].real(), 0, 1e-4);
    EXPECT_EQ(vec2[1].imag(), -2);
}
