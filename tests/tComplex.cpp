// Copyright 2023 Caleb Magruder

#include <complex>

#include "gtest/gtest.h"

#include "Complex.hpp"

template <typename T>
class tComplex : public testing::Test {};

using Types = ::testing::Types <
    std::complex<float>,
    std::complex<double>
#ifdef CMAKE_CUDA_COMPILER
    , gpuDouble
#endif
>;

TYPED_TEST_SUITE(tComplex, Types);

TYPED_TEST(tComplex, Ctor) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec(2);
    vec = {3, 4, 1.4, -1};
    std::initializer_list<Type> ans
        = {3, 4, 1.4, -1};
    EXPECT_EQ(vec, ans);
    std::initializer_list<Type> wrong_ans
        = {3, 4, 1.5, -1};
    EXPECT_NE(vec, wrong_ans);
}

TYPED_TEST(tComplex, Assignment) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1(2);
    vec1 = {1, -1, 0, 1};
    Complex<TypeParam> vec2(2);
    vec2 = vec1;
    std::initializer_list<Type> ans = {1, -1, 0, 1};
    EXPECT_EQ(vec2, ans);
}

TYPED_TEST(tComplex, Addition) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> x(1), y(1);
    x = {0, 1};
    y = {1.5, -1};
    std::initializer_list<Type> ans = {1.5, 0};
    EXPECT_EQ(x + y, ans);
}

TYPED_TEST(tComplex, Multiplication) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> x(1), y(1);
    x = {0, 1};
    y = {1, -1};
    std::initializer_list<Type> ans = {1, 1};
    EXPECT_EQ(x * y, ans);
}

TYPED_TEST(tComplex, ScalarMultiplication) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1(1), vec2(1);
    vec1 = {-1, 1};
    Type ad[2] = {0, 1};
    // TypeParam a(0, 1);
    vec2 = *reinterpret_cast<TypeParam*>(ad) * vec1;
    std::initializer_list<Type> ans = {-1, -1};
    EXPECT_EQ(vec2, ans);
}

TYPED_TEST(tComplex, Abs) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1(2), vec2(2);
    vec1 = {3, -4, 0, -1};
    vec2 = abs(vec1);
    std::initializer_list<Type> ans = {5, 0, 1, 0};
    EXPECT_EQ(vec2, ans);
}

TYPED_TEST(tComplex, Arg) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1(2), vec2(2);
    vec1 = {-1, 0, 0, -1};
    vec2 = arg(vec1);
    Type* data
        = reinterpret_cast<Type*>(&vec2[0]);
    EXPECT_NEAR(data[0], 3.14159, 1e-4);
    EXPECT_EQ(data[1], 0);
    EXPECT_NEAR(data[2], -3.14159/2, 1e-4);
    EXPECT_EQ(data[3], 0);
}

TYPED_TEST(tComplex, Conj) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1(2), vec2(2);
    vec1 = {-1, 0, 1, -2};
    vec2 = conj(vec1);
    std::initializer_list<Type> ans = {-1, 0, 1, 2};
    EXPECT_EQ(vec2, ans);
}

TYPED_TEST(tComplex, Cos) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1(2), vec2(2);
    vec1 = {3.14159, 4, 3.14159/2, -2};
    vec2 = cos(vec1);
    Type* data
        = reinterpret_cast<Type*>(&vec2[0]);
    EXPECT_NEAR(data[0], -1, 1e-4);
    EXPECT_EQ(data[1], 4);
    EXPECT_NEAR(data[2], 0, 1e-4);
    EXPECT_EQ(data[3], -2);
}
