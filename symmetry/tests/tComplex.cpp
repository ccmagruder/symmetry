// Copyright 2023 Caleb Magruder
//
// Unit tests for the Complex class template.
//
// Tests cover construction, arithmetic operations, and mathematical functions
// for Complex arrays. Tests are parameterized over float, double, and
// gpuDouble (when CUDA is available) to verify both CPU and GPU code paths.

#include <complex>

#include "gtest/gtest.h"

#include "Complex.hpp"

// Typed test fixture for Complex class tests.
//
// Parameterized over numeric types to test CPU implementations (float, double)
// and GPU implementation (gpuDouble) with the same test cases.
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

// Tests construction from an initializer list.
//
// Verifies that Complex arrays can be constructed from brace-enclosed
// lists of real values and compared correctly.
TYPED_TEST(tComplex, CtorInitializerList) {
    using Type = typename complex_traits<TypeParam>::value_type;
    using List = std::initializer_list<Type>;
    Complex<TypeParam> vec{3, 4, 1.4, -1};
    EXPECT_EQ(vec, List({3, 4, 1.4, -1}));
    EXPECT_NE(vec, List({3, 4, 1.5, -1}));
}

// Tests the copy constructor.
//
// Verifies that copying a Complex array creates an independent copy
// with identical values.
TYPED_TEST(tComplex, CopyCtor) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec1{1, -1, 0, 1};
    Complex<TypeParam> vec2(vec1);
    EXPECT_EQ(vec2, std::initializer_list<Type>({1, -1, 0, 1}));
}

// Tests element-wise addition of two Complex arrays.
//
// Verifies that operator+ correctly adds corresponding complex elements.
TYPED_TEST(tComplex, Addition) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> x{0, 1, 3.25, 0.25}, y{1.5, -1, 1, -1.5};
    EXPECT_EQ(x += y, std::initializer_list<Type>({1.5, 0, 4.25, -1.25}));
}

// Tests element-wise multiplication of two Complex arrays.
//
// Verifies that operator* correctly multiplies corresponding complex elements.
TYPED_TEST(tComplex, Multiplication) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> x{0, -1, 0, 1}, y{1.5, -1, 1, 0};
    EXPECT_EQ(x *= y, std::initializer_list<Type>({-1, -1.5, 0, 1}));
}

// Tests scalar multiplication of a Complex array.
//
// Verifies that multiplying by a complex scalar scales all elements correctly.
TYPED_TEST(tComplex, ScalarMultiplication) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec{-1, 1, 1.25, -0.125};
    std::complex<Type> ad = {0, 1};
    vec *= ad;
    std::initializer_list<Type> ans({-1, -1, 0.125, 1.25});
    EXPECT_EQ(vec, ans);
}

// Tests element-wise absolute value (magnitude).
//
// Verifies that abs() computes the magnitude of each complex number,
// storing results in the real part with imaginary part set to zero.
TYPED_TEST(tComplex, Abs) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec{3, -4, 0, -1};
    vec.abs();
    EXPECT_EQ(vec, std::initializer_list<Type>({5, 0, 1, 0}));
}

// Tests element-wise argument (phase angle).
//
// Verifies that arg() computes the phase angle of each complex number,
// storing results in the real part with imaginary part set to zero.
TYPED_TEST(tComplex, Arg) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec{-1, 0, 0, -1};
    vec.arg();
    EXPECT_NEAR(vec[0].real(), 3.14159, 1e-4);
    EXPECT_EQ(vec[0].imag(), 0);
    EXPECT_NEAR(vec[1].real(), -3.14159/2, 1e-4);
    EXPECT_EQ(vec[1].imag(), 0);
}

// Tests element-wise complex conjugate.
//
// Verifies that conj() negates the imaginary part of each complex number.
TYPED_TEST(tComplex, Conj) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec{-1, 0, 1, -2};
    std::initializer_list<Type> ans = {-1, 0, 1, 2};
    vec.conj();
    EXPECT_EQ(vec, ans);
}

// Tests element-wise cosine of the real part.
//
// Verifies that cos() applies the cosine function to the real part
// of each complex number, leaving the imaginary part unchanged.
TYPED_TEST(tComplex, Cos) {
    using Type = typename complex_traits<TypeParam>::value_type;
    Complex<TypeParam> vec{3.14159, 4, 3.14159/2, -2};
    vec.cos();
    EXPECT_NEAR(vec[0].real(), -1, 1e-4);
    EXPECT_EQ(vec[0].imag(), 4);
    EXPECT_NEAR(vec[1].real(), 0, 1e-4);
    EXPECT_EQ(vec[1].imag(), -2);
}
