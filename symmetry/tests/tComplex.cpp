// Copyright 2023 Caleb Magruder
//
// Unit tests for the Complex class template.
//
// Tests cover construction, arithmetic operations, and mathematical functions
// for Complex arrays. Tests are parameterized over float, double, and
// gpuDouble (when CUDA is available) to verify both CPU and GPU code paths.

#include "gtest/gtest.h"

#include "Complex.hpp"

// Typed test fixture for Complex class tests.
//
// Parameterized over numeric types to test CPU implementations (float, double)
// and GPU implementation (gpuDouble) with the same test cases.
template <typename T>
class tComplex : public testing::Test {};

using Types = ::testing::Types <
    cpuDouble
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
    using Scalar = typename TypeParam::Scalar;
    using List = std::initializer_list<Scalar>;
    Complex<TypeParam> vec{3, 4, 1.4, -1};
    EXPECT_EQ(vec, List({3, 4, 1.4, -1}));
    EXPECT_NE(vec, List({3, 4, 1.5, -1}));
}

// Tests element-wise addition of two Complex arrays.
//
// Verifies that operator+ correctly adds corresponding complex elements.
TYPED_TEST(tComplex, Addition) {
    using Scalar = typename TypeParam::Scalar;
    Complex<TypeParam> x{0, 1, 3.25, 0.25}, y{1.5, -1, 1, -1.5};
    EXPECT_EQ(x += y, std::initializer_list<Scalar>({1.5, 0, 4.25, -1.25}));
}

// Tests element-wise multiplication of two Complex arrays.
//
// Verifies that operator* correctly multiplies corresponding complex elements.
TYPED_TEST(tComplex, Multiplication) {
    using Scalar = typename TypeParam::Scalar;
    Complex<TypeParam> x{0, -1, 0, 1}, y{1.5, -1, 1, 0};
    EXPECT_EQ(x *= y, std::initializer_list<Scalar>({-1, -1.5, 0, 1}));
}

// Tests scalar multiplication of a Complex array.
//
// Verifies that multiplying by a complex scalar scales all elements correctly.
TYPED_TEST(tComplex, ScalarMultiplication) {
    using Scalar = typename TypeParam::Scalar;
    Complex<TypeParam> vec{-1, 1, 1.25, -0.125};
    typename TypeParam::Type ad = {0, 1};
    vec *= ad;
    std::initializer_list<Scalar> ans({-1, -1, 0.125, 1.25});
    EXPECT_EQ(vec, ans);
}

// Tests element-wise absolute value (magnitude).
//
// Verifies that abs() computes the magnitude of each complex number,
// storing results in the real part with imaginary part set to zero.
TYPED_TEST(tComplex, Abs) {
    using Scalar = typename TypeParam::Scalar;
    Complex<TypeParam> vec{3, -4, 0, -1};
    vec.abs();
    EXPECT_EQ(vec, std::initializer_list<Scalar>({5, 0, 1, 0}));
}

// Tests element-wise argument (phase angle).
//
// Verifies that arg() computes the phase angle of each complex number,
// storing results in the real part with imaginary part set to zero.
TYPED_TEST(tComplex, Arg) {
    using Scalar = typename TypeParam::Scalar;
    Complex<TypeParam> vec{-1, 0, 0, -1};
    vec.arg();
    const Scalar* z0 = reinterpret_cast<const Scalar*>(&vec[0]);
    const Scalar* z1 = reinterpret_cast<const Scalar*>(&vec[1]);
    EXPECT_NEAR(z0[0], 3.14159, 1e-4);
    EXPECT_EQ(z0[1], 0);
    EXPECT_NEAR(z1[0], -3.14159/2, 1e-4);
    EXPECT_EQ(z1[1], 0);
}

// Tests element-wise complex conjugate.
//
// Verifies that conj() negates the imaginary part of each complex number.
TYPED_TEST(tComplex, Conj) {
    using Scalar = typename TypeParam::Scalar;
    Complex<TypeParam> vec{-1, 0, 1, -2};
    std::initializer_list<Scalar> ans = {-1, 0, 1, 2};
    vec.conj();
    EXPECT_EQ(vec, ans);
}

// Tests element-wise cosine of the real part.
//
// Verifies that cos() applies the cosine function to the real part
// of each complex number, leaving the imaginary part unchanged.
TYPED_TEST(tComplex, Cos) {
    using Scalar = typename TypeParam::Scalar;
    Complex<TypeParam> vec{3.14159, 4, 3.14159/2, -2};
    vec.cos();
    const Scalar* z0 = reinterpret_cast<const Scalar*>(&vec[0]);
    const Scalar* z1 = reinterpret_cast<const Scalar*>(&vec[1]);
    EXPECT_NEAR(z0[0], -1, 1e-4);
    EXPECT_EQ(z0[1], 4);
    EXPECT_NEAR(z1[0], 0, 1e-4);
    EXPECT_EQ(z1[1], -2);
}
