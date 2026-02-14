// Copyright 2022 Caleb Magruder
//
// Unit tests for the FPI class template.
//
// Tests verify each term of the chaotic map F(z) in isolation by setting a
// single coefficient to a non-zero value and checking the resulting iterate.
// Additional tests cover construction from a config file, histogram
// accumulation, and writing output images.

#include "gtest/gtest.h"

#include "Complex.hpp"
#include "FPI.hpp"
#include "Param.h"

// Test harness that exposes FPI internals for unit testing.
//
// Disables transient iteration and noise perturbation so each test can
// evaluate a single application of F(z) from a known starting point
// z = (1, 0.5).
class TestFPI: public FPI<double> {
 public:
    explicit TestFPI(Param p) : FPI(p) {
        this->_init_iter = 0;
        this->_z = std::complex<double>(1, 0.5);
        this->_add_noise = false;
    }
    TestFPI() = delete;
    const std::complex<double>& z() { return this->_z; }
};

// Tests the lambda term: F(z) = lambda * z.
//
// With lambda=2 and z=(1,0.5), expects F(z) = (2, 1).
TEST(tFPI, FLambda) {
    Param p;
    p.lambda = 2;
    TestFPI fpi(p);
    fpi.run_fpi(1);
    EXPECT_NEAR(real(fpi.z()), 2, 1e-15);
    EXPECT_NEAR(imag(fpi.z()), 1, 1e-15);
}

// Tests the omega term: F(z) = i*omega * z.
//
// With omega=2 and z=(1,0.5), expects F(z) = (-1, 2).
TEST(tFPI, FOmega) {
    Param p;
    p.omega = 2;
    TestFPI fpi(p);
    fpi.run_fpi(1);
    EXPECT_NEAR(real(fpi.z()), -1, 1e-15);
    EXPECT_NEAR(imag(fpi.z()), 2, 1e-15);
}

// Tests the alpha term: F(z) = alpha * |z|^2 * z.
//
// With alpha=1 and z=(1,0.5), |z|^2 = 1.25, expects F(z) = (1.25, 0.625).
TEST(tFPI, FAlpha) {
    Param p;
    p.alpha = 2;
    TestFPI fpi(p);
    fpi.run_fpi(1);
    EXPECT_NEAR(real(fpi.z()), 2*1*1.25, 1e-15);
    EXPECT_NEAR(imag(fpi.z()), 2*0.5*1.25, 1e-15);
}

// Tests the beta term: F(z) = beta * Re(z^n) * z.
//
// With beta=1, n=2, and z=(1,0.5), Re(z^2) = 0.75, expects F(z) = (0.75, 0.375).
TEST(tFPI, FBeta) {
    Param p;
    p.beta = 2;
    p.n = 2;
    TestFPI fpi(p);
    fpi.run_fpi(1);
    EXPECT_NEAR(real(fpi.z()), 2*0.75*1, 1e-15);
    EXPECT_NEAR(imag(fpi.z()), 2*0.75*0.5, 1e-15);
}

// Tests the delta term: F(z) = delta * cos(n*p*arg(z)) * |z| * z.
//
// With delta=1, n=1, p=2, and z=(1,0.5), verifies the angular modulation.
TEST(tFPI, FDelta) {
    Param p;
    p.delta = 2;
    p.n = 1;
    p.p = 2;
    TestFPI fpi(p);
    fpi.run_fpi(1);
    EXPECT_NEAR(real(fpi.z()), 2*0.75/std::sqrt(1.25)*1, 1e-15);
    EXPECT_NEAR(imag(fpi.z()), 2*0.75/std::sqrt(1.25)*0.5, 1e-15);
}

// Tests the gamma term: F(z) = gamma * conj(z)^{n-1}.
//
// With gamma=2, n=3, and z=(1,0.5), conj(z)^2 = (0.75, -1),
// expects F(z) = (1.5, -2).
TEST(tFPI, FGamma) {
    Param p;
    p.gamma = 2;
    p.n = 3;
    TestFPI fpi(p);
    fpi.run_fpi(1);
    EXPECT_NEAR(real(fpi.z()), 2 * 0.75, 1e-15);
    EXPECT_NEAR(imag(fpi.z()), 2 * -1, 1e-15);
}

// Tests construction from a JSON config and histogram accumulation.
//
// Verifies image dimensions, that 10 iterates are distributed across
// the 2x2 image, and that CPU and GPU results match.
TEST(tFPI, Ctor) {
    Param p("config/test_iter10.json");
    FPI<double> fpi(p);
    EXPECT_EQ(fpi.rows(), 2);
    EXPECT_EQ(fpi.cols(), 2);
    EXPECT_EQ(fpi.min(), 0);
    EXPECT_EQ(fpi.max(), 0);
    fpi.run_fpi();
    EXPECT_EQ(fpi[0][0] + fpi[0][1] + fpi[1][0] + fpi[1][1], 10);
    EXPECT_EQ(fpi.min(), 0);

    FPI<gpuDouble> fpi2(p);
    fpi2.run_fpi();
    EXPECT_EQ(fpi[0][0], fpi2[0][0]);
    EXPECT_EQ(fpi[0][1], fpi2[0][1]);
    EXPECT_EQ(fpi[1][0], fpi2[1][0]);
    EXPECT_EQ(fpi[1][1], fpi2[1][1]);
}

// Tests writing the histogram to a PGM image file.
//
// Runs 100 iterations, verifies the total histogram count, writes the
// image to disk, reads it back, and cleans up the temporary file.
TEST(tFPI, Write) {
    Param p("config/test_iter10.json");
    FPI fpi(p);
    fpi.run_fpi(100);
    EXPECT_EQ(fpi[0][0] + fpi[0][1] + fpi[1][0] + fpi[1][1], 100);
    fpi.write("images/tFPI_Write.pgm");
    Image<uint16_t, 1> im("images/tFPI_Write.pgm");
    std::remove("images/tFPI_Write.pgm");
}
