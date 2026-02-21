// Copyright 2026 Caleb Magruder

#include "gtest/gtest.h"

#include "Param.h"

TEST(Param, Initialize) {
    Param p;
    EXPECT_NEAR(p.lambda, 0, 1e-16);
    EXPECT_NEAR(p.alpha, 0, 1e-16);
    EXPECT_NEAR(p.beta, 0, 1e-16);
    EXPECT_NEAR(p.gamma, 0, 1e-16);
    EXPECT_NEAR(p.omega, 0, 1e-16);
    EXPECT_NEAR(p.n, 0, 1e-16);
    EXPECT_NEAR(p.delta, 0, 1e-16);
    EXPECT_NEAR(p.p, 0, 1e-16);
    EXPECT_NEAR(p.scale, 0, 1e-16);
    EXPECT_NEAR(p.n_iter, 0, 1e-16);
    EXPECT_NEAR(p.resx, 1, 1e-16);
    EXPECT_NEAR(p.resy, 1, 1e-16);
}

TEST(Param, ReadFile) {
    std::string fileName("config/test_iter10.json");
    Param p(fileName);
    EXPECT_EQ(p.lambda, 1.56);
    EXPECT_EQ(p.alpha, -1);
    EXPECT_EQ(p.beta, 0.1);
    EXPECT_EQ(p.gamma, -0.82);
    EXPECT_EQ(p.omega, 0);
    EXPECT_EQ(p.n, 3);
    EXPECT_EQ(p.delta, 0);
    EXPECT_EQ(p.p, 0);
    EXPECT_EQ(p.scale, 0.65);
    EXPECT_EQ(p.n_iter, 10);
    EXPECT_EQ(p.resx, 2);
    EXPECT_EQ(p.resy, 2);
}

