// Copyright 2022 Caleb Magruder

#include "gtest/gtest.h"

#include "Param.h"

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

TEST(Param, Hash) {
    size_t h1 = Param::hash(Param("config/test_iter10.json"));
    Param p("config/test_iter10.json");
    size_t h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);

    p.lambda = 2;
    h2 = Param::hash(p);
    EXPECT_NE(h1, h2);
    p.lambda = 1.56;
    h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);

    p.alpha = 1;
    h2 = Param::hash(p);
    EXPECT_NE(h1, h2);
    p.alpha = -1;
    h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);

    p.beta = 1;
    h2 = Param::hash(p);
    EXPECT_NE(h1, h2);
    p.beta = 0.1;
    h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);

    p.gamma = 1;
    h2 = Param::hash(p);
    EXPECT_NE(h1, h2);
    p.gamma = -0.82;
    h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);

    p.omega = 1;
    h2 = Param::hash(p);
    EXPECT_NE(h1, h2);
    p.omega = 0;
    h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);

    p.n = 1;
    h2 = Param::hash(p);
    EXPECT_NE(h1, h2);
    p.n = 3;
    h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);

    p.scale = 1;
    h2 = Param::hash(p);
    EXPECT_NE(h1, h2);
    p.scale = 0.65;
    h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);

    p.n_iter = 1;
    h2 = Param::hash(p);
    EXPECT_NE(h1, h2);
    p.n_iter = 10;
    h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);

    p.resx = 1;
    h2 = Param::hash(p);
    EXPECT_NE(h1, h2);
    p.resx = 2;
    h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);

    p.resy = 3;
    h2 = Param::hash(p);
    EXPECT_NE(h1, h2);
    p.resy = 2;
    h2 = Param::hash(p);
    EXPECT_EQ(h1, h2);
}
