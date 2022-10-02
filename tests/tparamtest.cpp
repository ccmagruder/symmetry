// Copyright 2022 Caleb Magruder

#include "gtest/gtest.h"

#include "param.h"

TEST(Param, ReadFile){
    param p((char*) "config/default.txt");
    ASSERT_EQ(p.lambda, 1.56);
    ASSERT_EQ(p.alpha, -1);
    ASSERT_EQ(p.beta, 0.1);
    ASSERT_EQ(p.gamma, -0.82);
    ASSERT_EQ(p.omega, 0);
    ASSERT_EQ(p.n, 3);
    ASSERT_EQ(p.scale, 0.65);
    ASSERT_EQ(p.n_iter, 10);
    ASSERT_EQ(p.resx, 2);
    ASSERT_EQ(p.resy, 2);
}

TEST(Param, NonExistentFile){
    ASSERT_THROW(param p((char*) "nonexistent"), int);
}

TEST(Param, NonExistentSection){
    ASSERT_THROW(param p((char*) "missingImage"),int);
}

TEST(Param, NonExistentValue){
    ASSERT_THROW(param p((char*) "missingAlpha"), int);
}