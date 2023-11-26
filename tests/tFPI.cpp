// Copyright 2022 Caleb Magruder

#include "gtest/gtest.h"

#include "FPI.hpp"

TEST(tFPI, Ctor) {
    Param p("config/test_iter10.json");
    FPI fpi(p);
    EXPECT_EQ(fpi.rows(), 2);
    EXPECT_EQ(fpi.cols(), 2);
    EXPECT_EQ(fpi.min(), 0);
    EXPECT_EQ(fpi.max(), 0);
    fpi.run_fpi();
    EXPECT_EQ(fpi[0][0] + fpi[0][1] + fpi[1][0] + fpi[1][1], 10);
    EXPECT_EQ(fpi.min(), 0);
}

TEST(tFPI, Write) {
    Param p("config/test_iter10.json");
    FPI fpi(p);
    fpi.run_fpi(100);
    EXPECT_EQ(fpi[0][0] + fpi[0][1] + fpi[1][0] + fpi[1][1], 100);
    fpi.write("images/tFPI_Write.pgm");
    Image<uint16_t, 1> im("images/tFPI_Write.pgm");
    std::remove("images/tFPI_Write.pgm");
}
