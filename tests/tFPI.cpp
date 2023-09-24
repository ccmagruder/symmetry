// Copyright 2022 Caleb Magruder

#include "gtest/gtest.h"

#include "FPI.h"

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

TEST(tFPI, SaveLoad) {
    Param p("config/test_iter10.json");
    p.resy = 1;
    p.resx = 1;
    FPI fpi(p);
    fpi.run_fpi(100);
    fpi.save();
    FPI fpi2 = FPI::load(p);
    EXPECT_EQ(fpi, fpi2);
}

TEST(tFPI, SaveProgress) {
    Param p("config/test_iter10.json");
    p.resx = 20;
    p.resy = 20;

    std::string filename = FPI::getHashFilename(p);
    std::remove(filename.c_str());

    FPI fpi1 = FPI::load(p);
    // Noise is added to fpi every 1000 iterations so, for this test
    // to pass we need to send multiples of 1000.
    fpi1.run_fpi(1000);
    fpi1.save();

    FPI fpi2 = FPI::load(p);
    fpi2.run_fpi(2000);

    FPI fpi3(p);
    fpi3.run_fpi(3000);

    EXPECT_NE(fpi1, fpi3);
    EXPECT_EQ(fpi2, fpi3);
}
