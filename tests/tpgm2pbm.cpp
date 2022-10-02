// Copyright 2022 Caleb Magruder

#include "gtest/gtest.h"

#include "image.h"

TEST(tpgm2pbm, Test) {
    image im("images/default.pgm");
    ASSERT_EQ(im.getRows(), 2);
    ASSERT_EQ(im.getCols(), 2);
    ASSERT_EQ(im.getColors(), 1);
    // EXPECT_EQ(im[0][0], 10922);
    // EXPECT_EQ(im[0][1], 65535);
    // EXPECT_EQ(im[1][0], 10922);
    // EXPECT_EQ(im[1][1], 21845);
    EXPECT_EQ(im[0][0], 65535);
    EXPECT_EQ(im[0][1], 0);
    EXPECT_EQ(im[1][0], 65535);
    EXPECT_EQ(im[1][1], 0);
}