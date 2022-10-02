// Copyright 2022 Caleb Magruder

#include "gtest/gtest.h"

#include "image.h"

TEST(tColorBar, Test) {
    image im("images/default.pbm");
    ASSERT_EQ(im.getRows(), 2);
    ASSERT_EQ(im.getCols(), 2);
    ASSERT_EQ(im.getColors(), 3);
    // EXPECT_EQ(im[0][0], 39400);
    // EXPECT_EQ(im[0][1], 52368);
    // EXPECT_EQ(im[1][0], 39400);
    // EXPECT_EQ(im[1][1], 52368);
    EXPECT_EQ(im[0][0], 0);
    EXPECT_EQ(im[0][1], 65535);
    EXPECT_EQ(im[1][0], 0);
    EXPECT_EQ(im[1][1], 65535);
}