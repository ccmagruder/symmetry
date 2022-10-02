// Copyright 2022 Caleb Magruder

#include "gtest/gtest.h"

#include <image.h>

TEST(Image, PassThrough) {
    image im1(2, 2, 1);
    im1[0][0] = 0;
    im1[0][1] = 1;
    im1[1][0] = 2;
    im1[1][1] = 3;
    im1.write("images/PassThrough.pbm");
    image im2("images/PassThrough.pbm");
    ASSERT_EQ(im2[0][0], 0);
    ASSERT_EQ(im2[0][1], 1);
    ASSERT_EQ(im2[1][0], 2);
    ASSERT_EQ(im2[1][1], 3);
}