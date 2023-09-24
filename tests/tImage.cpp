// Copyright 2022 Caleb Magruder

#include <limits>

#include "gtest/gtest.h"

#include "Image.hpp"

using Types = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;

template<typename T>
class tCast : public ::testing::Test {};

TYPED_TEST_SUITE(tCast, Types);

TYPED_TEST(tCast, Types) {
    EXPECT_EQ(cast<TypeParam>(uint8_t(0xFF)),
        std::numeric_limits<TypeParam>::max());
    EXPECT_EQ(cast<TypeParam>(uint16_t(0xFFFF)),
        std::numeric_limits<TypeParam>::max());
    EXPECT_EQ(cast<TypeParam>(uint32_t(0xFFFFFFFF)),
        std::numeric_limits<TypeParam>::max());
    EXPECT_EQ(cast<TypeParam>(uint64_t(0xFFFFFFFFFFFFFFFF)),
        std::numeric_limits<TypeParam>::max());

    EXPECT_EQ(cast<TypeParam>(uint8_t(0xA3)),
        static_cast<TypeParam>(0xA3A3A3A3A3A3A3A3));
    EXPECT_EQ(cast<TypeParam>(uint16_t(0xA3A3)),
        static_cast<TypeParam>(0xA3A3A3A3A3A3A3A3));
    EXPECT_EQ(cast<TypeParam>(uint32_t(0xA3A3A3A3)),
        static_cast<TypeParam>(0xA3A3A3A3A3A3A3A3));
    EXPECT_EQ(cast<TypeParam>(uint64_t(0xA3A3A3A3A3A3A3A3)),
        static_cast<TypeParam>(0xA3A3A3A3A3A3A3A3));
}

template<typename T>
class tImageTyped : public ::testing::Test {};

TYPED_TEST_SUITE(tImageTyped, Types);

TYPED_TEST(tImageTyped, AssignmentInitializerList) {
    Image<TypeParam, 1> im1(2, 2);
    im1[0] = {2, 6};
    im1[1] = {100, 5};
    Image<TypeParam, 1> im2(2, 2);
    im2 = {2, 6, 100, 5};
    EXPECT_EQ(im1, im2);
}

TYPED_TEST(tImageTyped, Overflow) {
    Image<TypeParam, 1> im(1, 1);
    im[0][0] = std::numeric_limits<TypeParam>::max();
    EXPECT_EQ(im[0][0], std::numeric_limits<TypeParam>::max());
    EXPECT_THROW(++im[0][0], std::runtime_error);
}

TYPED_TEST(tImageTyped, EqualityOperator) {
    Image<TypeParam, 1> im1(1, 1);
    im1[0][0] = 0;

    Image<TypeParam, 1> im2(1, 1);
    im2[0][0] = 0;

    EXPECT_EQ(im1, im2);

    im2[0][0] = 1;
    EXPECT_NE(im1, im2);

    Image<TypeParam, 1> im3(2, 1);
    im3[0][0] = 0;
    EXPECT_NE(im1, im3);

    Image<TypeParam, 1> im4(1, 2);
    im4[0][0] = 0;
    EXPECT_NE(im1, im4);
}

TYPED_TEST(tImageTyped, PassThrough) {
    if (sizeof(TypeParam) <= 2) {
        // GrayMap Image
        Image<TypeParam, 1> im1(2, 2);
        im1 = {0, 1, 4, 255};
        im1.write("images/passthrough.pgm");
        Image<TypeParam, 1> im2("images/passthrough.pgm");
        EXPECT_EQ(im1, im2);
        std::remove("images/passthrough.pgm");

        // RGB Image
        Image<TypeParam, 3> im3(2, 2);
        im3 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        im3.write("images/passthrough.ppm");
        Image<TypeParam, 3> im4("images/passthrough.ppm");
        EXPECT_EQ(im3, im4);
        std::remove("images/passthrough.ppm");
    } else {
        Image<TypeParam, 1> im1(2, 2);
        EXPECT_THROW(im1.write("images/null.ppm"), std::runtime_error);
        Image<TypeParam, 3> im3(2, 2);
        EXPECT_THROW(im3.write("images/null.ppm"), std::runtime_error);
    }
}

TYPED_TEST(tImageTyped, Histogram) {
    Image<TypeParam, 1> im(2, 2);
    im = { cast<TypeParam>(uint8_t(1)),
           cast<TypeParam>(uint8_t(3)),
           cast<TypeParam>(uint8_t(4)),
           std::numeric_limits<TypeParam>::max() };
    std::vector<uint64_t> hist = im.hist();
    EXPECT_EQ(hist[0], 0);
    EXPECT_EQ(hist[1], 1);
    EXPECT_EQ(hist[2], 0);
    EXPECT_EQ(hist[3], 1);
    EXPECT_EQ(hist[4], 1);
    for (int i = 5; i < 255; i++) {
        EXPECT_EQ(hist[i], 0);
    }
    EXPECT_EQ(hist[255], 1);
}

TYPED_TEST(tImageTyped, Equalize) {
    Image<TypeParam, 1> im(2, 2);

    im = { 0, 1, 3, 7 };
    im.logRescale(1);

    constexpr double max = static_cast<double>(
        std::numeric_limits<TypeParam>::max());

    EXPECT_EQ(im[0][0], 0);
    EXPECT_EQ(im[0][1], max / 3);
    EXPECT_EQ(im[1][0], max * 2 / 3);
    EXPECT_EQ(im[1][1], max);

    im = { 0, 1, 127, 255 };
    im.logRescale(1);

    EXPECT_EQ(im[0][0], 0);
    EXPECT_EQ(im[0][1], std::floor(max / 8));
    EXPECT_EQ(im[1][0], std::floor(max * 7 / 8));
    EXPECT_EQ(im[1][1], max);

    im = { 0, 0, 0, 0 };
    im.logRescale(1);
    EXPECT_EQ(im[0][0], 0);
    EXPECT_EQ(im[0][1], 0);
    EXPECT_EQ(im[1][0], 0);
    EXPECT_EQ(im[1][1], 0);
}

TYPED_TEST(tImageTyped, Const) {
    Image<TypeParam, 1> im(2, 2);
    im[0] = {1, 2, 3, 4};
    const Image<TypeParam, 1>& cim = im;
    EXPECT_EQ(cim[0], 1);
    EXPECT_EQ(cim[1], 3);
    EXPECT_EQ(cim[1][1], 4);
}
