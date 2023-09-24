// Copyright 2022 Caleb Magruder

#include <limits>

#include "gtest/gtest.h"

#include "Pixel.hpp"

template<typename T>
class tPixel : public ::testing::Test {};

using Types = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;

TYPED_TEST_SUITE(tPixel, Types);

TYPED_TEST(tPixel, Overflow) {
    TypeParam* data = new TypeParam(std::numeric_limits<TypeParam>::max());
    Pixel<TypeParam, 1> p(data);
    EXPECT_THROW(++p, std::runtime_error);
    delete data;
}

TYPED_TEST(tPixel, IndexOperator) {
    TypeParam* data = new TypeParam[3];
    Pixel<TypeParam, 1> p(data);
    p = 1;
    p[1] = 2;
    p[2] = std::numeric_limits<TypeParam>::max();
    EXPECT_EQ(p[0], 1);
    EXPECT_EQ(++p[1], 3);
    EXPECT_EQ(p[2], std::numeric_limits<TypeParam>::max());
    EXPECT_THROW(++p[2], std::runtime_error);
}

TYPED_TEST(tPixel, ReadWrite) {
    TypeParam* data = new TypeParam[6];
    for (ptrdiff_t i = 0; i < 6; i++) data[i] = 0;

    Pixel<TypeParam, 1> bw(data);
    bw = 4;
    EXPECT_EQ(bw[0], 4);
    EXPECT_EQ(data[0], 4);
    bw[1] = {2, 6};
    EXPECT_EQ(data[1], 2);
    EXPECT_EQ(data[2], 6);

    Pixel<TypeParam, 3> rgb(data);
    rgb = {6, 5, 4, 3, 2, 1};
    EXPECT_EQ(rgb[1], 3);
    rgb[1] = {7, 8, 9};
    EXPECT_EQ(rgb[1][1], 8);

    const Pixel<TypeParam, 1> cbw(data);
    EXPECT_EQ(cbw[1], 5);

    delete[] data;
}
