// Copyright 2022 Caleb Magruder

#include <sstream>

#include "gtest/gtest.h"

#include "ColorMap.h"
#include "Image.hpp"

TEST(tColorMap, MissingFile) {
    EXPECT_THROW(ColorMap map("config/nonexistent_file.csv"),
                 std::runtime_error);
}

TEST(tColorMap, LookupMonoChrome) {
    ColorMap map("config/bwgradient.csv");
    EXPECT_EQ(map(0), Color(0, 0, 0));
    // Nonlinear Interpolation via squared values. For details, see:
    // https://sighack.com/post/averaging-rgb-colors-the-right-way
    EXPECT_EQ(map(32767), Color(46339, 46339, 46339));
    EXPECT_EQ(map(32768), Color(46340, 46340, 46340));
    EXPECT_EQ(map(65535), Color(65535, 65535, 65535));
}

TEST(tColorMap, LookupMultiChrome) {
    ColorMap map("config/rgbgradient.csv");
    EXPECT_EQ(map(0), Color(65535, 0, 0));
    EXPECT_EQ(map(16383), Color(46340, 46339, 0));
    EXPECT_EQ(map(16384), Color(46339, 46340, 0));
    EXPECT_EQ(map(32767), Color(0, 65535, 0));
    EXPECT_EQ(map(49151), Color(0, 46340, 46340));
    EXPECT_EQ(map(65535), Color(0, 0, 65535));
}

TEST(tColorMap, Write) {
    ColorMap map("config/rgbgradient.csv");
    map.write("images/colorbar.ppm");
    Image<uint16_t, 3> colorbar("images/colorbar.ppm");
}

TEST(tColorMap, Map) {
    Image<uint16_t, 1> im(2, 2);
    im[0][0] = 0;
    im[0][1] = 16383;
    im[1][0] = 32767;
    im[1][1] = 65535;
    ColorMap map("config/rgbgradient.csv");
    Image<uint16_t, 3> ppm = map(im);
    ppm.write("images/tColorBar_Map.ppm");
}

