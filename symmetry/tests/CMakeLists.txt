# Copyright 2023 Caleb Magruder

project(Symmetry)

find_package(GTest REQUIRED)
include(GoogleTest)

find_package(benchmark REQUIRED)

add_executable(tColorMap tColorMap.cpp)
add_executable(tComplex tComplex.cpp)
add_executable(tFPI tFPI.cpp)
add_executable(tImage tImage.cpp)
add_executable(tParam tParam.cpp)
add_executable(tPixel tPixel.cpp)

target_link_libraries(tColorMap Symmetry GTest::gtest GTest::gtest_main)
target_link_libraries(tComplex Symmetry GTest::gtest GTest::gtest_main)
target_link_libraries(tFPI Symmetry GTest::gtest GTest::gtest_main)
target_link_libraries(tImage Symmetry GTest::gtest GTest::gtest_main)
target_link_libraries(tParam Symmetry GTest::gtest GTest::gtest_main)
target_link_libraries(tPixel Symmetry GTest::gtest GTest::gtest_main)

add_executable(bComplex bComplex.cpp)
add_executable(bFPI bFPI.cpp)

target_link_libraries(bComplex Symmetry benchmark::benchmark benchmark::benchmark_main)
target_link_libraries(bFPI Symmetry benchmark::benchmark benchmark::benchmark_main)

install(TARGETS bComplex bFPI DESTINATION bin)

add_subdirectory(config)
add_subdirectory(images)

enable_testing()

add_test(NAME tColorMap
         WORKING_DIRECTORY tests
         COMMAND tColorMap)

add_test(NAME tComplex
         WORKING_DIRECTORY tests
         COMMAND tComplex)

add_test(NAME tFPI
         WORKING_DIRECTORY tests
         COMMAND tFPI)

add_test(NAME tImage
         WORKING_DIRECTORY tests
         COMMAND tImage)

add_test(NAME tParam
         WORKING_DIRECTORY tests
         COMMAND tParam)

add_test(NAME tPixel
         WORKING_DIRECTORY tests
         COMMAND tPixel)

add_test(NAME symmetry
         WORKING_DIRECTORY tests
         COMMAND symmetry run config/test_iter10.json images/test_iter10.pgm)
