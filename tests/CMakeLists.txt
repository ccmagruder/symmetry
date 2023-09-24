project(Symmetry)

add_executable(tColorMap tColorMap.cpp)
add_executable(tFPI tFPI.cpp)
add_executable(tImage tImage.cpp)
add_executable(tParam tParam.cpp)
add_executable(tPixel tPixel.cpp)

target_link_libraries(tColorMap Symmetry gtest gtest_main)
target_link_libraries(tFPI Symmetry gtest gtest_main)
target_link_libraries(tImage Symmetry gtest gtest_main)
target_link_libraries(tParam Symmetry gtest gtest_main)
target_link_libraries(tPixel Symmetry gtest gtest_main)

add_subdirectory(config)
add_subdirectory(images)

enable_testing()

add_test(NAME tColorMap
         WORKING_DIRECTORY tests
         COMMAND tColorMap)

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
