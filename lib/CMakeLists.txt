# Copyright 2022 Caleb Magruder

project(Symmetry)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/Color.h")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/ColorMap.h")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/FPI.h")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/Image.hpp")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/Param.h")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/PBar.h")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/Pixel.hpp")
endif()

add_library(Symmetry SHARED
    ${CMAKE_SOURCE_DIR}/src/Color.cpp
    ${CMAKE_SOURCE_DIR}/src/ColorMap.cpp
    ${CMAKE_SOURCE_DIR}/src/Image.cpp
    ${CMAKE_SOURCE_DIR}/src/FPI.cpp
    ${CMAKE_SOURCE_DIR}/src/Param.cpp
    ${CMAKE_SOURCE_DIR}/src/PBar.cpp
)

find_library(json json)
find_path(json_INCLUDE JSON.h)
message(FATAL ${json_INCLUDE})

target_include_directories(Symmetry PUBLIC ${CMAKE_SOURCE_DIR}/include ${json_INCLUDE})

target_link_libraries(Symmetry ${json})
