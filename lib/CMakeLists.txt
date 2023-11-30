# Copyright 2023 Caleb Magruder

project(Symmetry)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/Color.h")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/ColorMap.h")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/Complex.hpp")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/FPI.hpp")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/Image.hpp")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/Param.h")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/PBar.h")
    set(CMAKE_CXX_CPPLINT "${CMAKE_CXX_CPPLINT};${CMAKE_SOURCE_DIR}/include/Pixel.hpp")
endif()

if(CMAKE_CUDA_COMPILER)
    add_library(Symmetry SHARED
        ${CMAKE_SOURCE_DIR}/src/Color.cpp
        ${CMAKE_SOURCE_DIR}/src/ColorMap.cpp
        ${CMAKE_SOURCE_DIR}/src/Image.cpp
        ${CMAKE_SOURCE_DIR}/src/Param.cpp
        ${CMAKE_SOURCE_DIR}/src/PBar.cpp
        ${CMAKE_SOURCE_DIR}/src/Complex.cu
    )
    target_compile_definitions(Symmetry PUBLIC CMAKE_CUDA_COMPILER)
else()
    add_library(Symmetry SHARED
        ${CMAKE_SOURCE_DIR}/src/Color.cpp
        ${CMAKE_SOURCE_DIR}/src/ColorMap.cpp
        ${CMAKE_SOURCE_DIR}/src/Image.cpp
        ${CMAKE_SOURCE_DIR}/src/Param.cpp
        ${CMAKE_SOURCE_DIR}/src/PBar.cpp
    )
endif()

find_library(json json)
find_path(json_INCLUDE JSON.h)

target_include_directories(Symmetry PUBLIC ${CMAKE_SOURCE_DIR}/include ${json_INCLUDE})

target_link_libraries(Symmetry ${json})
