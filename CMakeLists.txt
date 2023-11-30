# Copyright 2022 Caleb Magruder

cmake_minimum_required(VERSION 3.22)
project(Symmetry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_CPPLINT "cpplint;--quiet;--filter=-build/include_subdir;--exclude=${CMAKE_BINARY_DIR}/*")
    set(CMAKE_CXX_CPPCHECK "cppcheck;--quiet;--suppress=*:${CMAKE_BINARY_DIR}/*")
endif()

add_subdirectory(config)
add_subdirectory(images)
add_subdirectory(lib)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(tests)
endif()

add_executable(symmetry src/Symmetry.cpp)

target_link_libraries(symmetry Symmetry)

if(CMAKE_CUDA_COMPILER)
    add_executable(complex src/Complex.cu)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    enable_testing()
endif()
