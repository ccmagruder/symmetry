# Copyright 2023 Caleb Magruder

cmake_minimum_required(VERSION 3.22)
project(Symmetry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include(CheckLanguage)
check_language(CUDA)

find_program(NVIDIA_SMI NAMES nvidia-smi)
if(NVIDIA_SMI STREQUAL "NVIDIA_SMI-NOTFOUND")
    message(WARNING "Compiler nvcc found; however, GPU not found. Disabling CUDA.")
    set(CMAKE_CUDA_COMPILER "")
endif()

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

add_subdirectory(config)
add_subdirectory(images)
add_subdirectory(lib)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(tests)
endif()

add_executable(symmetry src/Symmetry.cpp)

target_link_libraries(symmetry Symmetry)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    enable_testing()
endif()
