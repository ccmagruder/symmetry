# Copyright 2023 Caleb Magruder

cmake_minimum_required(VERSION 3.22)
project(Symmetry LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

add_subdirectory(config)
add_subdirectory(images)
add_subdirectory(lib)
add_subdirectory(tests)

add_executable(symmetry src/Symmetry.cpp)
target_link_libraries(symmetry Symmetry)

install(TARGETS symmetry DESTINATION bin)

enable_testing()
