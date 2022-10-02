# Copyright 2022 Caleb Magruder

project(Symmetry)

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/extern)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)

set(BUILD_GMOCK OFF)

FetchContent_MakeAvailable(googletest)

