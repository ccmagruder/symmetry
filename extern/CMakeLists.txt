# Copyright 2022 Caleb Magruder

project(Symmetry)

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/extern)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Google Test
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )

    set(BUILD_GMOCK OFF)

    FetchContent_MakeAvailable(googletest)
endif()

# ccmagruder/JSON
FetchContent_Declare(
    JSON
    GIT_REPOSITORY https://github.com/ccmagruder/JSON.git
    GIT_TAG v0.0.3)
FetchContent_MakeAvailable(JSON)
