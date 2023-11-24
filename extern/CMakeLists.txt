# Copyright 2023 Caleb Magruder

project(Symmetry)

include(FetchContent)

set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/extern)

# ccmagruder/JSON
FetchContent_Declare(
    JSON
    GIT_REPOSITORY https://github.com/ccmagruder/JSON.git
    GIT_TAG v0.0.3)
FetchContent_MakeAvailable(JSON)
