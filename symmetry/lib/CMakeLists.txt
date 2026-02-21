# Copyright 2023 Caleb Magruder

project(Symmetry)

set(SYMMETRY_SRCS
    ${CMAKE_SOURCE_DIR}/src/Color.cpp
    ${CMAKE_SOURCE_DIR}/src/ColorMap.cpp
    ${CMAKE_SOURCE_DIR}/src/Image.cpp
    ${CMAKE_SOURCE_DIR}/src/Param.cpp
    ${CMAKE_SOURCE_DIR}/src/PBar.cpp
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LIB_TYPE SHARED)
else()
    set(LIB_TYPE STATIC)
endif()

if(CMAKE_CUDA_COMPILER)
    add_library(Symmetry ${LIB_TYPE}
        ${SYMMETRY_SRCS}
        ${CMAKE_SOURCE_DIR}/src/Complex.cpp
        ${CMAKE_SOURCE_DIR}/src/Complex.cu
        ${CMAKE_SOURCE_DIR}/src/FPI.cu
    )
    target_link_libraries(Symmetry cublas)
    target_compile_definitions(Symmetry PUBLIC CMAKE_CUDA_COMPILER)
else()
    add_library(Symmetry ${LIB_TYPE}
        ${SYMMETRY_SRCS}
    )
endif()

find_library(json json)
find_path(json_INCLUDE JSON.h)

target_include_directories(Symmetry PUBLIC ${CMAKE_SOURCE_DIR}/include ${json_INCLUDE})

target_link_libraries(Symmetry ${json})
