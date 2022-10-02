cmake_minimum_required(VERSION 3.19)
project(Symmetry)

set(SOURCE_FILES
        colormap.cpp
        fpi.cpp
        ini.cpp
        image.cpp
        INIReader.cpp
        param.cpp
        pbar.cpp
        )

add_library(SymmetryLib STATIC ${SOURCE_FILES})
