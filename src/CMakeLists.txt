cmake_minimum_required(VERSION 3.19)
project(Symmetry)

set(HEADER_FILES
        colormap.h
        fpi.h
        ini.h
        image.h
        INIReader.h
        param.h
        pbar.h
        )

set(SOURCE_FILES
        colormap.cpp
        fpi.cpp
        ini.cpp
        image.cpp
        INIReader.cpp
        param.cpp
        pbar.cpp
        )

add_library(SymmetryLib STATIC ${SOURCE_FILES} ${HEADER_FILES})
