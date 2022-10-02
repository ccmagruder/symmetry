cmake_minimum_required(VERSION 3.19)
project(Symmetry)

set(CMAKE_CXX_STANDARD 14)

include_directories(src)

add_subdirectory(config)
add_subdirectory(images)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(extern)

add_executable(symmetry src/symmetry.cpp)
add_executable(pgm2pbm src/pgm2pbm.cpp)
add_executable(colorbar src/colorbar.cpp)
add_executable(cui src/cui.cpp)

target_link_libraries(symmetry SymmetryLib)
target_link_libraries(pgm2pbm SymmetryLib)
target_link_libraries(colorbar SymmetryLib)
target_link_libraries(cui SymmetryLib ncurses)

install(TARGETS symmetry)

enable_testing()