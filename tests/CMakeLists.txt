project(Symmetry)

add_executable(tParamTest tparamtest.cpp)
add_executable(tImage tImage.cpp)
add_executable(tpgm2pbm tpgm2pbm.cpp)
add_executable(tColorBar tColorBar.cpp)

target_link_libraries(tParamTest SymmetryLib gtest gtest_main)
target_link_libraries(tImage SymmetryLib gtest gtest_main)
target_link_libraries(tColorBar SymmetryLib gtest gtest_main)
target_link_libraries(tpgm2pbm SymmetryLib gtest gtest_main)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/config
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_subdirectory(images)

enable_testing()

add_test(NAME tParamTest
         WORKING_DIRECTORY tests
         COMMAND tParamTest)

add_test(NAME symmetry
         WORKING_DIRECTORY tests
         COMMAND symmetry config/default.txt images/default.pgm)
set_tests_properties(symmetry PROPERTIES FIXTURES_SETUP T1)

add_test(NAME pgm2pbm
         WORKING_DIRECTORY tests
         COMMAND pgm2pbm images/default.pgm config/purpleMonochrome.txt)
set_tests_properties(pgm2pbm PROPERTIES FIXTURES_REQUIRED T1 FIXTURES_SETUP T2)

add_test(NAME tpgm2pbm
         WORKING_DIRECTORY tests
         COMMAND tpgm2pbm
         PROPERTIES FIXTURES_REQUIRED T1)

add_test(NAME colorbar
         WORKING_DIRECTORY tests
         COMMAND colorbar images/colorbar.pbm config/purpleMonochrome.txt)
set_tests_properties(colorbar PROPERTIES FIXTURES_REQUIRED T2)

add_test(NAME tImage
         WORKING_DIRECTORY tests
         COMMAND tImage)

add_test(NAME tColorBar
         WORKING_DIRECTORY tests
         COMMAND tColorBar)