#!/bin/zsh

pushd $(dirname "$0")

if [ -d build ]; then
  rm -rf build
fi

export PATH=/opt/homebrew/bin:/usr/local/bin/:$PATH

cmake . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_C_COMPILER=/usr/bin/clang

if [ ! $? -eq 0 ]; then
  echo "<<<<<<<< CMake Configure Failed >>>>>>>>"
  exit 
fi

cmake --build build -j10

if [ ! $? -eq 0 ]; then
  echo "<<<<<<<< CMake Build Failed >>>>>>>>"
  exit 1
fi

pushd build

ctest --rerun-failed --output-on-failure

if [ ! $? -eq 0 ]; then
  echo "<<<<<<<< CTest Failed >>>>>>>>"
  exit 1
else
  echo "<<<<<<<< Build and Test Passed >>>>>>>>"
fi

popd

popd
