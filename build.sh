#!/bin/zsh

# earthly sat ls
#
# if $?
# then
#   earthly account login --token $(cat /run/secrets/user_earthly_token)
#   earthly org select ccmagruder
#   earthly sat select earthly-noble-arm64
# fi
#
# earthly +test

if [ -d build ]; then
  rm -rf build
fi

export PATH=/opt/homebrew/bin:/usr/local/bin/:$PATH

cmake -B build -S symmetry -DCMAKE_BUILD_TYPE=Debug

if [ ! $? -eq 0 ]; then
  echo "<<<<<<<< CMake Configure Failed >>>>>>>>"
  exit 
fi

cmake --build build -j10

if [ ! $? -eq 0 ]; then
  echo "<<<<<<<< CMake Build Failed >>>>>>>>"
  exit 1
fi

ctest --rerun-failed --output-on-failure --test-dir build

if [ ! $? -eq 0 ]; then
  echo "<<<<<<<< CTest Failed >>>>>>>>"
  exit 1
else
  echo "<<<<<<<< Build and Test Passed >>>>>>>>"
fi

