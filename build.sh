#!/bin/zsh

# earthly --sat earthly-noble-arm64 -i +all
if [ -d build ]; then
  rm -rf build
fi

cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -B build -S symmetry \
  && cmake --build ./build \
  && ctest --test-dir build
