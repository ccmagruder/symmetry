VERSION 0.8
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND noninteractive
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

RUN apt update \
  && apt install --no-install-recommends -y build-essential \
     ca-certificates \
     ccls \
     cmake \
     clang \
     gcc \
     git \
     libgtest-dev \
     pipx \
  && pipx ensurepath

WORKDIR /workspace

IMPORT github.com/ccmagruder/json:feature/earthly AS json

deps:
  BUILD json+build
  COPY json+build/JSON.h /usr/local/include
  COPY json+build/Variant.h /usr/local/include
  COPY json+build/libjson.so /usr/local/lib

code:
  FROM +deps
  COPY symmetry/CMakeLists.txt CMakeLists.txt
  COPY symmetry/config/CMakeLists.txt config/CMakeLists.txt
  COPY symmetry/images/CMakeLists.txt images/CMakeLists.txt
  COPY symmetry/src src
  COPY symmetry/include include
  COPY symmetry/lib lib
  COPY symmetry/tests tests

build:
  FROM +code
  RUN cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES -B build -S .
  RUN --mount type=cache,target=/workspace/build/src/CMakeFiles \
      --mount type=cache,target=/workspace/build/lib/CMakeFiles \
      --mount type=cache,target=/workspace/build/tests/CMakeFiles \
      PATH=$PATH:/root/.local/bin cmake --build ./build
  SAVE ARTIFACT build/compile_commands.json

test:
  FROM +build
  RUN ctest --test-dir ./build

output:
  LOCALLY
  COPY +build/compile_commands.json ./

all:
  BUILD +test
  BUILD +output

