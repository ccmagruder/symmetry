VERSION 0.8
FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++

RUN apt update && apt install --no-install-recommends -y build-essential ca-certificates cmake clang cppcheck gcc git libgtest-dev pipx

WORKDIR /workspace

deps:
  RUN pipx ensurepath
  RUN pipx install cpplint
  RUN git clone https://github.com/ccmagruder/json.git
  WORKDIR json
  RUN cmake . -B build -DCMAKE_BUILD_TYPE=Release
  RUN cmake --build build
  RUN cmake --install build

code:
  FROM +deps
  WORKDIR /workspace
  COPY symmetry/CMakeLists.txt symmetry/CMakeLists.txt
  COPY symmetry/config/CMakeLists.txt symmetry/config/CMakeLists.txt
  COPY symmetry/images/CMakeLists.txt symmetry/images/CMakeLists.txt
  COPY symmetry/src symmetry/src
  COPY symmetry/include symmetry/include
  COPY symmetry/lib symmetry/lib
  COPY symmetry/tests symmetry/tests

build:
  FROM +code
  RUN export PATH=$PATH:/root/.local/bin
  RUN cmake -B build -DCMAKE_BUILD_TYPE=Debug symmetry
  RUN export PATH=$PATH:/root/.local/bin && cmake --build build

test:
  FROM +build
  WORKDIR build
  RUN ctest

