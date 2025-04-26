FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

RUN apt update -y && apt upgrade -y \
  && apt install -y --no-install-recommends \
    build-essential \
    clang \
    clangd \
    clang-format \
    cmake \
    gcc \
    git \
    libbenchmark-dev \
    libgflags-dev \
    libgtest-dev \
    # Dependency for clangd-tidy
    python3 \
    python3-pip \
    python3-venv \
  && rm -rf /var/lib/apt/lists/* \
  && python3 -m venv /.venv \
  && PATH=/.venv/bin:$PATH pip install \
    clangd-tidy \
