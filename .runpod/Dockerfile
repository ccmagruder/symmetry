# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.245.2/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11, Ubuntu 18.04/22.04 on local arm64/Apple Silicon): debian-11, debian-10, ubuntu-22.04, ubuntu-20.04, ubuntu-18.04
FROM runpod/vscode-server:0.0.0

# [Optional] Uncomment this section to install additional packages.
RUN apt-get update \
    && mkdir -p /workspace \
    && cd /workspace \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && mkdir -p /opt/cmake \
    && curl -sSL https://github.com/Kitware/CMake/releases/download/v3.27.8/cmake-3.27.8-linux-x86_64.sh -O \
    && sh cmake-3.27.8-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && rm cmake-3.27.8-linux-x86_64.sh \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && ln -s /opt/cmake/bin/ctest /usr/local/bin/ctest \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        clang \
        cppcheck \
        git-lfs \
        libgtest-dev \
        make \
        pip \
        vim \
        zsh \
    && chsh -s /usr/bin/zsh \
    && pip install cpplint \
    && git clone https://github.com/ccmagruder/json.git \
    && cd json \
    && cmake . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build \
    && cmake --install build \
    && cd .. \
    && rm -rf json \
    && cd /workspace \
    && git clone https://github.com/ccmagruder/symmetry.git \
    && export GTEST_COLOR=1
