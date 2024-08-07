FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

RUN mkdir -p /workspace \
    && cd /workspace \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get update \
    && apt-get -y install --no-install-recommends \
        clang \
        cppcheck \
        curl \
        git-lfs \
        less \
        libgtest-dev \
        make \
        pip \
        vim \
        zsh \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && mkdir -p /opt/cmake \
    && curl -sSL https://github.com/Kitware/CMake/releases/download/v3.27.8/cmake-3.27.8-linux-x86_64.sh -O \
    && sh cmake-3.27.8-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && rm cmake-3.27.8-linux-x86_64.sh \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && ln -s /opt/cmake/bin/ctest /usr/local/bin/ctest \
    && pip install cpplint \
    && chsh -s /usr/bin/zsh \
    && git clone https://github.com/ccmagruder/json.git \
    && cd json \
    && cmake . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build \
    && cmake --install build \
    && cd .. \
    && rm -rf json \
    && cd /workspace \
    && git clone https://github.com/ccmagruder/symmetry.git \
    && export GTEST_COLOR=1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /home/vscode \
    && sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended \
    && git config --global core.pager "less -FRX"

CMD zsh

