FROM alpine:latest

RUN mkdir workspace \
    && cd workspace \
    && apk update \
    && apk add cmake make clang git build-base vim\
    && git clone https://www.github.com/ccmagruder/symmetry.git \
    && cd symmetry \
    && cmake -DCMAKE_BUILD_TYPE=Release . \
    && cmake --build . \
    && ln -s /workspace/Symmetry/symmetry /usr/bin/symmetry
