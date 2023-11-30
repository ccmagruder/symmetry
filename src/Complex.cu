// Copyright 2023 Caleb Magruder

#include <cuComplex.h>
#include <stdio.h>

#include "Complex.hpp"

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

template<>
Complex<gpuDoubleComplex>::Complex(size_t N) {
    this->_ptr = 0;
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
}
