// Copyright 2023 Caleb Magruder

#include <cuComplex.h>
#include <stdio.h>
#include <complex>

#include "Complex.hpp"

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

template<>
Complex<gpuDoubleComplex>::Complex(size_t N) {
    this->_ptr = reinterpret_cast<void*>(new std::complex<double>[N]);
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
}

template<>
Complex<gpuDoubleComplex>::~Complex() {
    if (_ptr) delete [] reinterpret_cast<std::complex<double>*>(_ptr);
}
