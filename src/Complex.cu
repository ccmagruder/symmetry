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
Complex<gpuDouble>::Complex(size_t N) : _N(N) {
    this->_ptr = reinterpret_cast<void*>(::operator new(2*N*sizeof(Type)));
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
}

template<>
Complex<gpuDouble>::~Complex() {
    if (_ptr) ::operator delete(_ptr);
}
