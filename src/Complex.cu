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
    this->_ptr = ::operator new(2*N*sizeof(Type));
    cudaMalloc(&this->_dptr, 2*N*sizeof(Type));
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
}

template<>
Complex<gpuDouble>::Complex(std::initializer_list<double> l) : Complex(l.size()/2) {
    using Iter = typename std::initializer_list<Type>::const_iterator;
    Type* ptr = reinterpret_cast<Type*>(this->_ptr);
    for (Iter i = l.begin(); i < l.end(); i++) {
        *ptr++ = *i;
    }
    cudaMemcpy(this->_dptr,
               this->_ptr,
               2 * this->_N * sizeof(Type),
               cudaMemcpyHostToDevice);
}

template<>
Complex<gpuDouble>::~Complex() {
    if (this->_ptr) ::operator delete(this->_ptr);
    if (this->_dptr) cudaFree(this->_dptr);
}
