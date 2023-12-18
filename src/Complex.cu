// Copyright 2023 Caleb Magruder

#include <cuComplex.h>
#include <stdio.h>
#include <complex>

#include "Complex.hpp"
#include "cublas_v2.h"

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

template<>
void Complex<gpuDouble>::_dmalloc() {
    cudaMalloc(&this->_dptr, 2*this->_N*sizeof(Type));
    // helloCUDA<<<1, 1>>>();
    // cudaDeviceSynchronize();
}

template<>
void Complex<gpuDouble>::_dfree() {
    if (this->_dptr) cudaFree(this->_dptr);
    this->_dptr = nullptr;
}

template<>
void Complex<gpuDouble>::_memcpyHostToDevice() const {
    cudaMemcpy(this->_dptr,
               this->_ptr,
               2 * this->_N * sizeof(Type),
               cudaMemcpyHostToDevice);
}

template<>
void Complex<gpuDouble>::_memcpyDeviceToHost() {
    cudaMemcpy(this->_ptr,
               this->_dptr,
               2 * this->_N * sizeof(Type),
               cudaMemcpyDeviceToHost);
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator+=(const Complex<gpuDouble>& other) {
    this->_memcpyHostToDevice();
    other._memcpyHostToDevice();
    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha(1);
    cublasDaxpy(handle, 2*this->_N, &alpha, (double*)other._dptr, 1, (double*)this->_dptr, 1);
    cublasDestroy(handle);
    this->_memcpyDeviceToHost();
    return *this;
}