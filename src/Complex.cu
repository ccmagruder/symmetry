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

class cublasHandle {
 public:
    cublasHandle() {
        cublasStatus_t status = cublasCreate(&this->_handle);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }

    ~cublasHandle() {
        cublasDestroy(this->_handle);
    }

    operator cublasHandle_t() { return this->_handle; }
 private:
    cublasHandle_t _handle;
};

template<>
void Complex<gpuDouble>::_dmalloc() {
    this->_handle = new cublasHandle;
    cudaMalloc(&this->_dptr, 2*this->_N*sizeof(Type));
    // helloCUDA<<<1, 1>>>();
    // cudaDeviceSynchronize();
}

template<>
void Complex<gpuDouble>::_dfree() {
    if (this->_dptr) cudaFree(this->_dptr);
    this->_dptr = nullptr;
    delete reinterpret_cast<cublasHandle*>(this->_handle);
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
    static const double alpha = 1.0;
    cublasDaxpy(
        *reinterpret_cast<cublasHandle*>(this->_handle),  // handle
        2*this->_N,                                       // n
        &alpha,                                           // alpha
        reinterpret_cast<double*>(other._dptr),           // x
        1,                                                // incx
        reinterpret_cast<double*>(this->_dptr),            // y
        1);                                               // incy
    // cudaDeviceSynchronize();
    this->_memcpyDeviceToHost();
    return *this;
}