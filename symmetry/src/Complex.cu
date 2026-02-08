// Copyright 2023 Caleb Magruder

#include <cuComplex.h>

#include "Complex.hpp"
#include "cublas_v2.h"

template<>
void Complex<gpuDouble>::_dmalloc() {
    this->_handle = new CublasHandleSingleton;
    cudaMalloc(&this->_dptr, 2*this->_N*sizeof(Type));
}

template<>
void Complex<gpuDouble>::_dfree() {
    if (this->_dptr) cudaFree(this->_dptr);
    this->_dptr = nullptr;
    delete reinterpret_cast<CublasHandleSingleton*>(this->_handle);
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

