// Copyright 2023 Caleb Magruder
//
// GPU-accelerated template specializations for Complex<gpuDouble>.
//
// This file contains CUDA implementations of memory management and data
// transfer operations for Complex arrays using GPU memory.

#include <cuComplex.h>

#include "Complex.hpp"
#include "cublas_v2.h"

// Allocates GPU memory for the complex array.
//
// Creates a cuBLAS handle singleton and allocates device memory for 2*N
// double values (N complex numbers stored as real/imaginary pairs).
template<>
void Complex<gpuDouble>::_dmalloc() {
    this->_handle = new CublasHandleSingleton;
    cudaMalloc(&this->_dptr, 2*this->_N*sizeof(Type));
}

// Frees GPU memory and releases the cuBLAS handle.
//
// Deallocates device memory and destroys the cuBLAS handle singleton.
// Sets _dptr to nullptr after freeing.
template<>
void Complex<gpuDouble>::_dfree() {
    if (this->_dptr) cudaFree(this->_dptr);
    this->_dptr = nullptr;
    delete reinterpret_cast<CublasHandleSingleton*>(this->_handle);
}

// Copies array data from host memory to device memory.
//
// Transfers 2*N double values from the host pointer to the device pointer.
template<>
void Complex<gpuDouble>::_memcpyHostToDevice() const {
    cudaMemcpy(this->_dptr,
               this->_ptr,
               2 * this->_N * sizeof(Type),
               cudaMemcpyHostToDevice);
}

// Copies array data from device memory to host memory.
//
// Transfers 2*N double values from the device pointer to the host pointer.
template<>
void Complex<gpuDouble>::_memcpyDeviceToHost() const {
    cudaMemcpy(this->_ptr,
               this->_dptr,
               2 * this->_N * sizeof(Type),
               cudaMemcpyDeviceToHost);
}
