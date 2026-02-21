// Copyright 2023 Caleb Magruder
//
// GPU-accelerated template specializations for Complex<gpuDouble>.
//
// This file contains CUDA implementations of memory management and data
// transfer operations for Complex arrays using GPU memory.

#include <cuComplex.h>

#include "Complex.hpp"

// Allocates GPU memory for the complex array.
//
// Creates a cuBLAS handle singleton and allocates device memory for 2*N
// double values (N complex numbers stored as real/imaginary pairs).
template<>
void Complex<gpuDouble>::_dmalloc() {
    this->_handle = new CublasHandleSingleton;
    cudaMalloc(&this->_dptr, 2*this->_N*sizeof(Scalar));
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
               2 * this->_N * sizeof(Scalar),
               cudaMemcpyHostToDevice);
}

// Copies array data from device memory to host memory.
//
// Transfers 2*N double values from the device pointer to the host pointer.
template<>
void Complex<gpuDouble>::_memcpyDeviceToHost() const {
    cudaMemcpy(this->_ptr,
               this->_dptr,
               2 * this->_N * sizeof(Scalar),
               cudaMemcpyDeviceToHost);
}

// Computes element-wise absolute value (magnitude) in place.
//
// Uses cuCabs for GPU-compatible complex magnitude computation.
// Stores the magnitude in the real part, sets imaginary part to zero.
//
// Returns:
//   Reference to this array after the operation.
__global__ void gpuDoubleAbs(cuDoubleComplex* data, int n) {
    double* ptr = reinterpret_cast<double*>(data);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ptr[2*i] = cuCabs(data[i]);
        ptr[2*i+1] = 0;
    }
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::abs() {
    Type* ptr = reinterpret_cast<Type*>(this->_dptr);
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    gpuDoubleAbs<<<blocks, threads>>>(ptr, this->_N);
    return *this;
}

// Computes element-wise argument (phase angle) in place.
//
// Computes atan2(imag, real) for each complex number.
// Stores the argument in the real part, sets imaginary part to zero.
//
// Returns:
//   Reference to this array after the operation.
__global__ void gpuDoubleArg(cuDoubleComplex* data, int n) {
    double* ptr = reinterpret_cast<double*>(data);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ptr[2*i] = atan2(ptr[2*i+1], ptr[2*i]);
        ptr[2*i+1] = 0;
    }
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::arg() {
    Type* ptr = reinterpret_cast<Type*>(this->_dptr);
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    gpuDoubleArg<<<blocks, threads>>>(ptr, this->_N);
    return *this;
}

// Computes element-wise complex conjugate in place.
//
// Uses cuConj for GPU-compatible complex conjugation.
// Negates the imaginary part of each complex number.
//
// Returns:
//   Reference to this array after the operation.
__global__ void gpuDoubleConj(cuDoubleComplex* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = cuConj(data[i]);
    }
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::conj() {
    Type* ptr = reinterpret_cast<Type*>(this->_dptr);
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    gpuDoubleConj<<<blocks, threads>>>(ptr, this->_N);
    return *this;
}

// Computes element-wise cosine of the real part in place.
//
// Applies std::cos to the real part of each complex number.
// The imaginary part is not modified.
//
// Returns:
//   Reference to this array after the operation.
__global__ void gpuDoubleCos(cuDoubleComplex* data, int n) {
    double* ptr = reinterpret_cast<double*>(data);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        ptr[2*i] = cos(ptr[2*i]);
    }
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::cos() {
    Type* ptr = reinterpret_cast<Type*>(this->_dptr);
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    gpuDoubleCos<<<blocks, threads>>>(ptr, this->_N);
    return *this;
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator=(const Complex<gpuDouble>& other) {
    cudaMemcpy(this->_dptr, other._dptr,
               2 * this->_N * sizeof(Scalar),
               cudaMemcpyDeviceToDevice);
    return *this;
}

__global__ void gpuDoubleFill(cuDoubleComplex* data, int n, double re, double im) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = make_cuDoubleComplex(re, im);
    }
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::fill(Scalar re, Scalar im) {
    Type* ptr = reinterpret_cast<Type*>(this->_dptr);
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    gpuDoubleFill<<<blocks, threads>>>(ptr, this->_N, re, im);
    return *this;
}

__global__ void gpuDoubleZeroImag(double* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[2*i+1] = 0.0;
    }
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::zero_imag() {
    double* ptr = reinterpret_cast<double*>(this->_dptr);
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    gpuDoubleZeroImag<<<blocks, threads>>>(ptr, this->_N);
    return *this;
}
