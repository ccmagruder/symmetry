// Copyright 2023 Caleb Magruder
//
// cuBLAS handle management and GPU-accelerated Complex operations.
//
// This file contains the CublasHandleSingleton implementation and
// GPU-accelerated template specializations for arithmetic and mathematical
// operations on Complex<gpuDouble> arrays.

#include "cublas_v2.h"
#include <cassert>
#include "Complex.hpp"

// Static member definitions for CublasHandleSingleton.
cublasHandle_t CublasHandleSingleton::_handle;
int CublasHandleSingleton::_count = 0;

// Constructs or increments reference to the shared cuBLAS handle.
//
// On first construction, creates the cuBLAS handle. Subsequent constructions
// increment the reference count without creating a new handle.
CublasHandleSingleton::CublasHandleSingleton() {
    if (CublasHandleSingleton::_count++ == 0) {
        cublasStatus_t status = cublasCreate(&CublasHandleSingleton::_handle);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }
}

// Decrements reference count and destroys the cuBLAS handle when zero.
//
// When the last reference is released, the cuBLAS handle is destroyed.
CublasHandleSingleton::~CublasHandleSingleton() {
    if (--CublasHandleSingleton::_count == 0) {
        cublasDestroy(CublasHandleSingleton::_handle);
    }
}

// Returns the underlying cuBLAS handle for use with cuBLAS API calls.
CublasHandleSingleton::operator cublasHandle_t() { return this->_handle; }

// GPU-accelerated element-wise addition using cuBLAS.
//
// Copies both arrays to GPU, performs y = y + x using cublasDaxpy,
// and copies the result back to host memory.
//
// Args:
//   other: The Complex array to add to this array.
//
// Returns:
//   Reference to this array after addition.
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator+=(const Complex<gpuDouble>& other) {
    static constexpr double alpha = 1.0;
    cublasDaxpy(
        *reinterpret_cast<CublasHandleSingleton*>(this->_handle),  // handle
        2*this->_N,                                                // n
        &alpha,                                                    // alpha
        reinterpret_cast<double*>(other._dptr),                    // x
        1,                                                         // incx
        reinterpret_cast<double*>(this->_dptr),                    // y
        1);                                                        // incy
    return *this;
}

// GPU-accelerated element-wise multiplication using cuBLAS.
//
// Copies both arrays to GPU, performs element-wise complex multiplication
// using cublasZdgmm, and copies the result back to host memory.
//
// Args:
//   other: The Complex array to multiply element-wise with this array.
//
// Returns:
//   Reference to this array after multiplication.
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator*=(const Complex<gpuDouble>& other) {
    cublasZdgmm(
        *reinterpret_cast<CublasHandleSingleton*>(this->_handle),  // handle
        CUBLAS_SIDE_LEFT,                                          // mode
        this->_N,                                                  // m
        1,                                                         // n
        reinterpret_cast<cuDoubleComplex*>(other._dptr),           // A
        this->_N,                                                  // lda
        reinterpret_cast<cuDoubleComplex*>(this->_dptr),           // x
        1,                                                         // incx
        reinterpret_cast<cuDoubleComplex*>(this->_dptr),           // C
        this->_N);                                                 // ldc
    return *this;
}

// GPU-accelerated scalar multiplication using cuBLAS.
//
// Copies array to GPU, scales all elements by the complex scalar a
// using cublasZscal, and copies the result back to host memory.
//
// Args:
//   a: The complex scalar to multiply all elements by.
//
// Returns:
//   Reference to this array after scaling.
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator*=(const std::complex<double>& a) {
    this->_memcpyHostToDevice();
    cublasZscal(
        *reinterpret_cast<CublasHandleSingleton*>(this->_handle),     // handle
        this->_N,                                                     // n
        reinterpret_cast<const cuDoubleComplex*>(&a),                 // alpha
        reinterpret_cast<cuDoubleComplex*>(this->_dptr),              // x
        1);                                                           // incx
    this->_memcpyDeviceToHost();
    return *this;
}

// Computes element-wise absolute value (magnitude) in place.
//
// Uses cuCabs for GPU-compatible complex magnitude computation.
// Stores the magnitude in the real part, sets imaginary part to zero.
//
// Returns:
//   Reference to this array after the operation.
template<>
Complex<gpuDouble>& Complex<gpuDouble>::abs() {
    this->_memcpyDeviceToHost();
    const cuDoubleComplex* cptr = reinterpret_cast<const cuDoubleComplex*>(this->_ptr);
    double* ptr = reinterpret_cast<double*>(this->_ptr);
    for (ptrdiff_t i = 0; i < this->_N; i++) {
        *ptr++ = cuCabs(*cptr++);
        *ptr++ = 0;
    }
    this->_memcpyHostToDevice();
    return *this;
}

// Computes element-wise argument (phase angle) in place.
//
// Computes atan2(imag, real) for each complex number.
// Stores the argument in the real part, sets imaginary part to zero.
//
// Returns:
//   Reference to this array after the operation.
template<>
Complex<gpuDouble>& Complex<gpuDouble>::arg() {
    this->_memcpyDeviceToHost();
    double* ptr = reinterpret_cast<double*>(this->_ptr);
    for (ptrdiff_t i = 0; i < this->_N; i++) {
        *ptr++ = atan2(ptr[1], ptr[0]);
        *ptr++ = 0;
    }
    this->_memcpyHostToDevice();
    return *this;
}

// Computes element-wise complex conjugate in place.
//
// Uses cuConj for GPU-compatible complex conjugation.
// Negates the imaginary part of each complex number.
//
// Returns:
//   Reference to this array after the operation.
template<>
Complex<gpuDouble>& Complex<gpuDouble>::conj() {
    this->_memcpyDeviceToHost();
    cuDoubleComplex* cptr = reinterpret_cast<cuDoubleComplex*>(this->_ptr);
    for (ptrdiff_t i = 0; i < this->_N; i++) {
        *cptr++ = cuConj(*cptr);
    }
    this->_memcpyHostToDevice();
    return *this;
}

// Computes element-wise cosine of the real part in place.
//
// Applies std::cos to the real part of each complex number.
// The imaginary part is not modified.
//
// Returns:
//   Reference to this array after the operation.
template<>
Complex<gpuDouble>& Complex<gpuDouble>::cos() {
    this->_memcpyDeviceToHost();
    double* ptr = reinterpret_cast<double*>(this->_ptr);
    for (ptrdiff_t i = 0; i < this->_N; i++) {
        *ptr++ = std::cos(*ptr);
        ptr++;
    }
    this->_memcpyHostToDevice();
    return *this;
}
