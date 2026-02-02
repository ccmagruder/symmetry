// Copyright 2023 Caleb Magruder

#include <cuComplex.h>
#include <stdio.h>

#include "Complex.hpp"
#include "cublas_v2.h"

class CublasHandleSingleton {
 public:
    CublasHandleSingleton() {
        if (CublasHandleSingleton::_count++ == 0) {
            cublasStatus_t status = cublasCreate(&CublasHandleSingleton::_handle);
            assert(status == CUBLAS_STATUS_SUCCESS);
        }
    }

    ~CublasHandleSingleton() {
        if (--CublasHandleSingleton::_count == 0) {
            cublasDestroy(CublasHandleSingleton::_handle);
        }
    }

    operator cublasHandle_t() { return this->_handle; }
 private:
    static cublasHandle_t _handle;
    static int _count;
};

cublasHandle_t CublasHandleSingleton::_handle;
int CublasHandleSingleton::_count = 0;

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

template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator+=(const Complex<gpuDouble>& other) {
    this->_memcpyHostToDevice();
    other._memcpyHostToDevice();
    static constexpr double alpha = 1.0;
    cublasDaxpy(
        *reinterpret_cast<CublasHandleSingleton*>(this->_handle),  // handle
        2*this->_N,                                                // n
        &alpha,                                                    // alpha
        reinterpret_cast<double*>(other._dptr),                    // x
        1,                                                         // incx
        reinterpret_cast<double*>(this->_dptr),                    // y
        1);                                                        // incy
    this->_memcpyDeviceToHost();
    return *this;
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator*=(const Complex<gpuDouble>& other) {
    this->_memcpyHostToDevice();
    other._memcpyHostToDevice();
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
    this->_memcpyDeviceToHost();
    return *this;
}

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

template<>
Complex<gpuDouble>& Complex<gpuDouble>::abs() {
    const cuDoubleComplex* cptr = reinterpret_cast<const cuDoubleComplex*>(this->_ptr);
    double* ptr = reinterpret_cast<double*>(this->_ptr);
    for (ptrdiff_t i = 0; i < this->_N; i++) {
        *ptr++ = cuCabs(*cptr++);
        *ptr++ = 0;
    }
    return *this;
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::arg() {
    double* ptr = reinterpret_cast<double*>(this->_ptr);
    for (ptrdiff_t i = 0; i < this->_N; i++) {
        *ptr++ = atan2(ptr[1], ptr[0]);
        *ptr++ = 0;
    }
    return *this;
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::conj() {
    cuDoubleComplex* cptr = reinterpret_cast<cuDoubleComplex*>(this->_ptr);
    for (ptrdiff_t i = 0; i < this->_N; i++) {
        *cptr++ = cuConj(*cptr);
    }
    return *this;
}

template<>
Complex<gpuDouble>& Complex<gpuDouble>::cos() {
    double* ptr = reinterpret_cast<double*>(this->_ptr);
    for (ptrdiff_t i = 0; i < this->_N; i++) {
        *ptr++ = std::cos(*ptr);
        *ptr++;
    }
    return *this;
}
