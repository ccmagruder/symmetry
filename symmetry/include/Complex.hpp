// Copyright 2023 Caleb Magruder

#pragma once

#include <cassert>
#include <complex>
#include <cstring>
#include <iostream>

#include "cublas_v2.h"

// Singleton manager for cuBLAS handle.
//
// Ensures a single cuBLAS handle is created and shared across all Complex
// instances. Reference counted to destroy the handle when the last instance
// is destroyed.
class CublasHandleSingleton {
 public:
    CublasHandleSingleton();

    ~CublasHandleSingleton();

    // Implicit conversion to cublasHandle_t for use with cuBLAS API.
    operator cublasHandle_t();
 private:
    static cublasHandle_t _handle;
    static int _count;
};

// Tag types for CPU and GPU double precision operations.
//
// Used as template parameters to select CPU or GPU-accelerated specializations
// of Complex and FPI operations.
class cpuDouble {
 public:
    using Scalar = double;
    using Type = std::complex<double>;
};
class gpuDouble {
 public:
    using Scalar = double;
    using Type = cuDoubleComplex;
};

// A container for an array of complex numbers.
//
// Stores N complex numbers as 2*N contiguous real values (real, imag pairs).
// When instantiated with gpuDouble, operations are GPU-accelerated via cuBLAS.
//
// Template Args:
//   T: The value type. Use gpuDouble for GPU acceleration, or a numeric type
//      (int, float, double) for CPU operations.
template<typename T>
class Complex{
    using Scalar = typename T::Scalar;
    using Type = typename T::Type;

 public:
    // Constructs a Complex array with N complex numbers.
    //
    // Args:
    //   N: The number of complex numbers to store.
    explicit Complex(size_t N) : _N(N), _dptr(nullptr) {
        this->_ptr = ::operator new(2*N*sizeof(Scalar));
        this->_dmalloc();
    }

    explicit Complex(const Complex<T>&) = delete;
    explicit Complex(const Complex<T>&&) = delete;

    // Constructs a Complex array from an initializer list.
    //
    // Args:
    //   l: Initializer list of real values as {re1, im1, re2, im2, ...}.
    Complex(std::initializer_list<Scalar> l) : Complex(l.size()/2) {
        using Iter = typename std::initializer_list<Scalar>::const_iterator;
        Scalar* ptr = reinterpret_cast<Scalar*>(this->_ptr);
        for (Iter i = l.begin(); i < l.end(); i++) {
            *ptr++ = *i;
        }
        this->_memcpyHostToDevice();
    }

    ~Complex() {
        if (_ptr) ::operator delete(_ptr);
        this->_dfree();
        assert(this->_dptr == nullptr);
    }

    // Copies data from another Complex array of the same size.
    Complex<T>& operator=(const Complex<T>& other) {
        assert(this->_N == other._N);
        memcpy(this->_ptr, other._ptr, 2 * _N * sizeof(Scalar));
        return *this;
    }

    Complex& operator=(const Complex<T>&&) = delete;

    // Compares the array against an initializer list.
    //
    // Args:
    //   l: Initializer list of values to compare.
    //
    // Returns:
    //   True if all values match exactly, false otherwise.
    bool operator==(std::initializer_list<Scalar> l) const {
        using Iter = typename std::initializer_list<Scalar>::const_iterator;
        this->_memcpyDeviceToHost();
        const Scalar* ptr = reinterpret_cast<const Scalar*>(this->_ptr);
        for (Iter i = l.begin(); i < l.end(); i++) {
            if (*ptr != *i) {
                return false;
            }
            ptr++;
        }
        return true;
    }

    // Accesses the i-th complex number.
    //
    // Args:
    //   i: Index of the complex number.
    //
    // Returns:
    //   Reference to the i-th complex number.
    const Type& operator[](ptrdiff_t i) const {
        this->_memcpyDeviceToHost();
        return *(reinterpret_cast<const Type*>(this->_ptr) + i);
    }

    // Element-wise addition. Adds other to this array in place.
    //
    // Args:
    //   other: The Complex array to add.
    //
    // Returns:
    //   Reference to this array after addition.
    Complex<T>& operator+=(const Complex<T>& other) {
        Scalar* other_ptr = reinterpret_cast<Scalar*>(other._ptr);
        Scalar* ptr = reinterpret_cast<Scalar*>(this->_ptr);
        for (ptrdiff_t i = 0; i < 2 * this->_N; i++) {
            *ptr++ += *other_ptr++;
        }
        return *this;
    }

    // Element-wise multiplication. Multiplies this array by other in place.
    //
    // Args:
    //   other: The Complex array to multiply by.
    //
    // Returns:
    //   Reference to this array after multiplication.
    Complex<T>& operator*=(const Complex<T>& other) {
        const Type* other_ptr
            = reinterpret_cast<const Type*>(other._ptr);
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ *= *other_ptr++;
        }
        return *this;
    }

    // Scalar multiplication. Multiplies all elements by a scalar in place.
    //
    // Args:
    //   a: The complex scalar to multiply by.
    //
    // Returns:
    //   Reference to this array after scaling.
    Complex<T>& operator*=(const Type& a) {
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ *= a;
        }
        return *this;
    }

    // Computes element-wise absolute value in place.
    //
    // Replaces each complex number with its magnitude (imaginary part set to 0).
    //
    // Returns:
    //   Reference to this array after the operation.
    Complex<T>& abs() {
        Type* cptr = reinterpret_cast<Type*>(this->_ptr);
        Scalar* ptr = reinterpret_cast<Scalar*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ = std::abs(*cptr++);
            *ptr++ = 0;
        }
        return *this;
    }

    // Computes element-wise argument (phase angle) in place.
    //
    // Replaces each complex number with its argument (imaginary part set to 0).
    //
    // Returns:
    //   Reference to this array after the operation.
    Complex<T>& arg() {
        Type* cptr = reinterpret_cast<Type*>(this->_ptr);
        Scalar* ptr = reinterpret_cast<Scalar*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ = std::arg(*cptr++);
            *ptr++ = 0;
        }
        return *this;
    }

    // Computes element-wise complex conjugate in place.
    //
    // Returns:
    //   Reference to this array after the operation.
    Complex<T>& conj() {
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ = std::conj(*ptr);
        }
        return *this;
    }

    // Computes element-wise cosine of the real part in place.
    //
    // Returns:
    //   Reference to this array after the operation.
    Complex<T>& cos() {
        Scalar* ptr = reinterpret_cast<Scalar*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ = std::cos(*ptr);
            ptr++;
        }
        return *this;
    }

    // Sets all elements to the same complex value.
    Complex<T>& fill(Scalar re, Scalar im) {
        Scalar* ptr = reinterpret_cast<Scalar*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) { *ptr++ = re; *ptr++ = im; }
        return *this;
    }

    // Zeros all imaginary parts, keeping real parts unchanged.
    Complex<T>& zero_imag() {
        Scalar* ptr = reinterpret_cast<Scalar*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) { ptr++; *ptr++ = 0; }
        return *this;
    }

    void* dptr() const { return _dptr; }
    size_t size() const { return _N; }

    // Outputs the Complex array to a stream.
    friend std::ostream& operator<<(std::ostream& os, const Complex<T>& c) {
        os << "Complex<T>{";
        Scalar* ptr = reinterpret_cast<Scalar*>(c._ptr);
        for (ptrdiff_t i = 0; i < 2 * c._N; i++) {
            os << *ptr++ << ",";
        }
        os << "}\n";
        return os;
    }

 private:
    // Allocates device memory. No-op for CPU types, allocates on GPU for gpuDouble.
    void _dmalloc() {}

    // Frees device memory. No-op for CPU types.
    void _dfree() {}

    // Copies data from host to device. No-op for CPU types.
    void _memcpyHostToDevice() const {}

    // Copies data from device to host. No-op for CPU types.
    void _memcpyDeviceToHost() const {}

    size_t _N;
    void* _ptr;     // Host pointer
    void* _dptr;    // Device pointer
    void* _handle;  // Device context
};

// GPU-accelerated specializations for Complex<gpuDouble>.
template<> void Complex<gpuDouble>::_dmalloc();
template<> void Complex<gpuDouble>::_dfree();
template<> void Complex<gpuDouble>::_memcpyHostToDevice() const;
template<> void Complex<gpuDouble>::_memcpyDeviceToHost() const;
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator+=(const Complex<gpuDouble>&);
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator*=(const Complex<gpuDouble>&);
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator*=(const gpuDouble::Type&);
template<> Complex<gpuDouble>& Complex<gpuDouble>::abs();
template<> Complex<gpuDouble>& Complex<gpuDouble>::arg();
template<> Complex<gpuDouble>& Complex<gpuDouble>::conj();
template<> Complex<gpuDouble>& Complex<gpuDouble>::cos();
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator=(const Complex<gpuDouble>&);
template<>
Complex<gpuDouble>& Complex<gpuDouble>::fill(gpuDouble::Scalar, gpuDouble::Scalar);
template<> Complex<gpuDouble>& Complex<gpuDouble>::zero_imag();
