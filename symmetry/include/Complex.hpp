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

// Tag type for GPU-accelerated double precision operations.
//
// Used as a template parameter to select GPU-accelerated specializations
// of Complex operations.
class gpuDouble {};

// Traits class to extract the underlying value type from a Complex type.
//
// For integral types, the value_type is the type itself.
// For gpuDouble, the value_type is double.
template <typename T>
struct complex_traits {
    typedef T value_type;
};

template<>
struct complex_traits<gpuDouble> {
    typedef double value_type;
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
    using Type = typename complex_traits<T>::value_type;
    using ComplexType = std::complex<Type>;

 public:
    // Constructs a Complex array with N complex numbers.
    //
    // Args:
    //   N: The number of complex numbers to store.
    explicit Complex(size_t N) : _N(N), _dptr(nullptr) {
        this->_ptr = ::operator new(2*N*sizeof(Type));
        this->_dmalloc();
    }

    // Copy constructor. Deep copies the data from other.
    //
    // Args:
    //   other: The Complex array to copy from.
    Complex(const Complex<T>& other) : Complex(other._N) {
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        Type* other_ptr = reinterpret_cast<Type*>(other._ptr);
        memcpy(ptr, other_ptr, 2 * this->_N * sizeof(Type));
    }

    explicit Complex(const Complex<T>&&) = delete;

    // Constructs a Complex array from an initializer list.
    //
    // Args:
    //   l: Initializer list of real values as {re1, im1, re2, im2, ...}.
    Complex(std::initializer_list<Type> l) : Complex(l.size()/2) {
        using Iter = typename std::initializer_list<Type>::const_iterator;
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        for (Iter i = l.begin(); i < l.end(); i++) {
            *ptr++ = *i;
        }
    }

    ~Complex() {
        if (_ptr) ::operator delete(_ptr);
        this->_dfree();
        assert(this->_dptr == nullptr);
    }

    Complex& operator=(const Complex<T>&) = delete;
    Complex& operator=(const Complex<T>&&) = delete;

    // Compares the array against an initializer list.
    //
    // Args:
    //   l: Initializer list of values to compare.
    //
    // Returns:
    //   True if all values match exactly, false otherwise.
    bool operator==(std::initializer_list<Type> l) const {
        using Iter = typename std::initializer_list<Type>::const_iterator;
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
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
    const ComplexType& operator[](ptrdiff_t i) const {
        return *(reinterpret_cast<const ComplexType*>(this->_ptr) + i);
    }

    // Element-wise addition. Adds other to this array in place.
    //
    // Args:
    //   other: The Complex array to add.
    //
    // Returns:
    //   Reference to this array after addition.
    Complex<T>& operator+=(const Complex<T>& other) {
        Type* other_ptr = reinterpret_cast<Type*>(other._ptr);
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
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
        const ComplexType* other_ptr
            = reinterpret_cast<const ComplexType*>(other._ptr);
        ComplexType* ptr = reinterpret_cast<ComplexType*>(this->_ptr);
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
    Complex<T>& operator*=(const ComplexType& a) {
        ComplexType* ptr = reinterpret_cast<ComplexType*>(this->_ptr);
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
        ComplexType* cptr = reinterpret_cast<ComplexType*>(this->_ptr);
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
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
        ComplexType* cptr = reinterpret_cast<ComplexType*>(this->_ptr);
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
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
        ComplexType* ptr = reinterpret_cast<ComplexType*>(this->_ptr);
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
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ = std::cos(*ptr);
            ptr++;
        }
        return *this;
    }

    // Element-wise addition of two Complex arrays.
    friend Complex<T> operator+(const Complex<T>& x, const Complex<T>& y) {
        assert(x._N == y._N);
        return Complex<T>(x) += y;
    }

    // Element-wise multiplication of two Complex arrays.
    friend Complex<T> operator*(const Complex<T>& x, const Complex<T>& y) {
        assert(x._N == y._N);
        return Complex<T>(x) *= y;
    }

    // Scalar multiplication of a Complex array.
    friend Complex<T> operator*(const ComplexType& a, const Complex<T>& x) {
        return Complex<T>(x) *= a;
    }

    // Returns a new Complex array with element-wise absolute values.
    friend Complex<T> abs(const Complex<T>& x) {
        return Complex<T>(x).abs();
    }

    // Returns a new Complex array with element-wise arguments.
    friend Complex<T> arg(const Complex<T>& x) {
        return Complex<T>(x).arg();
    }

    // Returns a new Complex array with element-wise conjugates.
    friend Complex<T> conj(const Complex<T>& x) {
        return Complex<T>(x).conj();
    }

    // Returns a new Complex array with element-wise cosines.
    friend Complex<T> cos(const Complex<T>& x) {
        return Complex<T>(x).cos();
    }

    // Outputs the Complex array to a stream.
    friend std::ostream& operator<<(std::ostream& os, const Complex<T>& c) {
        os << "Complex<T>{";
        Type* ptr = reinterpret_cast<Type*>(c._ptr);
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
    void _memcpyDeviceToHost() {}

    size_t _N;
    void* _ptr;     // Host pointer
    void* _dptr;    // Device pointer
    void* _handle;  // Device context
 
    enum class Memory { Device, Host };  // Current memory location

    volatile Memory _memory;
};

// GPU-accelerated specializations for Complex<gpuDouble>.
template<> void Complex<gpuDouble>::_dmalloc();
template<> void Complex<gpuDouble>::_dfree();
template<> void Complex<gpuDouble>::_memcpyHostToDevice() const;
template<> void Complex<gpuDouble>::_memcpyDeviceToHost();
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator+=(const Complex<gpuDouble>&);
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator*=(const Complex<gpuDouble>&);
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator*=(const std::complex<double>&);
template<> Complex<gpuDouble>& Complex<gpuDouble>::abs();
template<> Complex<gpuDouble>& Complex<gpuDouble>::arg();
template<> Complex<gpuDouble>& Complex<gpuDouble>::conj();
template<> Complex<gpuDouble>& Complex<gpuDouble>::cos();
