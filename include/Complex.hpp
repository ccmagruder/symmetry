// Copyright 2023 Caleb Magruder

#pragma once

#include <cassert>
#include <complex>

class gpuDouble {};

template <typename T>
struct complex_traits {
    typedef T value_type;  // Integral type case
};

template<>
struct complex_traits<gpuDouble> {
    typedef double value_type;
};

template<typename T>
class Complex{
    using Type = typename complex_traits<T>::value_type;
    using ComplexType = std::complex<Type>;

 public:
    explicit Complex(size_t N) : _N(N), _dptr(nullptr) {
        this->_ptr = ::operator new(2*N*sizeof(Type));
        this->_dmalloc();
    }

    Complex(const Complex<T>& other) : Complex(other._N) {
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        Type* other_ptr = reinterpret_cast<Type*>(other._ptr);
        memcpy(ptr, other_ptr, 2 * this->_N * sizeof(Type));
    }

    explicit Complex(const Complex<T>&&) = delete;

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

    const ComplexType& operator[](ptrdiff_t i) const {
        return *(reinterpret_cast<const ComplexType*>(this->_ptr) + i);
    }

    Complex<T>& operator+=(const Complex<T>& other) {
        Type* other_ptr = reinterpret_cast<Type*>(other._ptr);
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        for (ptrdiff_t i = 0; i < 2 * this->_N; i++) {
            *ptr++ += *other_ptr++;
        }
        return *this;
    }

    Complex<T>& operator*=(const Complex<T>& other) {
        const ComplexType* other_ptr
            = reinterpret_cast<const ComplexType*>(other._ptr);
        ComplexType* ptr = reinterpret_cast<ComplexType*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ *= *other_ptr++;
        }
        return *this;
    }

    Complex<T>& operator*=(const ComplexType& a) {
        ComplexType* ptr = reinterpret_cast<ComplexType*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ *= a;
        }
        return *this;
    }

    Complex<T>& abs() {
        ComplexType* cptr = reinterpret_cast<ComplexType*>(this->_ptr);
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ = std::abs(*cptr++);
            *ptr++ = 0;
        }
        return *this;
    }

    Complex<T>& arg() {
        ComplexType* cptr = reinterpret_cast<ComplexType*>(this->_ptr);
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ = std::arg(*cptr++);
            *ptr++ = 0;
        }
        return *this;
    }

    friend Complex<T> operator+(const Complex<T>& x, const Complex<T>& y) {
        assert(x._N == y._N);
        return Complex<T>(x) += y;
    }

    friend Complex<T> operator*(const Complex<T>& x, const Complex<T>& y) {
        assert(x._N == y._N);
        return Complex<T>(x) *= y;
    }

    friend Complex<T> operator*(const ComplexType& a, const Complex<T>& x) {
        return Complex<T>(x) *= a;
    }

    friend Complex<T> abs(const Complex<T>& x) {
        return Complex<T>(x).abs();
    }

    friend Complex<T> arg(const Complex<T>& x) {
        return Complex<T>(x).arg();
    }

    friend Complex<T> conj(const Complex<T>& x) {
        Complex<T> y(x._N);
        ComplexType* xptr = reinterpret_cast<ComplexType*>(x._ptr);
        ComplexType* yptr = reinterpret_cast<ComplexType*>(y._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *yptr++ = conj(*xptr++);
        }
        return y;
    }

    friend Complex<T> cos(const Complex<T>& x) {
        Complex<T> y(x._N);
        Type* xptr = reinterpret_cast<Type*>(x._ptr);
        Type* yptr = reinterpret_cast<Type*>(y._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *yptr++ = cos(*xptr++);
            *yptr++ = *xptr++;
        }
        return y;
    }

 private:
    void _dmalloc() {}
    void _dfree() {}
    void _memcpyHostToDevice() const {}
    void _memcpyDeviceToHost() {}

    size_t _N;
    void* _ptr;     // Host pointer
    void* _dptr;    // Device pointer
    void* _handle;  // Device context
};

template<> void Complex<gpuDouble>::_dmalloc();
template<> void Complex<gpuDouble>::_dfree();
template<> void Complex<gpuDouble>::_memcpyHostToDevice() const;
template<> void Complex<gpuDouble>::_memcpyDeviceToHost();
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator+=(const Complex<gpuDouble>&);
template<>
Complex<gpuDouble>& Complex<gpuDouble>::operator*=(const Complex<gpuDouble>&);
