// Copyright 2023 Caleb Magruder

#pragma once

#include <cassert>


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
    explicit Complex(size_t N) : _N(N) {
        this->_ptr = reinterpret_cast<void*>(::operator new(2*N*sizeof(Type)));
    }
    ~Complex() { if (_ptr) ::operator delete(_ptr); }

    Complex<T>& operator=(const Complex<T>& other) {
        assert(this->_N == other._N);
        Type* ptr
            = reinterpret_cast<Type*>(this->_ptr);
        Type* other_ptr
            = reinterpret_cast<Type*>(other._ptr);
        for (ptrdiff_t i = 0; i < 2 * this->_N; i++) {
            *ptr++ = *other_ptr++;
        }
        return *this;
    }

    Complex<T>& operator=(std::initializer_list<Type> l) {
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        for (auto i = l.begin(); i < l.end(); i++) {
            *ptr = *i;
            ptr++;
        }
        return *this;
    }

    bool operator==(std::initializer_list<Type> l) const {
        Type* ptr = reinterpret_cast<Type*>(this->_ptr);
        for (auto i = l.begin(); i < l.end(); i++) {
            if (*ptr != *i) {
                return false;
            }
            ptr++;
        }
        return true;
    }

    T& operator[](ptrdiff_t i) {
        return reinterpret_cast<T*>(this->_ptr)[i];
    }

    const T& operator[](ptrdiff_t i) const {
        return reinterpret_cast<T*>(this->_ptr)[i];
    }

    friend Complex<T> operator+(const Complex<T>& x, const Complex<T>& y) {
        assert(x._N == y._N);
        Complex<T> z(x._N);
        Type* xptr = reinterpret_cast<Type*>(x._ptr);
        Type* yptr = reinterpret_cast<Type*>(y._ptr);
        Type* zptr = reinterpret_cast<Type*>(z._ptr);
        for (ptrdiff_t i = 0; i < 2 * x._N; i++) {
            *zptr++ = *xptr++ + *yptr++;
        }
        return z;
    }

    friend Complex<T> operator*(const Complex<T>& x, const Complex<T>& y) {
        assert(x._N == y._N);
        Complex<T> z(x._N);
        ComplexType* xptr = reinterpret_cast<ComplexType*>(x._ptr);
        ComplexType* yptr = reinterpret_cast<ComplexType*>(y._ptr);
        ComplexType* zptr = reinterpret_cast<ComplexType*>(z._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *zptr++ = *xptr++ * *yptr++;
        }
        return z;
    }

    friend Complex<T> operator*(const Type a[2], const Complex<T>& x) {
        Complex<T> y(x._N);
        const ComplexType* const aptr = reinterpret_cast<const ComplexType*>(a);
        const ComplexType* xptr = reinterpret_cast<const ComplexType*>(x._ptr);
        ComplexType* yptr = reinterpret_cast<ComplexType*>(y._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *yptr++ = *aptr * *xptr++;
        }
        return y;
    }

    friend Complex<T> abs(const Complex<T>& x) {
        Complex<T> y(x._N);
        ComplexType* xptr = reinterpret_cast<std::complex<Type>*>(x._ptr);
        Type* yptr = reinterpret_cast<Type*>(y._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *yptr++ = abs(*xptr++);
            *yptr++ = 0;
        }
        return y;
    }

    friend Complex<T> arg(const Complex<T>& x) {
        Complex<T> y(x._N);
        ComplexType* xptr = reinterpret_cast<ComplexType*>(x._ptr);
        Type* yptr = reinterpret_cast<Type*>(y._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *yptr++ = arg(*xptr++);
            *yptr++ = 0;
        }
        return y;
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
    size_t _N;
    void* _ptr;
};

template<> Complex<gpuDouble>::Complex(size_t N);
template<> Complex<gpuDouble>::~Complex();

