// Copyright 2023 Caleb Magruder

#pragma once

#include <cassert>

class gpuDoubleComplex {
 public:
    typedef double value_type;
};

template<typename T>
class Complex{
 public:
    explicit Complex(size_t N) : _N(N) {
        this->_ptr = reinterpret_cast<void*>(new T[N]);
    }
    ~Complex() { if (_ptr) delete[] reinterpret_cast<T*>(_ptr); }

    Complex<T>& operator=(const Complex<T>& other) {
        assert(this->_N == other._N);
        std::complex<typename T::value_type>* ptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(
                this->_ptr);
        std::complex<typename T::value_type>* other_ptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(
                other._ptr);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            *ptr++ = *other_ptr++;
        }
        return *this;
    }

    Complex<T>& operator=(std::initializer_list<typename T::value_type> l) {
        typename T::value_type* ptr
            = reinterpret_cast<typename T::value_type*>(this->_ptr);
        for (auto i = l.begin(); i < l.end(); i++) {
            *ptr = *i;
            ptr++;
        }
        return *this;
    }

    bool operator==(std::initializer_list<typename T::value_type> l) const {
        typename T::value_type* ptr
            = reinterpret_cast<typename T::value_type*>(this->_ptr);
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
        std::complex<typename T::value_type>* xptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(x._ptr);
        std::complex<typename T::value_type>* yptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(y._ptr);
        std::complex<typename T::value_type>* zptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(z._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *zptr++ = *xptr++ + *yptr++;
        }
        return z;
    }

    friend Complex<T> operator*(const Complex<T>& x, const Complex<T>& y) {
        assert(x._N == y._N);
        Complex<T> z(x._N);
        std::complex<typename T::value_type>* xptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(x._ptr);
        std::complex<typename T::value_type>* yptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(y._ptr);
        std::complex<typename T::value_type>* zptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(z._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *zptr++ = *xptr++ * *yptr++;
        }
        return z;
    }

    friend Complex<T> operator*(const T& a, const Complex<T>& x) {
        Complex<T> y(x._N);
        const std::complex<typename T::value_type>* aptr
            = reinterpret_cast<const std::complex<typename T::value_type>*>(&a);
        std::complex<typename T::value_type>* xptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(x._ptr);
        std::complex<typename T::value_type>* yptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(y._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *yptr++ = *aptr * *xptr++;
        }
        return y;
    }

    friend Complex<T> abs(const Complex<T>& x) {
        Complex<T> y(x._N);
        std::complex<typename T::value_type>* xptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(x._ptr);
        typename T::value_type* yptr
            = reinterpret_cast<typename T::value_type*>(y._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *yptr++ = abs(*xptr++);
            *yptr++ = 0;
        }
        return y;
    }

    friend Complex<T> arg(const Complex<T>& x) {
        Complex<T> y(x._N);
        std::complex<typename T::value_type>* xptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(x._ptr);
        typename T::value_type* yptr
            = reinterpret_cast<typename T::value_type*>(y._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *yptr++ = arg(*xptr++);
            *yptr++ = 0;
        }
        return y;
    }

    friend Complex<T> conj(const Complex<T>& x) {
        Complex<T> y(x._N);
        std::complex<typename T::value_type>* xptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(x._ptr);
        std::complex<typename T::value_type>* yptr
            = reinterpret_cast<std::complex<typename T::value_type>*>(y._ptr);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            *yptr++ = conj(*xptr++);
        }
        return y;
    }

    friend Complex<T> cos(const Complex<T>& x) {
        Complex<T> y(x._N);
        typename T::value_type* xptr
            = reinterpret_cast<typename T::value_type*>(x._ptr);
        typename T::value_type* yptr
            = reinterpret_cast<typename T::value_type*>(y._ptr);
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

template<> Complex<gpuDoubleComplex>::Complex(size_t N);
template<> Complex<gpuDoubleComplex>::~Complex();

