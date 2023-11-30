// Copyright 2023 Caleb Magruder

#pragma once

#include <cassert>

enum class gpuDoubleComplex {};

template<typename T>
class Complex{
 public:
    explicit Complex(size_t N) : _N(N) {
        this->_ptr = new T[N];
    }
    ~Complex() { if (_ptr) delete[] _ptr; }

    Complex& operator=(const Complex<T>& other) {
        assert(this->_N == other._N);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            this->_ptr[i] = other[i];
        }
        return *this;
    }

    T& operator[](ptrdiff_t i) {
        return this->_ptr[i];
    }

    const T& operator[](ptrdiff_t i) const {
        return this->_ptr[i];
    }

    friend Complex<T> operator+(const Complex<T>& x, const Complex<T>& y) {
        assert(x._N == y._N);
        Complex<T> z(x._N);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            z[i] = x[i] + y[i];
        }
        return z;
    }

    friend Complex<T> operator*(const Complex<T>& x, const Complex<T>& y) {
        assert(x._N == y._N);
        Complex<T> z(x._N);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            z[i] = x[i] * y[i];
        }
        return z;
    }

    friend Complex<T> operator*(const T& a, const Complex<T>& x) {
        Complex<T> y(x._N);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            y[i] = a * x[i];
        }
        return y;
    }

    friend Complex<T> abs(const Complex<T>& x) {
        Complex<T> y(x._N);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            y[i] = T(abs(x[i]), 0);
        }
        return y;
    }

    friend Complex<T> arg(const Complex<T>& x) {
        Complex<T> y(x._N);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            y[i] = T(arg(x[i]), 0);
        }
        return y;
    }

    friend Complex<T> conj(const Complex<T>& x) {
        Complex<T> y(x._N);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            y[i] = conj(x[i]);
        }
        return y;
    }

    friend Complex<T> cos(const Complex<T>& x) {
        Complex<T> y(x._N);
        for (ptrdiff_t i = 0; i < x._N; i++) {
            y[i] = T(cos(x[i].real()), x[i].imag());
        }
        return y;
    }

 private:
    size_t _N;
    T* _ptr;
};

template<> Complex<gpuDoubleComplex>::Complex(size_t N);
