// Copyright 2023 Caleb Magruder

#pragma once

template<typename T>
class Complex{
 public:
    explicit Complex(size_t N) : _N(N) {
        this->_ptr = new T[N];
    }
    ~Complex(){ delete[] _ptr; }

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

    Complex<T>& operator*=(const Complex<T>& other) {
        assert(this->_N == other._N);
        for (ptrdiff_t i = 0; i < this->_N; i++) {
            this->_ptr[i] *= other[i];
        }
        return *this;
    }
 private:
    size_t _N;
    T* _ptr;
};

