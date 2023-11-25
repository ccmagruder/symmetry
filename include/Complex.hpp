// Copyright 2023 Caleb Magruder

#pragma once

#include <complex>

template<typename T>
class Complex{
 public:
    explicit Complex(size_t N){
        _ptr = malloc(N*sizeof(T));
    }
    ~Complex(){ free(_ptr); }
 private:
    void* _ptr;
};

