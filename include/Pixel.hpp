// Copyright 2022 Caleb Magruder

#pragma once

#include <iostream>
#include <limits>

template <typename T, int COLORS>
class Pixel {
 public:
    explicit Pixel(T* data) : _data(data) {}

    void operator=(const T& v) { *_data = v; }

    operator const T() const { return *_data; }

    operator void*() const { return _data; }

    T operator++() {
        if (*_data == std::numeric_limits<T>::max()) {
            throw std::runtime_error("Integer Limit Exceeded!");
        } else {
            return ++(*_data);
        }
    }

    Pixel<T, 1> operator[](ptrdiff_t cols) {
        return Pixel<T, 1>(_data + cols*COLORS);
    }

    const Pixel<T, 1> operator[](ptrdiff_t cols) const {
        return Pixel<T, 1>(_data + cols*COLORS);
    }

    Pixel& operator=(std::initializer_list<T> l) {
        T* ptr = _data;
        for (auto i = l.begin(); i < l.end(); i++) {
            *ptr = *i;
            ptr++;
        }
        return *this;
    }

 private:
    T* _data;
};
