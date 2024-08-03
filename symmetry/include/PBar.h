// Copyright 2022 Caleb Magruder

#pragma once

#include<ctime>
#include<iostream>
#include<string>

#include "stdint.h"

class PBar {
 public:
    PBar(const uint64_t n_iter, const int pbar_width,
        std::string tag, std::ostream& stream = std::cout);
    ~PBar() {
        time_t time_elapsed = static_cast<time_t>(
            std::round(static_cast<double>(time(0) - _tbegin)));
        display(time_elapsed);
        this->_stream << std::endl << std::flush;
    }
    void operator=(const uint64_t i);
    const bool operator<(const uint64_t i) const;
    PBar& operator++(int);
    operator const int() {
        return static_cast<int>(this->_i);
    }

    void display(const time_t time_elapsed);

 private:
    uint64_t _i;
    const uint64_t _n_iter;
    const int _pbar_width;
    const std::string _tag;
    const time_t _tbegin;
    time_t _tlast_display_update;
    std::ostream& _stream;
};
