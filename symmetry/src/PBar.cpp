// Copyright 2022 Caleb Magruder

#include <string>
#include <iostream>
#include <iomanip>  // std::setw
#include <cmath>  // std::round

#include "PBar.h"

PBar::PBar(const uint64_t n_iter,
           const int pbar_width,
           std::string tag,
           std::ostream& stream)
    : _n_iter(n_iter), _pbar_width(pbar_width), _i(0),
        _tag(tag), _stream(stream), _tbegin(time(0)) {
    _tlast_display_update = 0;
    display(0);
}

void PBar::operator=(const uint64_t i) {
    _i = i;
    // If the number of seconds rounded to int changes,
    // refresh progress bar. This causes one update a second.
    const time_t time_elapsed = static_cast<time_t>(
        std::round(static_cast<double>(time(0) - _tbegin)));
    if (time_elapsed > this->_tlast_display_update) {
        this->_tlast_display_update = time_elapsed;
        display(time_elapsed);
    }
}

const bool PBar::operator<(const uint64_t i) const { return _i < i; }

PBar& PBar::operator++(int) {
    *this = this->_i + 1;
    return *this;
}

void PBar::display(const time_t time_elapsed) {
    _stream << "\r";
    _stream << _tag << " : ||";
    int j;
    double ip1 = static_cast<double>(_i + 1);
    double width = static_cast<double>(_pbar_width);
    for (j = 0; j < std::round(ip1 * width / _n_iter); j++)
        _stream << "#";
    for ( ; j < _pbar_width; j++)
        _stream << " ";
    _stream << "|| : ";
    _stream << std::setw(3) << std::round(ip1/_n_iter*100);
    _stream << "%  ";
    uint64_t time_est = _i ? static_cast<uint64_t>(
        static_cast<double>(_n_iter) / static_cast<double>(_i)
        * time_elapsed) : 0;
    _stream << std::setw(2) << time_elapsed / 3600 << ":"
        << std::setw(2) << std::setfill('0') << time_elapsed / 60 % 60 << ":"
        << std::setw(2) << time_elapsed % 60 << " / "
        << std::setw(2) << std::setfill(' ') << time_est / 3600 << ":"
        << std::setw(2) << std::setfill('0') << time_est / 60 % 60 << ":"
        << std::setw(2) << time_est % 60 << " [h:m:s]"
        << std::setfill(' ') << std::flush;
}
