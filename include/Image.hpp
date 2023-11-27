// Copyright 2022 Caleb Magruder

#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include "Color.h"
#include "Pixel.hpp"
#include "PBar.h"

template <typename S, typename T>
static constexpr S cast(const T& in) {
    if constexpr(sizeof(S) > sizeof(T)) {
        S out = static_cast<S>(in);
        uint64_t mult = 1;
        for (int i = 0; i < sizeof(S) - sizeof(T); i += sizeof(T)) {
            mult = (mult << 8 * sizeof(T)) + 1;
        }
        return out * mult;
    } else if constexpr(sizeof(S) == sizeof(T)) {
        return static_cast<S>(in);
    } else {  // sizeof(S) < sizeof(T)
        return static_cast<S>(in >> 8 * (sizeof(T) - sizeof(S)));
    }
}

template <typename T, int COLORS>
class Image {
 public:
    Image(size_t rows, size_t cols) : _rows(rows), _cols(cols) {
        _allocate();
    }

    explicit Image(std::string filename);

    ~Image() { delete[] _data; }

    Pixel<T, COLORS> operator[](size_t row) {
        return Pixel<T, COLORS>(_data + row * _cols * COLORS);
    }

    const Pixel<T, COLORS> operator[](size_t row) const {
        return Pixel<T, COLORS>(_data + row * _cols * COLORS);
    }

    Image& operator=(std::initializer_list<T> l) {
        (*this)[0] = l;
        return *this;
    }

    bool operator==(const Image<T, COLORS>& im) const {
        if (this->rows() != im.rows()
            || this->cols() != im.cols()
            || this->colors() != im.colors()) {
            return false;
        }
        for (ptrdiff_t i = 0; i < _size(); i++) {
            if (this->_data[i] != im._data[i]) return false;
        }
        return true;
    }

    size_t rows() const { return _rows; }
    size_t cols() const { return _cols; }
    size_t colors() const { return COLORS; }

    T max() const {
        T max = 0;
        for (ptrdiff_t i = 0; i < _size(); i++)
            max = _data[i] > max ? _data[i] : max;
        return max;
    }

    T min() const {
        T min = std::numeric_limits<T>::max();
        for (ptrdiff_t i = 0; i < _size(); i++)
            min = _data[i] < min ? _data[i] : min;
        return min;
    }

    void write(const std::string& filename) const;

    std::vector<uint64_t> hist() const;
    std::vector<uint64_t> cdf() const;
    void logRescale(const double M = 100);

 protected:
    size_t _rows;
    size_t _cols;
    void _allocate();
    T* _data = NULL;

 private:
    size_t _size() const {
        return _rows * _cols * COLORS;
    }
};

template<typename T, int COLORS>
std::ostream& operator<<(std::ostream& os, const Image<T, COLORS>& im) {
    os << "Colors : " << COLORS << std::endl
        << "Rows   : " << im.rows() << std::endl
        << "Cols   : " << im.cols() << std::endl;
    std::vector<int> subtotals({0, 0, 0, 0});
    std::vector<uint64_t> hist = im.hist();
    for (int i = 0; i < hist.size() / 4; i++) {
        os << std::setw(5) << std::right << i  << " : "
            << std::setw(5) << std::left << hist[i]
                << "  ||  "
            << std::setw(5) << std::right << i + hist.size() / 4 << " : "
            << std::setw(5) << std::left << hist[i + hist.size() / 4]
                << "  ||  "
            << std::setw(5) << std::right << i + hist.size() / 2 << " : "
            << std::setw(5) << std::left << hist[i + hist.size() / 2]
                << "  ||  "
            << std::setw(5) << std::right << i + 3 * hist.size() / 4 <<
                " : "
            << std::setw(5) << std::left << hist[i + 3 * hist.size() / 4]
            << std::endl;
        subtotals[0] += hist[i];
        subtotals[1] += hist[i + hist.size() / 4];
        subtotals[2] += hist[i + hist.size() / 2];
        subtotals[3] += hist[i + 3 * hist.size() / 4];
    }
    os << "Hist<4> = {" << subtotals[0] << ", " << subtotals[1] << ", "
        << subtotals[2] << ", " << subtotals[3] << "}" <<  std::endl;
    return os;
}

template<typename T, int COLORS>
Image<T, COLORS>::Image(std::string filename) {
    std::ifstream filePGM;;
    filePGM.open(filename);
    if (!filePGM.good()) {
        throw std::runtime_error("Error: file " + filename + " not found.");
    }
    std::string magicIdentifier;
    filePGM >> magicIdentifier;
    if (magicIdentifier == "P5") {
        if (COLORS != 1) {
            throw std::runtime_error("P5: COLORS == " + std::to_string(COLORS));
        }
    } else if (magicIdentifier == "P6") {
        if (COLORS != 3)
            throw std::runtime_error("P6: COLORS == " + std::to_string(COLORS));
    } else {
        std::cerr << "Identifyer = " + magicIdentifier << std::endl;
        throw std::runtime_error(
            "Magic Identifier in " + filename + " Invalid.");
    }

    filePGM >> _cols >> _rows;
    _allocate();
    size_t max;
    filePGM >> max;
    if (max != std::numeric_limits<T>::max())
        throw std::runtime_error("PGM "
                + std::to_string(std::numeric_limits<T>::max())
                + "!=" + std::to_string(max)
                + " invalid.");

    size_t pos = filePGM.tellg();
    // Skip 8 bits over the '\n' character after 'max' in file
    filePGM.seekg(pos+1);
    char* ptr;
    for (size_t r=0; r < _rows; r++) {
        for (ptrdiff_t c=0; c < _cols * COLORS; c++) {
            for (ptrdiff_t k=0; k < COLORS; k++) {
                ptr = reinterpret_cast<char*>(
                    static_cast<void*>((*this)[r][c][k]));
                if (sizeof(T) == 2) {
                    filePGM.read(ptr+1, 1);
                }
                filePGM.read(ptr, 1);
            }
        }
    }
    filePGM.close();
}

template<typename T, int COLORS>
void Image<T, COLORS>::write(const std::string& filename) const {
    if (sizeof(T) > 2) {
        throw std::runtime_error("Invalid size: size(T) > 2");
    }
    std::ofstream myfile(filename, std::ios::trunc | std::ios_base::binary);
    if (COLORS == 1) {
        if (filename[filename.length() - 2] != 'g')
            throw std::runtime_error("Filename " + filename
                + " does not satisfy Portable GrayMap extension (*.pgm)");
        myfile << "P5\n";
    } else if (COLORS == 3) {
        if (filename[filename.length() - 2] != 'p')
            throw std::runtime_error("Filename " + filename
                + " does not satisfy Portable PixMap extension (*.ppm)");
        myfile << "P6\n";
    }
    myfile << this->_cols << " " << this->_rows << std::endl;
    myfile << int(std::numeric_limits<T>::max()) << "\n";

    // PBar progress(this->_rows * this->_cols, 8, "Write");
    for (ptrdiff_t r = 0; r < this->_rows; r++) {
        for (ptrdiff_t c = 0; c < this->_cols; c++) {
            for (ptrdiff_t k = 0; k < COLORS; k++) {
                uint8_t* ptr = reinterpret_cast<uint8_t*>(
                    static_cast<void*>((*this)[r][c][k]));
                if (sizeof(T) == 2) {
                    myfile.put(*(ptr+1));
                }
                myfile.put(*ptr);
            }
        }
        // progress = r * this->_cols;  // update pbar display
    }
    myfile.close();
}

template<typename T, int COLORS>
void Image<T, COLORS>::_allocate() {
    // Allocate image in memory; set to zeros
    _data = new T[this->_rows * this->_cols * COLORS];
    for (ptrdiff_t i = 0; i < this->_rows * this->_cols * COLORS; i++) {
        _data[i] = 0;
    }
}

template<typename T, int COLORS>
std::vector<uint64_t> Image<T, COLORS>::hist() const {
    if (COLORS != 1) {
        throw std::runtime_error("hist() invalid when COLORS="
            + std::to_string(COLORS));
    }

    std::vector<uint64_t> hist(__UINT8_MAX__ + 1);
    uint8_t index;
    for (ptrdiff_t r = 0; r < _rows; r++) {
        for (ptrdiff_t c = 0; c < _cols; c++) {
            index = cast<uint8_t>(T((*this)[r][c]));
            ++hist[index];
        }
    }
    return hist;
}

template<typename T, int COLORS>
void Image<T, COLORS>::logRescale(const double M) {
    if (this->max() == 0) return;

    const long double coef
        = static_cast<long double>(std::numeric_limits<T>::max())
            / std::log2(M * static_cast<long double>(this->max()) + 1);

    for (ptrdiff_t r = 0; r < _rows; r++) {
        for (ptrdiff_t c = 0; c < _cols; c++) {
            (*this)[r][c] = static_cast<T>(
                coef * std::log2(M * static_cast<long double>((*this)[r][c])
                + 1));
        }
    }
}
