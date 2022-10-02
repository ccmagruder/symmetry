#pragma once

#include <complex>
#include <string>

class image
{
protected:
    size_t _rows;
    size_t _cols;
    size_t _colors;
    uint16_t** _data = NULL;
public:
    image(size_t rows, size_t cols, size_t colors = 1);
    image(std::string filename);
    ~image();
    uint16_t* &operator[](size_t row);
    void write(std::string filename);

    size_t getRows(){ return _rows; }
    size_t getCols(){ return _cols; }
    size_t getColors(){ return _colors; }    
private:
    void _allocate();
};