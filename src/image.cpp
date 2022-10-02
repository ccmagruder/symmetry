#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>

#include "image.h"
#include "pbar.h"

#define PBAR_WIDTH 50

image::image(size_t rows, size_t cols, size_t colors) : _rows(rows), _cols(cols), _colors(colors)
{
    _allocate();
}

image::image(std::string filename)
{
    std::ifstream filePGM;;
    filePGM.open(filename);
    std::string magicIdentifier;
    filePGM >> magicIdentifier;
    if (strcmp(magicIdentifier.c_str(),"P5")==0)
    {
        _colors=1;
    }
    else if(strcmp(magicIdentifier.c_str(),"P6")==0){
        _colors=3;
    }
    else
    {
        std::cerr<< "Magic Identifier in " << filename << " Invalid.\n";
        assert(false);
    }

    filePGM >> _cols >> _rows;
    _allocate();
    size_t max;
    filePGM >> max;
    assert(max == __UINT16_MAX__);

    size_t pos = filePGM.tellg();
    filePGM.seekg(pos+1); //Skip 8 bits over the '\n' character after 'max' in file
    char* ptr;
    for (size_t r=0; r<_rows; r++){
        for (ptrdiff_t c=0;c<_cols*_colors;c++){
            ptr = reinterpret_cast<char*>(_data[r]+c);
            filePGM.read(ptr+1,1);
            filePGM.read(ptr,1);
        }
    }
    filePGM.close();
}

image::~image(){
    for(size_t r = 0; r < _rows; r++){
        delete[] _data[r];
    }
    delete[] _data;
}

uint16_t*& image::operator[](size_t row){
    return _data[row];
}

void image::write(std::string filename){
    std::ofstream myfile;
    myfile.open(filename, std::ios::trunc | std::ios_base::binary);
    if (_colors == 1){
        myfile << "P5\n";
    }
    else if (_colors == 3){
        myfile << "P6\n";
    }
    myfile << _cols << " " << _rows << std::endl;
    myfile << __UINT16_MAX__ << "\n";

    pbar progress(_rows*_cols, PBAR_WIDTH, "Write");
    for (ptrdiff_t r = 0; r < _rows; r++){
        for (ptrdiff_t c=0; c<_cols*_colors; c++){
            uint8_t* ptr = reinterpret_cast<uint8_t*>(_data[r]+c);
            myfile.put(*(ptr+1));
            myfile.put(*ptr);
            // myfile.write((char*)_data[r]+c,2); // write column-wise to file
        }
        progress = r * _cols; // update pbar display
    }
    myfile.close();
}

void image::_allocate()
{
    //Allocate image in memory; set to zeros
    _data = new uint16_t*[_rows];
    for(size_t r = 0; r < _rows; r++){
        _data[r] = new uint16_t[_cols*_colors];
        for(size_t c = 0; c < _cols*_colors; c++){
            _data[r][c] = 0;
        }
    }
}