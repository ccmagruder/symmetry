#include <cstring>
#include <fstream>
#include <iostream>

#include "image.h"
#include "colormap.h"

color colormap::lookup(uint16_t grey){
    color c;
    ptrdiff_t i = 0;
    while (grey > _alpha[i+1]){
        i++;
    }
    grey -= _alpha[i];
    grey = static_cast<uint16_t>(static_cast<long double>(__UINT16_MAX__) * grey / (_alpha[i+1] - _alpha[i]));
    c = interpolate(_map[i],_map[i+1],grey);

    return c;
}

inline color interpolate(const color& a, const color& b, const uint16_t& alpha){
    uint16_t* ptrA = a._r;
    uint16_t* ptrB = b._r;
    color c;
    uint16_t* ptrC = c._r;
    // for (ptrdiff_t i = 0; i<n; i++){
        const long double alpha64 = static_cast<long double>(alpha);
        // *(ptrC++) = *(ptrA++);
        // *(ptrC++) = *(ptrA++);
        // *(ptrC++) = *(ptrA++);
            *(ptrC++) = uint16_t( sqrtl( ( (__UINT16_MAX__-alpha64) * pow(*(ptrA++)+1,2) + alpha64 * pow(*(ptrB++)+1,2) ) / __UINT16_MAX__) - 1 );
            *(ptrC++) = uint16_t( sqrtl( ( (__UINT16_MAX__-alpha64) * pow(*(ptrA++)+1,2) + alpha64 * pow(*(ptrB++)+1,2) ) / __UINT16_MAX__) - 1 );
            *(ptrC++) = uint16_t( sqrtl( ( (__UINT16_MAX__-alpha64) * pow(*(ptrA++)+1,2) + alpha64 * pow(*(ptrB++)+1,2) ) / __UINT16_MAX__) - 1 );
        // ptrA -= 3;
        // ptrB -= 3;
        // alpha ++;
    // }
    return c;
}

void map2multichrome(image *imG, image* imRGB, colormap& map){
    size_t rows = imRGB->getRows();
    size_t cols = imRGB->getCols();
    color black;
    color fill(map._map[1]);
    color* mycolor = new color(0,0,0);
    for (size_t r = 0; r<rows; r++){
        uint16_t* ptrImG = imG->operator[](r);
        uint16_t* ptrImRGB = imRGB->operator[](r);
        for (size_t c = 0; c<cols; c++){

            *mycolor = map.lookup(*ptrImG);

            memcpy(ptrImRGB,mycolor->_r,6);
            
            ptrImG+=1;
            ptrImRGB+=3;

        }
    }
    delete mycolor;
};

void remap(image* imG){
    size_t rows = imG->getRows();
    size_t cols = imG->getCols();
    size_t count = 0;
    for (size_t r = 0; r<rows; r++){
        uint16_t* ptr = imG->operator[](r);
        for (size_t c = 0; c<cols; c++){
            long double v = static_cast<long double>(*ptr);
            *ptr = uint16_t(__UINT16_MAX__ * sqrtl(v/__UINT16_MAX__));
        }
    }
}

