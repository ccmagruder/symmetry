#include <cstring>
#include <fstream>
#include <iostream>
#include <bit>

#include "image.cpp"

#include "colormap.h"

int main(int argc, char** argv){
    if (argc <= 2) return 0;

    char* fileNameDest = argv[1];
    char* fileNameMap = argv[2];

    colormap map(fileNameMap);

    for (ptrdiff_t i=0; i<map._numColors; i++){
        std::cout << map._alpha[i]
            << "  " << *(map._map[i]._r)
            << "  " << *(map._map[i]._g)
            << "  " << *(map._map[i]._b) << std::endl;
    }

    size_t dim = 256;
    image im(dim,dim,3);
    const long double ratio = __UINT16_MAX__/dim;
    color ctmp;
    ptrdiff_t r;
    r = 0;
    for (r=0; r<dim/2; r++){
        for (size_t c=0; c<dim; c++){
            ctmp = map.lookup(static_cast<uint16_t>(static_cast<long double>(c)*ratio));
            memcpy(im[r]+3*c,ctmp._r,6);
        }
    }
    
    for (;r<dim; r++){
        for (ptrdiff_t c=0; c<dim; c++){
            uint16_t tmp = static_cast<uint16_t>(static_cast<long double>(c)*ratio);
            memcpy(im[r]+3*c,&tmp,2);
            memcpy(im[r]+3*c+1,&tmp,2);
            memcpy(im[r]+3*c+2,&tmp,2);
        }
    }

    im.write(fileNameDest);

    return 0;
}