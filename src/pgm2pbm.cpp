#include <cstring>
#include <fstream>
#include <iostream>

#include "image.h"

#include "colormap.h"

int main(int argc, char** argv){
    if (argc == 1)
        return 0;

    // Convert filename extension *.pgm -> *.pbm
    char* fileNamePGM = argv[1];
    size_t lengthFileName = strlen(fileNamePGM); //*.pgm
    char fileNamePBM[lengthFileName];
    strcpy(fileNamePBM,fileNamePGM);
    fileNamePBM[lengthFileName-2] = 'b';

    char* fileNameColorMap = argv[2];

    colormap map(fileNameColorMap);

    image* imPGM = new image(fileNamePGM); // Load pgm image from file
    size_t rows = imPGM->getRows();
    size_t cols = imPGM->getCols();
    size_t colors = 3;
    image* imPBM = new image(rows,cols,colors); // Allocate empty RGB

    uint16_t max(__UINT16_MAX__);
    // map2monochrome(imPGM, imPBM, map[1]);
    remap(imPGM);
    map2multichrome(imPGM, imPBM, map);
    imPBM->write(fileNamePBM);

    delete imPGM;
    delete imPBM;
    // delete[] percent;
    return 0;
}