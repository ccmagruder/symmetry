#include <iostream>
#include <fstream>

#include "pbar.h"
#include "image.h"
#include "fpi.h"

using namespace std;

int main(int argc, char** argv){
    char* fileNameConfig;
    char *fileNameImage;
    if (argc < 3)
    {
        cerr << "usage: config/source.txt image/target.pgm\n";
        exit(1);
    }
    else
    {
        fileNameConfig = argv[1];
        fileNameImage = argv[2];
    }
 
    try {
        param p(fileNameConfig);
        fpi im(p);
        im.run_fpi();
        im.write(fileNameImage);
    }
    catch(int){
        exit(1);
    }

    return 0;
}