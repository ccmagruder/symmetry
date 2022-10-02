#include <iostream>
#include <filesystem>

#include "param.h"
#include "INIReader.h"

param::param(char* fileName){

    reader = new INIReader(fileName);

    if (reader->ParseError() < 0) {
        std::cerr << "Error: Parsing " << fileName << " failed.\n";
        throw 1;
    }

    if (!reader->HasSection("param")) {
        std::cerr << "Error (INI): [param] cannot be parsed.\n";
        throw 1;
    }

    if (!reader->HasSection("image")) {
        std::cerr << "Error (INI): [image] cannot be parsed.\n";
        throw 1;
    }

    char* param = (char*) "param";
    char* image = (char*) "image";
    try {
        lambda = getDouble(param, (char *) "lambda");
        alpha = getDouble(param, (char *) "alpha");
        beta = getDouble(param, (char *) "beta");
        gamma = getDouble(param, (char *) "gamma");
        omega = getDouble(param, (char *) "omega");
        n = getDouble(param, (char *) "n");
        scale = getDouble(image, (char *) "scale");
        n_iter = getInteger(image, (char *) "n_iter");
        resx = getDouble(image, (char *) "resx");
        resy = getDouble(image, (char *) "resy");
    }
    catch (char* value) {
        std::cerr << "Error (INI): " << value << " cannot be parsed.\n";
        throw 1;
    }
}

double param::getDouble(char* section, char* value){
    if (reader->HasValue(section,value))
        return reader->GetReal(section,value, -1);
    else {
        throw value;
    }
}

uint64_t param::getInteger(char* section, char* value){
    if (reader->HasValue(section,value))
        return uint64_t(reader->GetInteger(section,value, -1));
    else {
        throw value;
    }
}