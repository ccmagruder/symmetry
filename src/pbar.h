#pragma once

#include<sstream>
#include<string>
#include<ctime>

class pbar
{
public:
    pbar(uint64_t n_iter, int pbar_width, std::string tag, std::ostream& stream=std::cout);
    void operator=(uint64_t i);
    bool operator<(uint64_t i);
    pbar operator++(int);
    operator int(){ return static_cast<int>(_i); };
    void display();
private:
    uint64_t _i;
    uint64_t _n_iter;
    int _pbar_width;
    std::string _tag;
    time_t _tbegin;
    std::ostream& _stream;
};
