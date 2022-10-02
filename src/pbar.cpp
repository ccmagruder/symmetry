#include <string>
#include <iostream>
#include <iomanip> //std::setw
#include <cmath> //std::round

#include "pbar.h"

pbar::pbar(uint64_t n_iter, int pbar_width, std::string tag, std::ostream& stream)
    : _n_iter(n_iter), _pbar_width(pbar_width), _i(0), _tag(tag), _stream(stream)
{
    _tbegin = time(0);
    display();
}

void pbar::operator=(uint64_t i){ _i = i; display(); }

bool pbar::operator<(uint64_t i){ return _i < i; }

pbar pbar::operator++(int)
{
    _i++;

    if(_i % std::max(1,int(double(_n_iter) / 100)) == 0)
        display();

    if(_i == _n_iter)
        _stream << std::endl;

    return *this;
}

void pbar::display()
{
    _stream << "\r";
    _stream << _tag << " : ||";
    int j;
    for (j = 0; j < std::round(double(_i+1) * double(_pbar_width) / double(_n_iter)); j++)
        _stream << "#";
    for( ; j<_pbar_width; j++)
        _stream << " ";
    _stream << "|| : " ;
    _stream << std::setw(3) << std::round(double(_i+1)/_n_iter*100);
    _stream << "%%  ";
    if(_i)
    {
        double time_elapsed = static_cast<double>(time(0) - _tbegin);
        time_t time_est = int(double(_n_iter) / double(_i) * time_elapsed);
        _stream << std::setw(4) << time_elapsed
            << " / " << std::setw(4) << time_est << " [s]";
    }
    _stream << std::flush;
}
