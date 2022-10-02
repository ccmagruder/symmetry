#include "image.h"
#include "fpi.h"
#include "pbar.h"

#define PBAR_WIDTH 50

fpi::fpi(param p) : image(p.resy,p.resx), _param(p){}

std::complex<double> fpi::F(std::complex<double> z){

    std::complex<double> znew = (_param.lambda+_param.alpha*pow(abs(z),2)
                            + _param.beta*real(pow(z,_param.n))
                            + _param.omega*std::complex<double>(0,1))*z
                           + _param.gamma * pow(conj(z),(_param.n-1));

    return znew;
}

void fpi::run_fpi()
{
    std::complex<double> p(0,-0.2);

    //Initialize 1e3 transient beginning
    for(int i = 0; i < 1e3; i++)
        p = F(p);

    pbar i(_param.n_iter, PBAR_WIDTH, " FPI ");
    for(i = 0; i < _param.n_iter; i++){
        p = F(p);
        // Imaginary axis sometimes closed under FPI
        // Kicks iterate out of loop, re-initializes
        if(std::abs(real(p)) < 1e-15)
        {
            p.real(0.001);
            for(int j = 0; j < 1e3; j++)
                p = F(p);
        }
        int size = std::sqrt(_rows*_cols);
        int c = floor(_param.scale*size/2*real(p) + _cols/2);
        int r = floor(_param.scale*size/2*imag(p) + _rows/2);
        if(r >=0 && r < _rows && c >=0 && c < _cols)
        {
            _data[r][c]++;
        }
    }
    uint16_t max = 0;
    for(int r = 0; r < _rows; r++){
        for(int c = 0; c < _cols; c++) {
            if(_data[r][c] > max){
                max = _data[r][c];
            }
        }
    }
//    std::cout << "Max = " << max << std::endl;
    for(int r = 0; r < _rows; r++){
        for(int c = 0; c < _cols; c++) {
            _data[r][c] = uint16_t(double(_data[r][c])/max*65535);
        }
    }

}