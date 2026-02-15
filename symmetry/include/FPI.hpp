// Copyright 2022 Caleb Magruder

#pragma once

#include <complex>
#include <string>

#include "Complex.hpp"
#include "Image.hpp"
#include "Param.h"
#include "PBar.h"

// Fixed Point Iteration (FPI) class for generating symmetric chaotic attractors.
//
// Iterates a complex-valued map z_{k+1} = F(z_k) and accumulates a histogram
// of orbit visits into a 64-bit grayscale image. The template parameter T
// controls the floating-point precision (defaults to double).
template <typename T = cpuDouble>
class FPI : public Image<uint64_t, 1>{
 protected:
    using Scalar = typename T::Scalar;
    using Type = typename T::Type;

 public:
    // Constructs the FPI from a parameter set.
    //
    // Casts all parameters to the working precision type and seeds the orbit
    // at z = (0, -0.12).
    //
    // Args:
    //   p:     Parameter set defining the map coefficients and image dimensions.
    //   label: Label displayed on the progress bar.
    explicit FPI(Param p, const std::string& label = " FPI ")
        : Image(p.resy, p.resx),
          _param(p),
          _z(0, -0.12),
          _label(label) {
        // Cast double-precision parameters to the working type T
        this->_alpha = static_cast<Scalar>(_param.alpha);
        this->_beta = static_cast<Scalar>(_param.beta);
        this->_delta = static_cast<Scalar>(_param.delta);
        this->_gamma = static_cast<Scalar>(_param.gamma);
        this->_n = static_cast<Scalar>(_param.n);
        this->_p = static_cast<Scalar>(_param.p);
        this->_lambda = _param.lambda;
        this->_omega = _param.omega;
    }

    ~FPI() {}

    // Evaluates the chaotic map F(z) on Complex<T> arrays.
    //
    // The map has the general form, where mu = complex(lambda, omega):
    //   F(z) = (mu + alpha*|z|^2 + beta*Re(z^n)
    //           + delta*cos(n*p*arg(z))*|z|) * z + gamma*conj(z)^{n-1}
    //
    // Operates on arrays of N complex numbers via Complex<T> operations,
    // enabling both CPU (N=1) and GPU (N=65536) execution with the same code.
    // Does not include renormalization — callers handle that separately.
    void F(Complex<T>& z, Complex<T>& znm1, Complex<T>& bracket,
           Complex<T>& alphaAbsSq, Complex<T>& betaReZn,
           Complex<T>& deltaCosArg, Complex<T>& abs_copy,
           Complex<T>& gammaConjZnm1,
           const Type& s_alpha, const Type& s_beta,
           const Type& s_delta, const Type& s_gamma,
           const Type& s_np) {
        int ni = static_cast<int>(std::round(this->_n));

        // z^{n-1}
        znm1 = z;
        for (int j = 1; j < ni - 1; j++) znm1 *= z;

        // mu
        bracket.fill(this->_lambda, this->_omega);

        // alpha * |z|^2
        abs_copy = z;
        abs_copy.abs();
        alphaAbsSq = abs_copy;
        alphaAbsSq *= abs_copy;
        alphaAbsSq *= s_alpha;

        // beta * Re(z^n)
        betaReZn = znm1;
        betaReZn *= z;
        betaReZn.zero_imag();
        betaReZn *= s_beta;

        // delta * cos(n*p*arg(z)) * |z|
        deltaCosArg = z;
        deltaCosArg.arg();
        deltaCosArg *= s_np;
        deltaCosArg.cos();
        deltaCosArg *= abs_copy;
        deltaCosArg *= s_delta;

        // bracket = mu + alpha|z|^2 + beta*Re(z^n) + delta*cos(...)*|z|
        bracket += alphaAbsSq;
        bracket += betaReZn;
        bracket += deltaCosArg;

        // bracket * z
        bracket *= z;

        // gamma * conj(z^{n-1})
        gammaConjZnm1 = znm1;
        gammaConjZnm1.conj();
        gammaConjZnm1 *= s_gamma;

        // z = bracket + gamma*conj(z)^{n-1}
        z = bracket;
        z += gammaConjZnm1;
    }

    // Runs the fixed-point iteration for niter steps.
    //
    // Allocates Complex<T>(1) arrays and iterates F(z), accumulating orbit
    // visits into a histogram image. Periodic perturbations break spurious
    // cycles, and iterates on invariant axes are nudged off.
    //
    // Args:
    //   niter: Number of iterations to run.
    void run_fpi(uint64_t niter) {
        const int N = 1;
        Complex<T> z(N), znm1(N), bracket(N), alphaAbsSq(N),
                   betaReZn(N), deltaCosArg(N), abs_copy(N),
                   gammaConjZnm1(N);

        Type s_alpha(this->_alpha, 0), s_beta(this->_beta, 0),
             s_delta(this->_delta, 0), s_gamma(this->_gamma, 0),
             s_np(this->_n * this->_p, 0);

        // Seed from _z member
        z.fill(this->_z.real(), this->_z.imag());

        // Discard initial transient iterates to reach the attractor
        for (int i = 0; i < 1e2 * this->_init_iter; i++) {
            F(z, znm1, bracket, alphaAbsSq, betaReZn,
              deltaCosArg, abs_copy, gammaConjZnm1,
              s_alpha, s_beta, s_delta, s_gamma, s_np);
            // Renormalize
            auto zval = z[0];
            auto a = std::abs(zval);
            if (a > 8.0) z.fill(zval.real() * 3.0/a, zval.imag() * 3.0/a);
        }

        PBar pbar(niter, 8, this->_label);
        for (pbar = 0; pbar < niter; pbar++) {
            F(z, znm1, bracket, alphaAbsSq, betaReZn,
              deltaCosArg, abs_copy, gammaConjZnm1,
              s_alpha, s_beta, s_delta, s_gamma, s_np);
            auto zval = z[0];
            auto a = std::abs(zval);
            if (a > 8.0) {
                z.fill(zval.real() * 3.0/a, zval.imag() * 3.0/a);
                zval = z[0];
            }

            // Perturb every 1000 iterations to break periodic cycles
            if (this->_add_noise && static_cast<int>(pbar) % 1000 == 0) {
                z.fill(zval.real() * 0.99 - 1e-2 * sgn(zval.real()),
                       zval.imag() * 0.99 - 1e-2 * sgn(zval.imag()));
                for (int j = 0; j < this->_init_iter; j++) {
                    F(z, znm1, bracket, alphaAbsSq, betaReZn,
                      deltaCosArg, abs_copy, gammaConjZnm1,
                      s_alpha, s_beta, s_delta, s_gamma, s_np);
                    zval = z[0]; a = std::abs(zval);
                    if (a > 8.0) z.fill(zval.real()*3.0/a, zval.imag()*3.0/a);
                }
                zval = z[0];
            }

            // Nudge off invariant axes
            if (std::abs(zval.real()) < 1e-15) {
                z.fill(0.001, zval.imag());
                for (int j = 0; j < this->_init_iter; j++) {
                    F(z, znm1, bracket, alphaAbsSq, betaReZn,
                      deltaCosArg, abs_copy, gammaConjZnm1,
                      s_alpha, s_beta, s_delta, s_gamma, s_np);
                    zval = z[0]; a = std::abs(zval);
                    if (a > 8.0) z.fill(zval.real()*3.0/a, zval.imag()*3.0/a);
                }
                zval = z[0];
            }
            if (std::abs(zval.imag()) < 1e-15) {
                z.fill(zval.real(), 0.001);
                for (int j = 0; j < this->_init_iter; j++) {
                    F(z, znm1, bracket, alphaAbsSq, betaReZn,
                      deltaCosArg, abs_copy, gammaConjZnm1,
                      s_alpha, s_beta, s_delta, s_gamma, s_np);
                    zval = z[0]; a = std::abs(zval);
                    if (a > 8.0) z.fill(zval.real()*3.0/a, zval.imag()*3.0/a);
                }
                zval = z[0];
            }

            // Map the complex iterate to pixel coordinates and accumulate
            double size = std::sqrt(_rows*_cols);
            int c = floor(_param.scale*size/2*zval.real() + _cols/2);
            int r = floor(_param.scale*size/2*zval.imag() + _rows/2);
            if (r >= 0 && r < _rows && c >= 0 && c < _cols) {
                ++(*this)[r][c];
            }
        }

        // Copy back for test access
        auto zfinal = z[0];
        this->_z = Type(zfinal.real(), zfinal.imag());
    }

    // Convenience overload that runs for the default iteration count from Param.
    void run_fpi() { run_fpi(_param.n_iter); }

    // Writes the accumulated histogram to a 16-bit PGM image file.
    //
    // The 64-bit histogram counts are linearly rescaled to [0, UINT16_MAX],
    // then a logarithmic rescale is applied to compress the dynamic range
    // and reveal structure in low-count regions.
    //
    // Args:
    //   filename: Output path for the PGM file.
    void write(const std::string& filename) const {
        // Linearly rescale 64-bit histogram to 16-bit range
        Image<uint16_t, 1> im(_rows, _cols);
        uint64_t max = this->max();
        for (int r = 0; r < _rows; r++) {
            for (int c = 0; c < _cols; c++) {
                im[r][c] = static_cast<uint16_t>(
                    static_cast<double>((*this)[r][c]) / max * __UINT16_MAX__);
            }
        }
        // Apply log rescale to balance dynamic range across the image
        im.logRescale();
        im.write(filename);
    }

    // Returns the sign of val: -1, 0, or +1.
    template <typename U>
    static int sgn(U val) {
        return (U(0) < val) - (val < U(0));
    }

 protected:
    Type _z;        // Current orbit iterate
    uint64_t _init_iter = 10;     // Transient iterations before accumulation
    bool _add_noise = true;       // Whether to perturb the orbit to break cycles

 private:
    const Param _param;           // Configuration parameters for the map

    Scalar _lambda;               // Complex linear coefficient (real part)
    Scalar _omega;                // Complex linear coefficient (imag part)
    Scalar _alpha;                // Coefficient for |z|^2 term
    Scalar _beta;                 // Coefficient for Re(z^n) term
    Scalar _delta;                // Coefficient for angular modulation term
    Scalar _gamma;                // Coefficient for conjugate coupling term
    Scalar _n;                    // Symmetry order (n-fold rotational symmetry)
    Scalar _p;                    // Angular frequency multiplier for delta term

    const std::string _label;     // Label displayed on the progress bar
};

#ifdef CMAKE_CUDA_COMPILER
template<>
void FPI<gpuDouble>::run_fpi(uint64_t niter);
#endif
