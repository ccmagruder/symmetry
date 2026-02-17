// Copyright 2022 Caleb Magruder

#pragma once

#include <complex>
#include <string>

#include "Complex.hpp"
#include "Image.hpp"
#include "Param.h"
#include "PBar.h"

// Default number of parallel orbits per type.
// GPU uses many more orbits to saturate the hardware.
template<typename T> struct FPITraits           { static constexpr int N = 256; };
template<>           struct FPITraits<gpuDouble> { static constexpr int N = 65536; };

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
          _z(0, -0.12),
          _param(p),
          _label(label),
          _znm1(_N), _bracket(_N), _alphaAbsSq(_N), _betaReZn(_N),
          _deltaCosArg(_N), _abs_copy(_N), _gammaConjZnm1(_N) {
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
    void F(Complex<T>& z) {
        int ni = static_cast<int>(std::round(this->_n));
        Type s_alpha{this->_alpha, 0}, s_beta{this->_beta, 0},
             s_delta{this->_delta, 0}, s_gamma{this->_gamma, 0},
             s_np{this->_n * this->_p, 0};

        // z^{n-1}
        _znm1 = z;
        for (int j = 1; j < ni - 1; j++) _znm1 *= z;

        // mu
        _bracket.fill(this->_lambda, this->_omega);

        // alpha * |z|^2
        _abs_copy = z;
        _abs_copy.abs();
        _alphaAbsSq = _abs_copy;
        _alphaAbsSq *= _abs_copy;
        _alphaAbsSq *= s_alpha;

        // beta * Re(z^n)
        _betaReZn = _znm1;
        _betaReZn *= z;
        _betaReZn.zero_imag();
        _betaReZn *= s_beta;

        // delta * cos(n*p*arg(z)) * |z|
        _deltaCosArg = z;
        _deltaCosArg.arg();
        _deltaCosArg *= s_np;
        _deltaCosArg.cos();
        _deltaCosArg *= _abs_copy;
        _deltaCosArg *= s_delta;

        // bracket = mu + alpha|z|^2 + beta*Re(z^n) + delta*cos(...)*|z|
        _bracket += _alphaAbsSq;
        _bracket += _betaReZn;
        _bracket += _deltaCosArg;

        // bracket * z
        _bracket *= z;

        // gamma * conj(z^{n-1})
        _gammaConjZnm1 = _znm1;
        _gammaConjZnm1.conj();
        _gammaConjZnm1 *= s_gamma;

        // z = bracket + gamma*conj(z)^{n-1}
        z = _bracket;
        z += _gammaConjZnm1;
    }

    // Runs the fixed-point iteration, accumulating niter points into the
    // histogram.
    //
    // Advances _N orbits in parallel per step via F(), mirroring the GPU
    // specialization's structure.  Per-element operations (renorm, noise,
    // nudge, accumulate) use for-loops where the GPU uses CUDA kernels.
    // Total steps = ceil(niter / _N); the last step may accumulate fewer
    // than _N points.
    //
    // Args:
    //   niter: Total number of points to accumulate.
    void run_fpi(uint64_t niter) {
        Complex<T> z(this->_N);

        seed(z);

        // Discard initial transient iterates to reach the attractor
        for (int i = 0; i < 1e2 * this->_init_iter; i++) {
            F(z);
            renorm(z);
        }

        uint64_t num_steps = (niter + this->_N - 1) / this->_N;
        uint64_t total_accumulated = 0;
        PBar pbar(niter, 8, this->_label);

        for (uint64_t step = 0; step < num_steps; step++) {
            F(z);
            renorm(z);

            // Perturb every 1000 steps to break periodic cycles
            if (this->_add_noise && (step % 1000 == 0) && step > 0) {
                noise(z);
                for (int j = 0; j < this->_init_iter; j++) {
                    F(z);
                    renorm(z);
                }
            }

            nudge(z);

            // Accumulate up to _N points; last step may be partial
            uint64_t points = std::min(
                static_cast<uint64_t>(this->_N), niter - total_accumulated);
            accumulate(z, points);
            total_accumulated += points;
            pbar = total_accumulated;
        }

        // Copy back for test access via z()
        Type zfinal = z[0];
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
    int _N = FPITraits<T>::N;     // Number of parallel orbits
    unsigned long long* _d_heatmap = nullptr;  // Device heatmap buffer (GPU only)

 private:
    // Seeds _N orbits as roots-of-unity rotations of _z.
    // z[tid] = _z * exp(i * 2*pi*tid / _N).  For _N=1, z[0] = _z exactly.
    // CPU analog of fpi_seed_kernel.
    void seed(Complex<T>& z) {
        for (int tid = 0; tid < this->_N; tid++) {
            double angle = 2.0 * 3.14159265358979323846 * tid / this->_N;
            double ca = std::cos(angle), sa = std::sin(angle);
            z[tid] = Type(this->_z.real()*ca - this->_z.imag()*sa,
                          this->_z.real()*sa + this->_z.imag()*ca);
        }
    }

    // Rescales orbits that escape |z| > 8 back to radius 3.
    // CPU analog of fpi_renorm_kernel.
    void renorm(Complex<T>& z) {
        for (int tid = 0; tid < this->_N; tid++) {
            double a = std::abs(z[tid]);
            if (a > 8.0) {
                double s = 3.0 / a;
                z[tid] = Type(z[tid].real()*s, z[tid].imag()*s);
            }
        }
    }

    // Contracts orbits by 1% and shifts away from the origin to break
    // periodic cycles.  CPU analog of fpi_noise_kernel.
    void noise(Complex<T>& z) {
        for (int tid = 0; tid < this->_N; tid++) {
            z[tid] = Type(z[tid].real()*0.99 - 1e-2*sgn(z[tid].real()),
                          z[tid].imag()*0.99 - 1e-2*sgn(z[tid].imag()));
        }
    }

    // Nudges orbits off invariant axes where |Re(z)| or |Im(z)| < 1e-15.
    // CPU analog of fpi_nudge_kernel.
    void nudge(Complex<T>& z) {
        for (int tid = 0; tid < this->_N; tid++) {
            if (std::abs(z[tid].real()) < 1e-15)
                z[tid] = Type(0.001, z[tid].imag());
            if (std::abs(z[tid].imag()) < 1e-15)
                z[tid] = Type(z[tid].real(), 0.001);
        }
    }

    // Maps complex iterates to pixel coordinates and increments the
    // histogram.  CPU analog of fpi_heatmap_kernel.
    void accumulate(const Complex<T>& z, uint64_t points) {
        double size = std::sqrt(_rows * _cols);
        for (uint64_t tid = 0; tid < points; tid++) {
            int c = floor(_param.scale*size/2*z[tid].real() + _cols/2);
            int r = floor(_param.scale*size/2*z[tid].imag() + _rows/2);
            if (r >= 0 && r < _rows && c >= 0 && c < _cols)
                ++(*this)[r][c];
        }
    }

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

    // Scratch arrays for F(), sized to _N.
    Complex<T> _znm1, _bracket, _alphaAbsSq, _betaReZn,
               _deltaCosArg, _abs_copy, _gammaConjZnm1;
};

#ifdef CMAKE_CUDA_COMPILER
template<>
void FPI<gpuDouble>::seed(Complex<gpuDouble>& z);

template<>
void FPI<gpuDouble>::noise(Complex<gpuDouble>& z);

template<>
void FPI<gpuDouble>::renorm(Complex<gpuDouble>& z);

template<>
void FPI<gpuDouble>::nudge(Complex<gpuDouble>& z);

template<>
void FPI<gpuDouble>::accumulate(const Complex<gpuDouble>& z, uint64_t points);

template<>
void FPI<gpuDouble>::run_fpi(uint64_t niter);
#endif
