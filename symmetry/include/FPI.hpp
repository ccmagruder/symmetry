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
template <typename T = double>
class FPI : public Image<uint64_t, 1>{
 protected:
    using Type = typename complex_traits<T>::value_type;
    using S = std::complex<Type>;

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
        this->_alpha = static_cast<Type>(_param.alpha);
        this->_beta = static_cast<Type>(_param.beta);
        this->_delta = static_cast<Type>(_param.delta);
        this->_gamma = static_cast<Type>(_param.gamma);
        this->_n = static_cast<Type>(_param.n);
        this->_p = static_cast<Type>(_param.p);
        this->_lambda = _param.lambda;
        this->_omega = _param.omega;
    }

    ~FPI() {}

    // Evaluates the chaotic map F(z).
    //
    // The map has the general form, where mu = complex(lambda, omega):
    //   F(z) = (mu + alpha*|z|^2 + beta*Re(z^n)
    //           + delta*cos(n*p*arg(z))*|z|) * z + gamma*conj(z)^{n-1}
    // The symmetry group of the attractor is determined by the integer
    // parameter n. Aborts if the orbit diverges to NaN; renormalizes if
    // |z| exceeds 8 to keep the orbit bounded.
    //
    // Args:
    //   z: The current complex iterate.
    //
    // Returns:
    //   The next iterate F(z).
    S F(S z) {
        // Guard against divergence
        if (std::isnan(z.real())) exit(1);

        // Compute z^{n-1} (znm1 equals 'z to the n minus 1')
        this->_znm1 = z;
        for (int i = 1; i < std::round(this->_n) - 1; i++) {
            this->_znm1 *= z;
        }


        // Evaluate the equivariant map:
        //   Term 1: mu * z                           — rotation/scaling
        S mu = S(this->_lambda, this->_omega);

        //   Term 2: alpha * |z|^2 * z                — radial nonlinearity
        Type zabs = abs(z);
        S alphaZAbsSqr(zabs, 0);
        alphaZAbsSqr *= alphaZAbsSqr;
        alphaZAbsSqr *= this->_alpha;

        //   Term 3: beta * Re(z^n) * z               — n-fold symmetric coupling
        S betaReZn(this->_znm1);
        betaReZn *= z;
        betaReZn.imag(0);
        betaReZn *= this->_beta;

        //   Term 4: delta * cos(n*p*arg(z))*|z| * z  — angular modulation
        S deltaCosArgAbs(arg(z), 0);
        deltaCosArgAbs *= this->_n * this->_p;
        deltaCosArgAbs.real(std::cos(deltaCosArgAbs.real()));
        deltaCosArgAbs *= zabs;
        deltaCosArgAbs *= this->_delta;

        //   Term 5: gamma * conj(z)^{n-1}            — conjugate coupling
        S deltaConjZnm1(std::conj(this->_znm1));
        deltaConjZnm1 *= this->_gamma;

        this->_znew = (
                // this->alpha + i*this->omega
                mu
                // + this->_alpha * abs(z) * abs(z)
                + alphaZAbsSqr
                // + _param.beta * real(pow(z, _param.n))
                + betaReZn
                // + this->_beta * real(z*this->_znm1)
                + deltaCosArgAbs
                // + S(
                //     this->_delta * cos(arg(z) * this->_n * this->_p) * abs(z),
                //     0
                //   )
            ) * z                                           // NOLINT
            + deltaConjZnm1;
            //+ _param.gamma * pow(conj(z), _param.n - 1);
            // + this->_gamma * conj(this->_znm1);

        // Renormalize to prevent unbounded growth
        if (abs(this->_znew) > 8) {
            std::cerr << "Warning: abs(z)=" << abs(this->_znew)
                << " renormalized to 1\n";
            this->_znew /= abs(this->_znew) / 3.0;
        }

        return this->_znew;
    }

    // Runs the fixed-point iteration for niter steps.
    //
    // Discards initial transient iterates so the orbit settles onto the
    // attractor before accumulation. Each iterate z is then mapped to a pixel
    // in the histogram image, incrementing the corresponding bin. Periodic
    // perturbations break spurious cycles, and iterates that collapse onto
    // the real or imaginary axis are nudged off to maintain coverage.
    //
    // Args:
    //   niter: Number of iterations to run.
    void run_fpi(uint64_t niter) {
        // Discard initial transient iterates to reach the attractor
        for (int i = 0; i < 1e2 * this->_init_iter; i++) {
            _z = F(_z);
        }

        PBar i(niter, 8, this->_label);
        for (i = 0; i < niter; i++) {
            _z = F(_z);

            // Perturb every 1000 iterations to break periodic cycles
            // that can trap the orbit and leave gaps in the image
            if (this->_add_noise && static_cast<int>(i) % 1000 == 0) {
                _z.real(_z.real() * 0.99 - 1e-2 * sgn(_z.real()));
                _z.imag(_z.imag() * 0.99 - 1e-2 * sgn(_z.imag()));
                for (int j = 0; j < this->_init_iter; j++)
                    _z = F(_z);
            }

            // The real and imaginary axes can be invariant under the map.
            // If the orbit collapses onto an axis, nudge it off and
            // re-iterate to restore two-dimensional wandering.
            if (std::abs(real(_z)) < 1e-15) {
                _z.real(0.001);
                for (int j = 0; j < this->_init_iter; j++)
                    _z = F(_z);
            }

            if (std::abs(imag(_z)) < 1e-15) {
                _z.imag(0.001);
                for (int j = 0; j < this->_init_iter; j++)
                    _z = F(_z);
            }

            // Map the complex iterate to pixel coordinates and accumulate
            double size = std::sqrt(_rows*_cols);
            int c = floor(_param.scale*size/2*real(_z) + _cols/2);
            int r = floor(_param.scale*size/2*imag(_z) + _rows/2);
            if (r >=0 && r < _rows && c >=0 && c < _cols) {
                ++(*this)[r][c];
            }
        }
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
    S _z;        // Current orbit iterate
    uint64_t _init_iter = 10;     // Transient iterations before accumulation
    bool _add_noise = true;       // Whether to perturb the orbit to break cycles

 private:
    const Param _param;           // Configuration parameters for the map

    S _znm1;     // Cached z^{n-1} used in F(z)
    S _znew;     // Result of the latest map evaluation
    Type _lambda;                 // Complex linear coefficient (real part)
    Type _omega;                  // Complex linear coefficient (imag part)
    Type _alpha;                  // Coefficient for |z|^2 term
    Type _beta;                   // Coefficient for Re(z^n) term
    Type _delta;                  // Coefficient for angular modulation term
    Type _gamma;                  // Coefficient for conjugate coupling term
    Type _n;                      // Symmetry order (n-fold rotational symmetry)
    Type _p;                      // Angular frequency multiplier for delta term

    const std::string _label;     // Label displayed on the progress bar
};

#ifdef CMAKE_CUDA_COMPILER
template<>
void FPI<gpuDouble>::run_fpi(uint64_t niter);
#endif
