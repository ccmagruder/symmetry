// Copyright 2026 Caleb Magruder

#pragma once

#include <cstdint>
#include <string>

// Parameter set defining an equivariant chaotic map and its output image.
//
// Holds the coefficients of the map F(z), the symmetry order n, image
// resolution, iteration count, and display scale. Can be constructed
// directly or parsed from a JSON configuration file.
class Param {
 public:
    // Constructs a Param with explicit coefficient values.
    //
    // Args:
    //   lambda: Complex linear coefficient (real part).
    //   alpha:  Coefficient for the |z|^2 term.
    //   beta:   Coefficient for the Re(z^n) term.
    //   gamma:  Coefficient for the conjugate coupling term.
    //   omega:  Complex linear coefficient (imag part).
    //   n:      Symmetry order (n-fold rotational symmetry).
    //   delta:  Coefficient for the angular modulation term.
    //   p:      Angular frequency multiplier for the delta term.
    //   scale:  Zoom scale mapping iterates to pixel coordinates.
    //   n_iter: Number of fixed-point iterations to run.
    //   resx:   Output image width in pixels.
    //   resy:   Output image height in pixels.
    explicit Param(double lambda = 0,
                   double alpha = 0,
                   double beta = 0,
                   double gamma = 0,
                   double omega = 0,
                   double n = 0,
                   double delta = 0,
                   double p = 0,
                   double scale = 0,
                   uint64_t n_iter = 0,
                   uint64_t resx = 1,
                   uint64_t resy = 1);

    // Constructs a Param by parsing a JSON configuration file.
    //
    // Args:
    //   fileName: Path to the JSON file.
    explicit Param(std::string fileName);

    double lambda;    // Complex linear coefficient (real part)
    double alpha;     // Coefficient for |z|^2 term
    double beta;      // Coefficient for Re(z^n) term
    double gamma;     // Coefficient for conjugate coupling term
    double omega;     // Complex linear coefficient (imag part)
    double n;         // Symmetry order (n-fold rotational symmetry)
    double delta;     // Coefficient for angular modulation term
    double p;         // Angular frequency multiplier for delta term
    double scale;     // Zoom scale mapping iterates to pixel coordinates
    uint64_t n_iter;  // Number of fixed-point iterations
    uint64_t resx;    // Output image width in pixels
    uint64_t resy;    // Output image height in pixels

};
