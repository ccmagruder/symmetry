// Copyright 2026 Caleb Magruder

#pragma once

#include <cstdint>
#include <string>

class Param {
 public:
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

    explicit Param(std::string fileName);

    double lambda;
    double alpha;
    double beta;
    double gamma;
    double omega;
    double n;
    double delta;
    double p;
    double scale;
    uint64_t n_iter;
    uint64_t resx;
    uint64_t resy;
};
