// Copyright 2026 Caleb Magruder

#include "Param.h"
#include "JSON.h"

Param::Param(double lambda, double alpha, double beta, double gamma, double omega,
      double n, double delta, double p, double scale, uint64_t n_iter,
      uint64_t resx, uint64_t resy) : 
        lambda(lambda), alpha(alpha), beta(beta), gamma(gamma), omega(omega), n(n),
        delta(delta), p(p), scale(scale), n_iter(n_iter), resx(resx), resy(resy)
      {}

Param::Param(std::string fileName) {
    JSON json(fileName);

    lambda = static_cast<double>(json["lambda"]);
    alpha = static_cast<double>(json["alpha"]);
    beta = static_cast<double>(json["beta"]);
    gamma = static_cast<double>(json["gamma"]);
    omega = static_cast<double>(json["omega"]);
    n = static_cast<double>(json["n"]);
    delta = static_cast<double>(json["delta"]);
    p = static_cast<double>(json["p"]);
    scale = static_cast<double>(json["scale"]);
    n_iter = static_cast<double>(json["n_iter"]);
    scale = static_cast<double>(json["scale"]);
    n_iter = static_cast<double>(json["n_iter"]);
    resx = static_cast<double>(json["resx"]);
    resy = static_cast<double>(json["resy"]);
}
