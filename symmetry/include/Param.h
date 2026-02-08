// Copyright 2022 Caleb Magruder

#pragma once

#include <cstdint>
#include <string>

class Param {
 public:
    explicit Param(std::string fileName);

    static std::size_t hash(const Param& p) {
        std::size_t h1 = std::hash<double>{}(p.lambda);
        std::size_t h2 = std::hash<double>{}(p.alpha);
        std::size_t h3 = std::hash<double>{}(p.beta);
        std::size_t h4 = std::hash<double>{}(p.gamma);
        std::size_t h5 = std::hash<double>{}(p.omega);
        std::size_t h6 = std::hash<double>{}(p.n);
        std::size_t h7 = std::hash<double>{}(p.delta);
        std::size_t h8 = std::hash<double>{}(p.p);
        std::size_t h9 = std::hash<double>{}(p.scale);
        std::size_t h10 = std::hash<uint64_t>{}(p.n_iter);
        std::size_t h11 = std::hash<uint64_t>{}(p.resx);
        std::size_t h12 = std::hash<uint64_t>{}(p.resy);

        // https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
        std::size_t h = 17;
        h = h * 31 + h1;
        h = h * 31 + h2;
        h = h * 31 + h3;
        h = h * 31 + h4;
        h = h * 31 + h5;
        h = h * 31 + h6;
        h = h * 31 + h7;
        h = h * 31 + h8;
        h = h * 31 + h9;
        h = h * 31 + h10;
        h = h * 31 + h11;
        h = h * 31 + h12;

        return h;
    }

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
