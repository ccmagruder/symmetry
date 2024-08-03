// Copyright 2022 Caleb Magruder

#include <iostream>
#include <filesystem>

#include "Param.h"
#include "JSON.h"

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
