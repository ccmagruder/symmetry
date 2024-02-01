// Copyright 2023 Caleb Magruder

#include <benchmark/benchmark.h>

#include "Complex.hpp"

static void bComplexAddition(benchmark::State& state) {  // NOLINT
    Complex<float> x(state.range(0)), y(state.range(0));
    for (auto _ : state)
        Complex<float> z = x + y;
}

BENCHMARK(bComplexAddition)->RangeMultiplier(4)->Range(64, 4096);
