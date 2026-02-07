// Copyright 2023 Caleb Magruder

#include <benchmark/benchmark.h>

#include "Complex.hpp"

template <typename T>
void bComplexAddition(benchmark::State& state) {  // NOLINT
    Complex<T> x(state.range(0)), y(state.range(0));
    for (auto _ : state)
        x += y;
}

BENCHMARK_TEMPLATE(bComplexAddition, double)
    ->RangeMultiplier(4)->Range(4*1024, 16*1024)
    ->Unit(benchmark::kMicrosecond);
#ifdef CMAKE_CUDA_COMPILER
BENCHMARK_TEMPLATE(bComplexAddition, gpuDouble)
    ->RangeMultiplier(4)->Range(4*1024, 16*1024)
    ->Unit(benchmark::kMicrosecond);
#endif
