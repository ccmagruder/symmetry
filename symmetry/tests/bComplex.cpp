// Copyright 2023 Caleb Magruder

#include <benchmark/benchmark.h>

#include "Complex.hpp"

template <typename T>
void bComplexAddition(benchmark::State& state) {  // NOLINT
    Complex<T> x(state.range(0)), y(state.range(0));
    for (auto _ : state)
        x += y;
}

BENCHMARK(bComplexAddition<float>)
    ->RangeMultiplier(16)->Range(4*1024, 16*1024)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(bComplexAddition<double>)
    ->RangeMultiplier(16)->Range(4*1024, 16*1024)
    ->Unit(benchmark::kMicrosecond);
#ifdef CMAKE_CUDA_COMPILER
BENCHMARK(bComplexAddition<gpuDouble>)
    ->RangeMultiplier(16)->Range(4*1024, 16*1024)
    ->Unit(benchmark::kMicrosecond);
#endif
