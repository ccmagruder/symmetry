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
    ->RangeMultiplier(16)->Range(1024, 256*1024*1024)
    ->Unit(benchmark::kMillisecond);
BENCHMARK(bComplexAddition<double>)
    ->RangeMultiplier(16)->Range(1024, 256*1024*1024)
    ->Unit(benchmark::kMillisecond);
#ifdef CMAKE_CUDA_COMPILER
BENCHMARK(bComplexAddition<gpuDouble>)
    ->RangeMultiplier(16)->Range(1024, 256*1024*1024)
    ->Unit(benchmark::kMillisecond);
#endif
