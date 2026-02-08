// Copyright 2023 Caleb Magruder
//
// Benchmarks for the Complex class template.
//
// Measures performance of Complex array operations using Google Benchmark.
// Benchmarks are parameterized over numeric types to compare CPU (double)
// and GPU (gpuDouble) implementations.

#include <benchmark/benchmark.h>

#include "Complex.hpp"

// Benchmarks element-wise addition of Complex arrays.
//
// Measures the time to perform in-place addition (x += y) for arrays
// of varying sizes. The size is controlled by state.range(0).
//
// Args:
//   state: Benchmark state containing the array size in range(0).
template <typename T>
void bComplexAddition(benchmark::State& state) {  // NOLINT
    Complex<T> x(state.range(0)), y(state.range(0));
    for (auto _ : state)
        x += y;
}

// CPU benchmark: Complex<double> addition.
// Tests array sizes from 4K to 16K elements, multiplied by 4 each step.
BENCHMARK_TEMPLATE(bComplexAddition, double)
    ->RangeMultiplier(4)->Range(4*1024, 16*1024)
    ->Unit(benchmark::kMicrosecond);

// GPU benchmark: Complex<gpuDouble> addition.
// Tests array sizes from 4K to 16K elements, multiplied by 4 each step.
// Only enabled when CUDA compiler is available.
#ifdef CMAKE_CUDA_COMPILER
BENCHMARK_TEMPLATE(bComplexAddition, gpuDouble)
    ->RangeMultiplier(4)->Range(4*1024, 16*1024)
    ->Unit(benchmark::kMicrosecond);
#endif
