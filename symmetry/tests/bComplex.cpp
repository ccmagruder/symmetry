// Copyright 2023 Caleb Magruder
//
// Benchmarks for the Complex class template.
//
// Measures performance of Complex array operations using Google Benchmark.
// Benchmarks are parameterized over numeric types to compare CPU (double)
// and GPU (gpuDouble) implementations.

#include <benchmark/benchmark.h>

#include "Complex.hpp"

const int rangeMin = 64 * 1024;
const int rangeMult = 4;
const int rangeMax = 4 * 64 * 1024;

////////////////////////////////////////////////////////
//////////////////////ADDITION//////////////////////////
////////////////////////////////////////////////////////

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
    ->RangeMultiplier(rangeMult)->Range(rangeMin, rangeMax)
    ->Unit(benchmark::kMicrosecond);

// GPU benchmark: Complex<gpuDouble> addition.
// Tests array sizes from 4K to 16K elements, multiplied by 4 each step.
// Only enabled when CUDA compiler is available.
#ifdef CMAKE_CUDA_COMPILER
BENCHMARK_TEMPLATE(bComplexAddition, gpuDouble)
    ->RangeMultiplier(rangeMult)->Range(rangeMin, rangeMax)
    ->Unit(benchmark::kMicrosecond);
#endif

////////////////////////////////////////////////////////
///////////////////MULTIPLICATION///////////////////////
////////////////////////////////////////////////////////

// Benchmarks element-wise multiplication of Complex arrays.
//
// Measures the time to perform in-place multiplication (x *= y) for
// arrays of varying sizes. The size is controlled by state.range(0).
//
// Args:
//   state: Benchmark state containing the array size in range(0).
template <typename T>
void bComplexMultiplication(benchmark::State& state) {  // NOLINT
    Complex<T> x(state.range(0)), y(state.range(0));
    for (auto _ : state)
        x *= y;
}

// CPU benchmark: Complex<double> multiplication.
// Tests array sizes from 4K to 16K elements, multiplied by 4 each step.
BENCHMARK_TEMPLATE(bComplexMultiplication, double)
    ->RangeMultiplier(rangeMult)->Range(rangeMin, rangeMax)
    ->Unit(benchmark::kMicrosecond);

// GPU benchmark: Complex<gpuDouble> multiplication.
// Tests array sizes from 4K to 16K elements, multiplied by 4 each step.
// Only enabled when CUDA compiler is available.
#ifdef CMAKE_CUDA_COMPILER
BENCHMARK_TEMPLATE(bComplexMultiplication, gpuDouble)
    ->RangeMultiplier(rangeMult)->Range(rangeMin, rangeMax)
    ->Unit(benchmark::kMicrosecond);
#endif

////////////////////////////////////////////////////////
////////////////SCALAR MULTIPLICATION///////////////////
////////////////////////////////////////////////////////

// Benchmarks scalar multiplication of Complex arrays.
//
// Measures the time to perform in-place multiplication (x *= a) for
// arrays of varying sizes. The size is controlled by state.range(0).
//
// Args:
//   state: Benchmark state containing the array size in range(0).
template <typename T>
void bComplexScalarMultiplication(benchmark::State& state) {  // NOLINT
    Complex<T> x(state.range(0));
    std::complex<double> a(state.range(0));
    for (auto _ : state)
        x *= a;
}

// CPU benchmark: Complex<double> multiplication.
// Tests array sizes from 4K to 16K elements, multiplied by 4 each step.
BENCHMARK_TEMPLATE(bComplexScalarMultiplication, double)
    ->RangeMultiplier(rangeMult)->Range(rangeMin, rangeMax)
    ->Unit(benchmark::kMicrosecond);

// GPU benchmark: Complex<gpuDouble> multiplication.
// Tests array sizes from 4K to 16K elements, multiplied by 4 each step.
// Only enabled when CUDA compiler is available.
#ifdef CMAKE_CUDA_COMPILER
BENCHMARK_TEMPLATE(bComplexScalarMultiplication, gpuDouble)
    ->RangeMultiplier(rangeMult)->Range(rangeMin, rangeMax)
    ->Unit(benchmark::kMicrosecond);
#endif
