// Copyright 2026 Caleb Magruder
//
// Benchmarks for the FPI class template.
//
// Measures performance of run_fpi using Google Benchmark.
// Benchmarks are parameterized over iteration count to compare CPU (cpuDouble)
// and GPU (gpuDouble) implementations.

#include <cstdint>

#include <benchmark/benchmark.h>

#include "FPI.hpp"

// FPI subclass that disables warmup iterations for benchmarking.
template <typename T>
class BenchFPI : public FPI<T> {
 public:
    explicit BenchFPI(Param p) : FPI<T>(p) {
        this->_init_iter = 0;
        this->_enable_pbar = false;
    }
};

// Benchmarks FPI::run_fpi for varying iteration counts.
//
// Uses fig13-7 map coefficients at 256x256 resolution. The iteration
// count is controlled by state.range(0). Warmup is disabled via
// BenchFPI to isolate the iteration loop cost.
//
// Args:
//   state: Benchmark state containing the iteration count in range(0).
template <typename T>
void bFPIRunFPI(benchmark::State& state) {  // NOLINT
    uint64_t niter = static_cast<uint64_t>(state.range(0));
    Param p(1.56, -1.0, 0.1, -0.82, 0, 3, 0, 0, 0.65, niter, 256, 256);
    BenchFPI<T> fpi(p);
    for (auto _ : state) {
        fpi.run_fpi(niter);
    }
}

// CPU benchmark: FPI<cpuDouble> run_fpi.
BENCHMARK_TEMPLATE(bFPIRunFPI, cpuDouble)
    ->RangeMultiplier(4)->Range(65536, 65536 * 16)
    ->Unit(benchmark::kMillisecond);

// GPU benchmark: FPI<gpuDouble> run_fpi.
#ifdef CMAKE_CUDA_COMPILER
BENCHMARK_TEMPLATE(bFPIRunFPI, gpuDouble)
    ->RangeMultiplier(4)->Range(65536, 65536 * 16)
    ->Unit(benchmark::kMillisecond);
#endif
