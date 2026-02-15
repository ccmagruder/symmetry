// Copyright 2022 Caleb Magruder

#include <cuComplex.h>
#include <cstdint>
#include <iostream>

#include "FPI.hpp"

__global__ void fpi_seed_kernel(cuDoubleComplex* z, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double angle = 2.0 * 3.14159265358979323846 * tid / N;
    z[tid] = make_cuDoubleComplex(
        0.1 * cos(angle), -0.12 + 0.1 * sin(angle));
}

__global__ void fpi_renorm_kernel(cuDoubleComplex* z, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    double a = cuCabs(z[tid]);
    if (a > 8.0) {
        double s = 3.0 / a;
        z[tid].x *= s;
        z[tid].y *= s;
    }
}

__global__ void fpi_noise_kernel(cuDoubleComplex* z, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int sgn_x = (0.0 < z[tid].x) - (z[tid].x < 0.0);
    int sgn_y = (0.0 < z[tid].y) - (z[tid].y < 0.0);
    z[tid].x = z[tid].x * 0.99 - 1e-2 * sgn_x;
    z[tid].y = z[tid].y * 0.99 - 1e-2 * sgn_y;
}

__global__ void fpi_nudge_kernel(cuDoubleComplex* z, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    if (fabs(z[tid].x) < 1e-15) z[tid].x = 0.001;
    if (fabs(z[tid].y) < 1e-15) z[tid].y = 0.001;
}

__global__ void fpi_heatmap_kernel(
    const cuDoubleComplex* z, int count,
    unsigned long long* heatmap, int rows, int cols, double scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    double size = sqrt((double)(rows * cols));
    int c = (int)floor(scale * size / 2.0 * z[tid].x + cols / 2.0);
    int r = (int)floor(scale * size / 2.0 * z[tid].y + rows / 2.0);
    if (r >= 0 && r < rows && c >= 0 && c < cols) {
        atomicAdd(&heatmap[r * cols + c], 1ULL);
    }
}

template<>
void FPI<gpuDouble>::run_fpi(uint64_t niter) {
    const int N = 65536;
    int rows = static_cast<int>(this->_rows);
    int cols = static_cast<int>(this->_cols);

    size_t heatmap_bytes = rows * cols * sizeof(unsigned long long);
    unsigned long long* d_heatmap = nullptr;
    cudaMalloc(&d_heatmap, heatmap_bytes);
    cudaMemset(d_heatmap, 0, heatmap_bytes);

    Complex<gpuDouble> z(N);
    Complex<gpuDouble> znm1(N);
    Complex<gpuDouble> bracket(N);
    Complex<gpuDouble> alphaAbsSq(N);
    Complex<gpuDouble> betaReZn(N);
    Complex<gpuDouble> deltaCosArg(N);
    Complex<gpuDouble> abs_copy(N);
    Complex<gpuDouble> gammaConjZnm1(N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cuDoubleComplex s_alpha = make_cuDoubleComplex(this->_alpha, 0);
    cuDoubleComplex s_beta = make_cuDoubleComplex(this->_beta, 0);
    cuDoubleComplex s_delta = make_cuDoubleComplex(this->_delta, 0);
    cuDoubleComplex s_gamma = make_cuDoubleComplex(this->_gamma, 0);
    cuDoubleComplex s_np = make_cuDoubleComplex(this->_n * this->_p, 0);

    // Seed orbits on a circle
    fpi_seed_kernel<<<blocks, threads>>>(
        reinterpret_cast<cuDoubleComplex*>(z.dptr()), N);

    // Warmup transient
    int warmup = 100 * static_cast<int>(this->_init_iter);
    for (int i = 0; i < warmup; i++) {
        this->F(z, znm1, bracket, alphaAbsSq, betaReZn,
                deltaCosArg, abs_copy, gammaConjZnm1,
                s_alpha, s_beta, s_delta, s_gamma, s_np);
        fpi_renorm_kernel<<<blocks, threads>>>(
            reinterpret_cast<cuDoubleComplex*>(z.dptr()), N);
    }

    // Main iteration loop
    // Each step advances all N orbits and accumulates N points.
    uint64_t num_steps = (niter + N - 1) / N;
    uint64_t total_accumulated = 0;

    for (uint64_t step = 0; step < num_steps; step++) {
        this->F(z, znm1, bracket, alphaAbsSq, betaReZn,
                deltaCosArg, abs_copy, gammaConjZnm1,
                s_alpha, s_beta, s_delta, s_gamma, s_np);
        fpi_renorm_kernel<<<blocks, threads>>>(
            reinterpret_cast<cuDoubleComplex*>(z.dptr()), N);

        // Noise perturbation every 1000 steps
        if (this->_add_noise && (step % 1000 == 0) && step > 0) {
            fpi_noise_kernel<<<blocks, threads>>>(
                reinterpret_cast<cuDoubleComplex*>(z.dptr()), N);
            for (int j = 0; j < this->_init_iter; j++) {
                this->F(z, znm1, bracket, alphaAbsSq, betaReZn,
                        deltaCosArg, abs_copy, gammaConjZnm1,
                        s_alpha, s_beta, s_delta, s_gamma, s_np);
                fpi_renorm_kernel<<<blocks, threads>>>(
                    reinterpret_cast<cuDoubleComplex*>(z.dptr()), N);
            }
        }

        // Axis nudge
        fpi_nudge_kernel<<<blocks, threads>>>(
            reinterpret_cast<cuDoubleComplex*>(z.dptr()), N);

        // Heatmap accumulation (only accumulate up to niter total points)
        uint64_t points_this_step = std::min(
            static_cast<uint64_t>(N), niter - total_accumulated);
        fpi_heatmap_kernel<<<blocks, threads>>>(
            reinterpret_cast<const cuDoubleComplex*>(z.dptr()),
            static_cast<int>(points_this_step),
            d_heatmap, rows, cols, this->_param.scale);
        total_accumulated += points_this_step;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(this->_data, d_heatmap, heatmap_bytes,
               cudaMemcpyDeviceToHost);
    cudaFree(d_heatmap);
}
