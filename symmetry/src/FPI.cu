// Copyright 2022 Caleb Magruder

#include <cuComplex.h>
#include <cstdint>
#include <iostream>

#include "FPI.hpp"

// Device-side sign function: returns -1, 0, or +1.
__device__ int device_sgn(double val) {
    return (0.0 < val) - (val < 0.0);
}

// Device-side implementation of the chaotic map F(z).
//
// Mirrors the CPU FPI::F(z) using CUDA math intrinsics and cuDoubleComplex.
// Evaluates:
//   F(z) = (mu + alpha*|z|^2 + beta*Re(z^n)
//           + delta*cos(n*p*arg(z))*|z|) * z + gamma*conj(z)^{n-1}
__device__ cuDoubleComplex device_F(
    cuDoubleComplex z,
    double lambda, double omega, double alpha, double beta,
    double delta, double gamma, double n, double p) {

    if (isnan(z.x)) return make_cuDoubleComplex(0.0, 0.0);

    int ni = __double2int_rn(n);

    // Compute z^{n-1}
    cuDoubleComplex znm1 = z;
    for (int i = 1; i < ni - 1; i++) {
        znm1 = cuCmul(znm1, z);
    }

    // Term 1: mu = lambda + i*omega
    cuDoubleComplex mu = make_cuDoubleComplex(lambda, omega);

    // Term 2: alpha * |z|^2
    double zabs = cuCabs(z);
    double alphaZAbsSqr = alpha * zabs * zabs;

    // Term 3: beta * Re(z^n)
    cuDoubleComplex zn = cuCmul(znm1, z);
    double betaReZn = beta * zn.x;

    // Term 4: delta * cos(n*p*arg(z)) * |z|
    double argz = atan2(z.y, z.x);
    double deltaCosArgAbs = delta * cos(n * p * argz) * zabs;

    // Sum the bracket and multiply by z
    cuDoubleComplex bracket = make_cuDoubleComplex(
        mu.x + alphaZAbsSqr + betaReZn + deltaCosArgAbs,
        mu.y);
    cuDoubleComplex result = cuCmul(bracket, z);

    // Term 5: gamma * conj(z)^{n-1}
    cuDoubleComplex conjz = cuConj(z);
    cuDoubleComplex conjznm1 = conjz;
    for (int i = 1; i < ni - 1; i++) {
        conjznm1 = cuCmul(conjznm1, conjz);
    }
    result.x += gamma * conjznm1.x;
    result.y += gamma * conjznm1.y;

    // Renormalize if orbit diverges
    double absResult = cuCabs(result);
    if (absResult > 8.0) {
        double s = 3.0 / absResult;
        result.x *= s;
        result.y *= s;
    }

    return result;
}

// GPU kernel: each thread runs an independent orbit of F(z) and accumulates
// visits into a shared heatmap via atomicAdd.
__global__ void fpi_kernel(
    unsigned long long* d_heatmap,
    int rows, int cols, double scale,
    double lambda, double omega, double alpha, double beta,
    double delta, double gamma, double n, double p,
    uint64_t iters_per_thread, uint64_t total_threads,
    uint64_t niter, int init_iter, bool add_noise) {

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    // Seed each thread on a small circle for orbit diversity
    double angle = 2.0 * 3.14159265358979323846 * tid / total_threads;
    cuDoubleComplex z = make_cuDoubleComplex(
        0.1 * cos(angle), -0.12 + 0.1 * sin(angle));

    // Warmup transient to reach the attractor
    int warmup = 100 * init_iter;
    for (int i = 0; i < warmup; i++) {
        z = device_F(z, lambda, omega, alpha, beta, delta, gamma, n, p);
    }

    // This thread's share of iterations (remainder distributed to first threads)
    uint64_t my_iters = iters_per_thread
        + (tid < (niter % total_threads) ? 1 : 0);

    double size = sqrt((double)(rows * cols));

    for (uint64_t i = 0; i < my_iters; i++) {
        z = device_F(z, lambda, omega, alpha, beta, delta, gamma, n, p);

        // Perturb every 1000 iterations to break periodic cycles
        if (add_noise && (i % 1000 == 0)) {
            z.x = z.x * 0.99 - 1e-2 * device_sgn(z.x);
            z.y = z.y * 0.99 - 1e-2 * device_sgn(z.y);
            for (int j = 0; j < init_iter; j++)
                z = device_F(z, lambda, omega,
                             alpha, beta, delta, gamma, n, p);
        }

        // Nudge off real axis if orbit collapses
        if (fabs(z.x) < 1e-15) {
            z.x = 0.001;
            for (int j = 0; j < init_iter; j++)
                z = device_F(z, lambda, omega,
                             alpha, beta, delta, gamma, n, p);
        }

        // Nudge off imaginary axis if orbit collapses
        if (fabs(z.y) < 1e-15) {
            z.y = 0.001;
            for (int j = 0; j < init_iter; j++)
                z = device_F(z, lambda, omega,
                             alpha, beta, delta, gamma, n, p);
        }

        // Map iterate to pixel coordinates and accumulate
        int c = (int)floor(scale * size / 2.0 * z.x + cols / 2.0);
        int r = (int)floor(scale * size / 2.0 * z.y + rows / 2.0);
        if (r >= 0 && r < rows && c >= 0 && c < cols) {
            atomicAdd(&d_heatmap[r * cols + c], 1ULL);
        }
    }
}

template<>
void FPI<gpuDouble>::run_fpi(uint64_t niter) {
    int rows = static_cast<int>(this->_rows);
    int cols = static_cast<int>(this->_cols);
    size_t heatmap_bytes = rows * cols * sizeof(unsigned long long);

    // Allocate and zero device heatmap
    unsigned long long* d_heatmap = nullptr;
    cudaMalloc(&d_heatmap, heatmap_bytes);
    cudaMemset(d_heatmap, 0, heatmap_bytes);

    // Launch configuration: 256 blocks x 256 threads = 65,536 parallel orbits
    int threads = 256;
    int blocks = 256;
    uint64_t total_threads = threads * blocks;
    uint64_t iters_per_thread = niter / total_threads;

    fpi_kernel<<<blocks, threads>>>(
        d_heatmap,
        rows, cols, this->_param.scale,
        this->_lambda, this->_omega, this->_alpha, this->_beta,
        this->_delta, this->_gamma, this->_n, this->_p,
        iters_per_thread, total_threads,
        niter, this->_init_iter, this->_add_noise);

    cudaDeviceSynchronize();

    // Copy device heatmap to host Image::_data
    cudaMemcpy(this->_data, d_heatmap, heatmap_bytes,
               cudaMemcpyDeviceToHost);
    cudaFree(d_heatmap);
}
