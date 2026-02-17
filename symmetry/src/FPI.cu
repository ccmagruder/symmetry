// Copyright 2026 Caleb Magruder
//
// GPU (gpuDouble) template specializations for FPI per-element helpers.
// Each specialization wraps a CUDA kernel launched over _N threads.
// The kernels themselves are defined as __global__ functions above
// the specializations.

#include <cuComplex.h>
#include <cstdint>

#include "FPI.hpp"

// --- CUDA kernels ---

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
    uint64_t* heatmap, int rows, int cols, double scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    double size = sqrt((double)(rows * cols));
    int c = (int)floor(scale * size / 2.0 * z[tid].x + cols / 2.0);
    int r = (int)floor(scale * size / 2.0 * z[tid].y + rows / 2.0);
    if (r >= 0 && r < rows && c >= 0 && c < cols) {
        atomicAdd(reinterpret_cast<unsigned long long*>(
            &heatmap[r * cols + c]), 1ULL);
    }
}

// --- gpuDouble template specializations ---

template<>
void FPI<gpuDouble>::seed(Complex<gpuDouble>& z) {
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    fpi_seed_kernel<<<blocks, threads>>>(
        reinterpret_cast<cuDoubleComplex*>(z.dptr()), this->_N);
}

template<>
void FPI<gpuDouble>::noise(Complex<gpuDouble>& z) {
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    fpi_noise_kernel<<<blocks, threads>>>(
        reinterpret_cast<cuDoubleComplex*>(z.dptr()), this->_N);
}

template<>
void FPI<gpuDouble>::renorm(Complex<gpuDouble>& z) {
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    fpi_renorm_kernel<<<blocks, threads>>>(
        reinterpret_cast<cuDoubleComplex*>(z.dptr()), this->_N);
}

template<>
void FPI<gpuDouble>::nudge(Complex<gpuDouble>& z) {
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    fpi_nudge_kernel<<<blocks, threads>>>(
        reinterpret_cast<cuDoubleComplex*>(z.dptr()), this->_N);
}

// Allocates and zeroes a device-side heatmap buffer before iteration.
template<>
void FPI<gpuDouble>::pre_run() {
    int rows = static_cast<int>(this->rows());
    int cols = static_cast<int>(this->cols());
    size_t heatmap_bytes = rows * cols * sizeof(uint64_t);
    cudaMalloc(&_d_heatmap, heatmap_bytes);
    cudaMemset(_d_heatmap, 0, heatmap_bytes);
}

// Synchronizes the device, copies the heatmap to host _data, and frees
// the device buffer.
template<>
void FPI<gpuDouble>::post_run() {
    int rows = static_cast<int>(this->rows());
    int cols = static_cast<int>(this->cols());
    size_t heatmap_bytes = rows * cols * sizeof(uint64_t);
    cudaDeviceSynchronize();
    cudaMemcpy(this->data(), this->_d_heatmap, heatmap_bytes,
               cudaMemcpyDeviceToHost);
    cudaFree(_d_heatmap);
    _d_heatmap = nullptr;
}

template<>
void FPI<gpuDouble>::accumulate(const Complex<gpuDouble>& z, uint64_t points) {
    int threads = 256;
    int blocks = (this->_N + threads - 1) / threads;
    fpi_heatmap_kernel<<<blocks, threads>>>(
        reinterpret_cast<const cuDoubleComplex*>(z.dptr()),
        static_cast<int>(points),
        _d_heatmap, static_cast<int>(this->rows()),
        static_cast<int>(this->cols()), this->_param.scale);
}


