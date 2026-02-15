# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

The project uses Nix flakes for environment management. The shellHook execs into zsh, so programmatic commands require:
```bash
bash -c 'eval "$(nix print-dev-env /home/remote/symmetry 2>/dev/null | grep -v "exec.*zsh")" && <command>'
```

Within the nix shell:
```bash
# Clean build
rm -rf symmetry/build && cmake -G Ninja -B symmetry/build symmetry -DCMAKE_BUILD_TYPE=Release && cmake --build symmetry/build

# Run all tests
ctest --test-dir symmetry/build --output-on-failure

# Run a single test suite (e.g., tFPI)
ctest --test-dir symmetry/build -R tFPI --output-on-failure

# Run benchmark
symmetry/build/tests/bComplex

# Generate an attractor image
symmetry/build/symmetry run symmetry/config/fig13-7.json output.pgm
```

## Architecture

Generates chaotic attractor visualizations from "Symmetry in Chaos" (Field & Golubitsky). The core equation iterated is:

```
F(z) = (λ + α|z|² + β·Re(zⁿ) + δ·cos(n·p·arg(z))·|z|) · z + γ·conj(z)^(n-1)
```

### GPU Dispatch via Tag Types

The `gpuDouble` empty class acts as a template tag. `complex_traits<T>::value_type` maps it to `double`. Template specializations in `.cu` files provide GPU code paths:

- `FPI<double>` → CPU: single orbit, sequential histogram accumulation
- `FPI<gpuDouble>` → GPU: 65,536 parallel orbits, `atomicAdd` histogram accumulation

GPU specializations are forward-declared after class bodies with `#ifdef CMAKE_CUDA_COMPILER` guards (this macro is set by `target_compile_definitions` in `lib/CMakeLists.txt`). Implementations live in `.cu` files.

### Key Class Relationships

- **`FPI<T>` → `Image<uint64_t, 1>`**: FPI inherits from a 64-bit grayscale image used as the histogram buffer. `Type = complex_traits<T>::value_type` extracts the scalar type; `S = std::complex<Type>` is the complex type used for orbit iteration.
- **`Image<T, COLORS>` → `Pixel<T, COLORS>`**: `operator[]` returns a `Pixel` wrapper for pointer-based row/column access. `Pixel::operator++()` increments with overflow checking.
- **`Complex<T>`**: Array of N complex numbers stored as 2N interleaved reals. `Complex<gpuDouble>` manages paired host/device pointers and uses cuBLAS for arithmetic.
- **`CublasHandleSingleton`**: Reference-counted cuBLAS handle shared across `Complex<gpuDouble>` instances.

### CUDA Conditional Compilation

CUDA is auto-detected via `check_language(CUDA)` in CMake. When available, `Complex.cu` and `FPI.cu` are compiled into the library and cuBLAS is linked. The `CMAKE_CUDA_COMPILER` preprocessor macro gates GPU code in headers and tests. Target architecture: sm_89 (Ada Lovelace).

## Test Patterns

Tests use GoogleTest. The `TestFPI` subclass pattern exposes protected FPI state for unit testing by disabling warmup (`_init_iter = 0`), perturbation (`_add_noise = false`), and setting a known start point. Each map term (lambda, omega, alpha, beta, delta, gamma) is tested in isolation.

`tComplex` uses `TYPED_TEST_SUITE` to run the same tests across `float`, `double`, and (when CUDA is available) `gpuDouble`.

## Config Format

JSON files in `symmetry/config/` define map parameters. Test config (`config/test_iter10.json`) uses a 2×2 image with 10 iterations. Production configs go up to 3840×2160 at 2×10¹⁰ iterations.
