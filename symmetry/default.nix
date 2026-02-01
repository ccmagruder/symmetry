# default.nix
{ lib, stdenv, cmake, cudaPackages, gtest, json, gbenchmark }:

stdenv.mkDerivation {
  pname = "symmetry";
  version = "0.0.0";

  src = ./.;

  requiredSystemFeatures = [ "cuda" ];
  __noChroot = true;

  nativeBuildInputs = [
    cmake
    cudaPackages.cuda_nvcc
  ];

  buildInputs = [
    cudaPackages.cuda_cudart
    cudaPackages.libcublas
    gtest
    json
    gbenchmark
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DCMAKE_CUDA_COMPILER=${cudaPackages.cuda_nvcc}/bin/nvcc"
  ];

  doCheck = true;

  checkPhase = ''
    runHook preCheck
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver/lib64:$LD_LIBRARY_PATH"

    ls -l /run/opengl-driver/lib/libcuda.so.1 || echo "❌ Still missing libcuda in sandbox!"

    if [ ! -L /run/opengl-driver ]; then
      echo "❌ ERROR: /run/opengl-driver is missing in sandbox. Check nix.conf extra-sandbox-paths."
    fi

    ctest --test-dir . --output-on-failure
    runHook postCheck
  '';

  installPhase = ''
    runHook preInstall
    cmake --install . --prefix $out
    runHook postInstall
  '';


  meta = with lib; {
    description = "A CUDA Symmetry In Chaos Image Generator";
    license = licenses.unfree; # CUDA requires this
    platforms = platforms.linux;
  };
}
