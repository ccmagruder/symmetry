# default.nix
{ lib, stdenv, cmake, cudaPackages, gtest, gbenchmark, makeWrapper, json }:

stdenv.mkDerivation {
  pname = "symmetry";
  version = "0.0.0";

  src = symmetry/.;

  requiredSystemFeatures = [ "cuda" ];
  __noChroot = true;

  nativeBuildInputs = [
    cmake
    cudaPackages.cuda_nvcc
    makeWrapper
  ];

  buildInputs = [
    cudaPackages.cuda_cudart
    cudaPackages.libcublas
    gtest
    gbenchmark
    json
  ];

  doCheck = true;

  checkPhase = ''
    runHook preCheck
    export LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver/lib64:$LD_LIBRARY_PATH"

    ls -l /run/opengl-driver/lib/libcuda.so.1 || echo "❌ Still missing libcuda in sandbox!"
  
    if [ ! -L /run/opengl-driver ]; then
      echo "❌ ERROR: /run/opengl-driver is missing in sandbox. Check nix.conf extra-sandbox-paths."
    fi

    ctest --output-on-failure
    runHook postCheck
  '';

  postFixup = ''
    for bin in $out/bin/*; do
      wrapProgram "$bin" \
        --prefix LD_LIBRARY_PATH : "/run/opengl-driver/lib"
    done
  '';

  meta = with lib; {
    description = "A CUDA Accelerated Chaos in Symmetry Image Generator";
    license = licenses.unfree; # CUDA requires this
    platforms = platforms.linux;
  };
}
