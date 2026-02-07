# flake.nix
{
  description = "Professional CUDA C++ Project Entry Point";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    claude-code-nix.url = "github:sadjow/claude-code-nix";
    json.url = "github:ccmagruder/json";
  };

  outputs = { self, nixpkgs, claude-code-nix, json, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # Required for CUDA
        config.cudaSupport = true;
      };
    in {
      # 1. The Package: Allows 'nix build'
      packages.${system}.default = pkgs.callPackage ./default.nix {
        json = json.packages.${system}.default;
      };

      # 2. The Dev Shell: Allows 'nix develop'
      devShells.${system}.default = pkgs.mkShell {
        # 'inputsFrom' pulls all tools and libs from default.nix!
        # No need to repeat gcc, cmake, or gtest here.
        inputsFrom = [ self.packages.${system}.default ];

        buildInputs = with pkgs; [
          claude-code-nix.packages.${system}.claude-code
          clang-tools  # provides clangd
          zsh
        ];

        shellHook = ''
          export CUDA_PATH=${pkgs.cudaPackages.cuda_nvcc}
          export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"
          echo "🚀 Environment ready. Run 'cmake -B build' to begin."
          exec ${pkgs.zsh}/bin/zsh
        '';
      };
    };
}
