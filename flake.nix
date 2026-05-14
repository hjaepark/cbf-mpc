{
  description = "MPC Python Demo - Model Predictive Control";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nixgl.url = "github:nix-community/nixGL";  # GPU support
  };

  outputs = { self, nixpkgs, flake-utils, nixgl }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        nixgl-pkg = nixgl.packages.${system};

        # --- Python packages ---
        python-with-packages = pkgs.python3.withPackages (ps: with ps; [
          numpy
          matplotlib
          pybullet
          cvxpy
          scipy
          osqp
          # e.g. jupyterlab, ipython, black, ruff, pytest, mypy, ...
        ]);

      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python-with-packages
            nixgl-pkg.nixGLDefault

            # --- Additional binaries ---
            # e.g. pkgs.cmake, pkgs.gdb, pkgs.htop, pkgs.just, ...
          ];

          shellHook = ''
            # --- Custom shell stuff ---
            # e.g. export PATH="$PWD/scripts:$PATH"
            #      source .venv/bin/activate
            #      alias lint='ruff check .'
          '';
        };
      });
}
