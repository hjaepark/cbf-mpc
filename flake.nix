{
  description = "MPC Python Demo - Model Predictive Control";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nixgl.url = "github:nix-community/nixGL";  # GPU-accelerated OpenGL wrapper
  };

  outputs = { self, nixpkgs, flake-utils, nixgl }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        nixgl-pkg = nixgl.packages.${system};

        # --- Python packages ---
        # Add/remove packages here (use the short name, e.g. "numpy" not "python3Packages.numpy")
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
            nixgl-pkg.nixGLDefault  # auto-detects Intel/AMD/NVIDIA GPU drivers

            # --- Additional binaries ---
            # e.g. pkgs.cmake, pkgs.gdb, pkgs.htop, pkgs.just, ...
          ];

          shellHook = ''
            # --- Custom shell setup ---
            # e.g. export PATH="$PWD/scripts:$PATH"
            #      source .venv/bin/activate
            #      alias lint='ruff check .'

            echo "MPC Python Demo development shell"
            echo ""
            echo "Note: use 'nix develop --impure' to enter the shell (required for GPU detection)"
            echo ""
            echo "Run with GUI:"
            echo "  nixGL python mpc_pybullet_demo/mpc_demo_pybullet.py"
            echo ""
            echo "Run without GUI:"
            echo "  python mpc_pybullet_demo/mpc_demo_nosim.py"
          '';
        };
      });
}
