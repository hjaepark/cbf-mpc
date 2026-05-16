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
        # core ones for running the demo, don't touch these
        core-python-pkgs = ps: with ps; [
          numpy
          matplotlib
          mujoco
          cvxpy
          scipy
        ];

        # dev packages, add extra stuff here
        dev-python-pkgs = ps: with ps; [
            jupyterlab
            ipython
            ipywidgets
            nbformat
            nbconvert
            black
            ruff
            osqp
            casadi
          ];

        python-demo = pkgs.python3.withPackages core-python-pkgs;

        python-dev = pkgs.python3.withPackages (ps:
          (core-python-pkgs ps) ++ (dev-python-pkgs ps)
        );

        # --- System binaries ---
        dev-bin-pkgs = with pkgs; [
          # add here extra CLI tools for the dev shell
          # cmake, gdb, htop, just, ...
        ];

      in
      {
        devShells = {
          demo = pkgs.mkShell {
            buildInputs = [
              python-demo
              nixgl-pkg.nixGLDefault
            ];

            shellHook = ''
              echo "demo shell — bare deps to run the sim"
              echo ""
              echo "Run with GUI (MuJoCo):"
              echo "  nixGL python mpc_python/mpc_demo_mujoco.py"
              echo ""
              echo "Run without GUI:"
              echo "  python mpc_python/mpc_demo_nosim.py"
            '';
          };

          default = pkgs.mkShell {
            buildInputs = [
              python-dev
              nixgl-pkg.nixGLDefault
            ] ++ dev-bin-pkgs;

            shellHook = ''
              echo "full dev shell — deps + jupyter + dev tools"
              echo ""
              echo "Run with GUI (MuJoCo):"
              echo "  nixGL python mpc_python/mpc_demo_mujoco.py"
              echo ""
              echo "Run without GUI:"
              echo "  python mpc_python/mpc_demo_nosim.py"
              echo ""
              echo "Notebooks: jupyter lab"
              echo "Lint: ruff check ."
              echo "Format: black ."
            '';
          };
        };
      });
}
