# mpc_python

I keep here my (old) notebooks on Model Predictive Control for path-following problems. Includes a Pybullet simulation to demo the controller. 

This mainly uses **[CVXPY](https://www.cvxpy.org/)** as a framework. This repo contains code from other projecs, check them out in the special thanks section.

## Contents

### Jupyter Notebooks

1. State space model derivation -> analytical and numerical derivaion of the model

2. MPC -> implementation and testing of various tweaks/improvements

3. Obstacle Avoidance -> Using halfplane constrains to avaoid track collisions -> Sill **work in progress**!

### Results

Racing car model is from: *https://github.com/erwincoumans/pybullet_robots*.

![](img/f10.png)

Results:

![](img/demo_bullet.gif)

![](img/demo.gif)

The settings used for tuning the MPC controller are in the **[mpc_config](./mpc_pybullet_demo/mpcpy/mpc_config.py)** class.

### Usage

The results above can be reproduced via Conda or Nix.

#### Conda Environment

The environment used for this project can be repoduced via [conda](https://www.anaconda.com/products/distribution):
```bash
conda env create -f env.yml
conda activate simulation
```

The demos can be run with:
```
python3 mpc_demo_pybullet.py
python3 mpc_demo_nosim.py
```

This environment also includes `jupyter lab` to experiment with the jupyter notebooks.

#### Nix Flake ❄️

A [Nix flake](flake.nix) is provided for a reproducible development shell:

```bash
nix develop --impure        # default with extra dev tools
nix develop .#demo --impure # minimal shell just to run the demo
```

GUI demos require `nixGL` (auto-detects Intel/AMD/NVIDIA GPU):

```bash
nixGL python mpc_pybullet_demo/mpc_demo_pybullet.py
```

Headless demos run without it:

```bash
python mpc_pybullet_demo/mpc_demo_nosim.py
```

## References & Special Thanks :star: :
* [Prof. Borrelli - mpc papers and material](https://borrelli.me.berkeley.edu/pdfpub/IV_KinematicMPC_jason.pdf)
* [AtsushiSakai - pythonrobotics](https://github.com/AtsushiSakai/PythonRobotics/)
* [erwincoumans - pybullet](https://pybullet.org/wordpress/)
* [alexliniger - mpcc](https://github.com/alexliniger/MPCC) and his [paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/oca.2123)
* [arex18 - rocket-lander](https://github.com/arex18/rocket-lander)
