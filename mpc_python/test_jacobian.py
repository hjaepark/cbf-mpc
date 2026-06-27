import numpy as np
from cvxpy_mpc.cvxpy_mpc import MPC

L = 0.2965  # wheelbase, matches config

# Continuous-time kinematic bicycle dynamics
def f_true(state, u):
    x, y, v, th = state
    a, delta = u
    return np.array([v * np.cos(th), v * np.sin(th), a, v * np.tan(delta) / L])

# Strip the (I + dt*A) discretization to recover continuous A.
def continuous_A(mpc, x_bar, u_bar):
    A_lin, _, _ = mpc._compute_linear_model_matrices(x_bar, u_bar)
    return (A_lin - np.eye(4)) / mpc.dt


mpc = MPC("config/mpc.yaml")
x_bar = np.array([0.0, 0.0, 1.2, 0.6])
u_bar = np.array([0.3, 0.25])
A = continuous_A(mpc, x_bar, u_bar)

print("A[3,2] =", A[3, 2], " | correct tan(d)/L =", np.tan(u_bar[1]) / L)

print(f"{'step':>10} {'residual':>14}")
for step in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
    dx = np.array([0.0, 0.0, step, 0.0])
    lhs = f_true(x_bar + dx, u_bar) - f_true(x_bar, u_bar)
    print(f"{step:>10.0e} {np.linalg.norm(lhs - A @ dx):>14.3e}")