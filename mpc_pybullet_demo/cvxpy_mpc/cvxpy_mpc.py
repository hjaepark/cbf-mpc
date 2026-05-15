import cvxpy as opt
import numpy as np

from .vehicle_model import VehicleModel

np.seterr(divide="ignore", invalid="ignore")


class MPC:
    def __init__(
        self,
        vehicle: VehicleModel,
        T: float,
        DT: float,
        state_cost: list,
        final_state_cost: list,
        input_cost: list,
        input_rate_cost: list,
    ):
        """

        Args:
            vehicle ():
            T ():
            DT ():
            state_cost ():
            final_state_cost ():
            input_cost ():
            input_rate_cost ():
        """
        self.nx = 4  # number of state vars
        self.nu = 2  # umber of input/control vars

        if len(state_cost) != self.nx:
            raise ValueError(f"State Error cost matrix shuld be of size {self.nx}")
        if len(final_state_cost) != self.nx:
            raise ValueError(f"End State Error cost matrix shuld be of size {self.nx}")
        if len(input_cost) != self.nu:
            raise ValueError(f"Control Effort cost matrix shuld be of size {self.nu}")
        if len(input_rate_cost) != self.nu:
            raise ValueError(
                f"Control Effort Difference cost matrix shuld be of size {self.nu}"
            )

        self.vehicle = vehicle
        self.dt = DT
        self.control_horizon = int(T / DT)
        self.Q = np.diag(state_cost)
        self.Qf = np.diag(final_state_cost)
        self.R = np.diag(input_cost)
        self.P = np.diag(input_rate_cost)

    def compute_linear_model_matrices(self, x_bar: list, u_bar: list):
        """
        Computes the approximated LTI state space model $x' = Ax + Bu + C$
        Check out 1.0-lti-system-modelling.ipynb for more details

        Args:
            x_bar (array-like): state space equilibrium point
            u_bar (array-like): control input equilibrium point

        Returns:
            A_lin (np.ndarray): nx*nx matrix
            B_lin (np.ndarray): nx*nu matrix
            C_lin (np.ndarray): nx*1 matrix
        """

        v = x_bar[2]
        theta = x_bar[3]

        a = u_bar[0]
        delta = u_bar[1]

        ct = np.cos(theta)
        st = np.sin(theta)
        cd = np.cos(delta)
        td = np.tan(delta)

        A = np.zeros((self.nx, self.nx))
        A[0, 2] = ct
        A[0, 3] = -v * st
        A[1, 2] = st
        A[1, 3] = v * ct
        A[3, 2] = v * td / self.vehicle.wheelbase
        A_lin = np.eye(self.nx) + self.dt * A

        B = np.zeros((self.nx, self.nu))
        B[2, 0] = 1
        B[3, 1] = v / (self.vehicle.wheelbase * cd**2)
        B_lin = self.dt * B

        f_xu = np.array([v * ct, v * st, a, v * td / self.vehicle.wheelbase]).reshape(
            self.nx, 1
        )
        C_lin = (
            self.dt
            * (
                f_xu
                - np.dot(A, x_bar.reshape(self.nx, 1))
                - np.dot(B, u_bar.reshape(self.nu, 1))
            ).flatten()
        )
        return A_lin, B_lin, C_lin

    def step(
        self,
        initial_state: list,
        target: list,
        prev_cmd: list,
        verbose: bool = False,
    ):
        """

        Args:
            initial_state (array-like): current estimate of [x, y, v, heading]
            target (ndarray): state space reference, in the same frame as the provided current state
            prev_cmd (array-like): previous [acceleration, steer]. note this is used in bounds and has to be realistic.
            verbose (bool):

        Returns:
            x (array-like): predicted optimal state trajectory of size nx * K+1
            u (array-like): predicted optimal control sequence of size nu * K

        """
        assert len(initial_state) == self.nx
        assert len(prev_cmd) == self.nu
        assert target.shape == (self.nx, self.control_horizon)

        # Create variables needed for setting up cvxpy problem
        x = opt.Variable((self.nx, self.control_horizon + 1), name="states")
        u = opt.Variable((self.nu, self.control_horizon), name="actions")
        cost = 0
        constr = []

        # NOTE: here the state linearization is performed around the starting condition to simplify the controller.
        # This approximation gets more inaccurate as the controller looks at the future.
        # To improve performance we can keep track of previous optimized x, u and compute these matrices for each timestep k
        # Ak, Bk, Ck = self.compute_linear_model_matrices(x_prev[:,k], u_prev[:,k])
        # A, B, C = self.compute_linear_model_matrices(initial_state, prev_cmd)

        # Tracking error cost
        for k in range(self.control_horizon):
            # Linearize around the state at timestep k
            # x_bar is approximated as the target state Feedback (Linearization along a Reference).
            # u_bar can be approximated as zero or (better) a feedforward hold
            x_bar = target[:, k]
            u_bar = np.array([0.0, 0.0])
            # TODO: Iterative MPC (iMPC).
            #  Instead of linearising based on the track, take the optimal trajectory calculated by the MPC in the previous control cycle
            #  shift it forward by one timestep, and use that predicted trajectory as the linearization baseline (hence ITERATIVE).
            #
            # It looks like this:
            # 1. Take predicted trajectory from the last frame.
            # 2. Linearize physics around THAT trajectory (not the path).
            # 3. Solve optimization.
            # 4. Save the new output trajectory to use for linearization in the next frame.

            A_k, B_k, C_k = self.compute_linear_model_matrices(x_bar, u_bar)

            # Kinematics constrains
            constr += [x[:, k + 1] == A_k @ x[:, k] + B_k @ u[:, k] + C_k]

            # (NAIVE) Tracking error cost
            # cost += opt.quad_form(x[:, k + 1] - target[:, k], self.Q)

            # Tracking raw XY is not the best for cruising vehicles...
            # we care more about the **cross-track-error**: how far to the side I am from the path.
            # But how we can achive this and keep the simple to understand kinematics?
            # Solution: dynamically rotate the cost matrix Q at every single step of the horizon
            # to align with the direction of the road at that specific point.

            # Extract underlying base costs from diagonal components
            # [q_along_track, q_cross_track, q_vel, q_heading]
            q_along_track = self.Q[0, 0]
            q_cross_track = self.Q[1, 1]
            q_v = self.Q[2, 2]
            q_theta = self.Q[3, 3]

            # Rotation matrix mapping local (path-aligned) errors to global(world xy) coordinates
            theta_ref = x_bar[3]
            ct = np.cos(theta_ref)
            st = np.sin(theta_ref)
            R = np.array([[ct, -st], [st, ct]])

            # We want to minimize local cost:  Error_local^T * Q_local * Error_local
            # Since Global_Error = R * Local_Error, then Local_Error = R^T * Global_Error.
            #
            # So the equivalent global weight matrix is: Q_global = R * Q_local * R^T
            # Compute custom global position weights for this specific path segment
            Q_pos_local = np.array([[q_along_track, 0.0], [0.0, q_cross_track]])
            Q_pos_global = R @ Q_pos_local @ R.T

            # Reassemble the full 4x4 Q_k matrix for this horizon step (velocity and heading dont change)
            Q_k = np.zeros((self.nx, self.nx))
            Q_k[0:2, 0:2] = Q_pos_global
            Q_k[2, 2] = q_v
            Q_k[3, 3] = q_theta

            cost += opt.quad_form(x[:, k + 1] - target[:, k], Q_k)

            # Actuation magnitude cost
            cost += opt.quad_form(u[:, k], self.R)

            constr += [opt.abs(u[0, k]) <= self.vehicle.max_acc]
            constr += [opt.abs(u[1, k]) <= self.vehicle.max_steer]

        # Final point tracking cost
        # TODO: this should also rotate
        # cost += opt.quad_form(x[:, -1] - target[:, -1], self.Qf)

        # initial state
        constr += [x[:, 0] == initial_state]

        # Actuation rate of change bounds (step 0 uses last cmd)
        constr += [opt.abs(u[0, 0] - prev_cmd[0]) / self.dt <= self.vehicle.max_d_acc]
        constr += [opt.abs(u[1, 0] - prev_cmd[1]) / self.dt <= self.vehicle.max_d_steer]
        for k in range(1, self.control_horizon):
            constr += [
                opt.abs(u[0, k] - u[0, k - 1]) / self.dt <= self.vehicle.max_d_acc
            ]
            constr += [
                opt.abs(u[1, k] - u[1, k - 1]) / self.dt <= self.vehicle.max_d_steer
            ]

        prob = opt.Problem(opt.Minimize(cost), constr)
        prob.solve(solver=opt.OSQP, warm_start=True, verbose=False)
        return np.array(x.value), np.array(u.value)
