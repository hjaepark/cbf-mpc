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

        # CVXPY vars
        self.x = opt.Variable((self.nx, self.control_horizon + 1), name="states")
        self.u = opt.Variable((self.nu, self.control_horizon), name="actions")

        # CVXPY params (placeholder for run-time data)
        self.initial_state_param = opt.Parameter(self.nx, name="x0")
        self.target_param = opt.Parameter(
            (self.nx, self.control_horizon), name="target"
        )
        self.last_cmd_param = opt.Parameter(self.nu, name="last_applied_command")

        self.A_params = [
            opt.Parameter((self.nx, self.nx), name=f"A_{k}")
            for k in range(self.control_horizon)
        ]
        self.B_params = [
            opt.Parameter((self.nx, self.nu), name=f"B_{k}")
            for k in range(self.control_horizon)
        ]
        self.C_params = [
            opt.Parameter(self.nx, name=f"C_{k}") for k in range(self.control_horizon)
        ]
        self.Q_params = [
            opt.Parameter((self.nx, self.nx), PSD=True, name=f"Q_{k}")
            for k in range(self.control_horizon)
        ]
        self.Qf_param = opt.Parameter((self.nx, self.nx), PSD=True, name="Qf")

        # optimised vars
        # used for constrains and IMPC logic
        self.prev_cmd = None
        self.prev_trajectory = None

        # build the problem ONCE
        self.prob = self.make_mpc_problem()

    def compute_linear_model_matrices(self, x_bar: list, u_bar: list):
        """
        Computes the approximated linearised state space model $x' = Ax + Bu + C$
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

        # CVXPY does not like abosolute 0 in params
        if abs(st) < 1e-9:
            st = 1e-9
        if abs(ct) < 1e-9:
            ct = 1e-9
        if abs(cd) < 1e-9:
            cd = 1e-9
        if abs(td) < 1e-9:
            td = 1e-9

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

    def make_mpc_problem(
        self,
    ):

        cost = 0
        constr = []

        # Tracking error cost
        for k in range(self.control_horizon):

            # Kinematics constrains
            # Note each step uses the LTV matrix for that step
            constr += [
                self.x[:, k + 1]
                == self.A_params[k] @ self.x[:, k]
                + self.B_params[k] @ self.u[:, k]
                + self.C_params[k]
            ]

            cost += opt.quad_form(
                self.x[:, k + 1] - self.target_param[:, k], self.Q_params[k]
            )

            # Actuation magnitude cost
            cost += opt.quad_form(self.u[:, k], self.R)

            constr += [opt.abs(self.u[0, k]) <= self.vehicle.max_acc]
            constr += [opt.abs(self.u[1, k]) <= self.vehicle.max_steer]

        # Final point tracking cost
        cost += opt.quad_form(self.x[:, -1] - self.target_param[:, -1], self.Qf_param)

        # Initial state
        constr += [self.x[:, 0] == self.initial_state_param]

        # Actuation rate of change bounds (step 0 uses last cmd)
        constr += [
            opt.abs(self.u[0, 0] - self.last_cmd_param[0]) / self.dt
            <= self.vehicle.max_d_acc
        ]
        constr += [
            opt.abs(self.u[1, 0] - self.last_cmd_param[1]) / self.dt
            <= self.vehicle.max_d_steer
        ]
        for k in range(1, self.control_horizon):
            constr += [
                opt.abs(self.u[0, k] - self.u[0, k - 1]) / self.dt
                <= self.vehicle.max_d_acc
            ]
            constr += [
                opt.abs(self.u[1, k] - self.u[1, k - 1]) / self.dt
                <= self.vehicle.max_d_steer
            ]

        prob = opt.Problem(opt.Minimize(cost), constr)
        return prob

    def step(
        self,
        initial_state: list,
        target: list,
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
        assert target.shape == (self.nx, self.control_horizon)

        # update the parameter values
        self.initial_state_param.value = initial_state
        self.target_param.value = target
        if self.prev_cmd is not None:
            self.last_cmd_param.value = self.prev_cmd[:, 0]
        else:
            self.last_cmd_param.value = np.zeros(self.nu)

        ## compute system matrices
        # Option 1: The state linearization is performed **once** (LTI) around the starting condition to simplify the controller.
        # This approximation gets more inaccurate as the controller looks at the future, as the system changes (a lot!) along the trajectory
        # A, B, C = self.compute_linear_model_matrices(initial_state, prev_cmd)
        # you will see the prediction is MUCH less accurate...

        # Option 2: Feedback Linearization along a Reference.
        # x_bar is approximated as the target state
        # u_bar can be approximated as zero or a feedforward hold

        # option 3: Iterative MPC (iMPC) sort-of.
        # Instead of linearising based on the track, take the optimal trajectory calculated by the MPC in the previous control cycle
        # shift it forward by one timestep, and use that predicted trajectory as the linearization baseline (hence ITERATIVE).
        # Because the vehicle's actual movement matches its own recent predictions much closer than the "ideal raw path", the predicition is further improved
        #
        # It looks like this:
        # 1. Take predicted trajectory from the last frame.
        # 2. Linearize physics around THAT trajectory (not the path).
        # 3. Solve optimization.
        # 4. Save the new output trajectory to use for linearization in the next frame.
        #
        # in real iMPC this preocess is reiterated for the same time step

        # Compute LTV matrices
        for k in range(self.control_horizon):
            if self.prev_trajectory is not None and self.prev_cmd is not None:
                x_bar = self.prev_trajectory[:, k + 1]
                u_bar = (
                    self.prev_cmd[:, k + 1]
                    if k + 1 < self.control_horizon
                    else np.zeros(self.nu)
                )
            else:
                x_bar = target[:, k]
                u_bar = np.zeros(self.nu)

            A_k, B_k, C_k = self.compute_linear_model_matrices(x_bar, u_bar)
            self.A_params[k].value = A_k
            self.B_params[k].value = B_k
            self.C_params[k].value = C_k

        ## compute cost matrices
        for k in range(self.control_horizon):
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

            Q_k += np.eye(self.nx) * 1e-9
            self.Q_params[k].value = Q_k

        # Rotated terminal cost matrix
        theta_ref_f = target[3, -1]
        ct_f, st_f = np.cos(theta_ref_f), np.sin(theta_ref_f)
        R_f = np.array([[ct_f, -st_f], [st_f, ct_f]])
        Qf_pos_global = R_f @ np.array([[self.Qf[0, 0], 0], [0, self.Qf[1, 1]]]) @ R_f.T

        Qf_k = np.zeros((self.nx, self.nx))
        Qf_k[0:2, 0:2] = Qf_pos_global
        Qf_k[2, 2] = self.Qf[2, 2]
        Qf_k[3, 3] = self.Qf[3, 3]

        Qf_k += np.eye(self.nx) * 1e-9
        self.Qf_param.value = Qf_k
        self.prob.solve(solver=opt.OSQP, warm_start=True, verbose=False)

        # store for next loop (iMPC)
        self.prev_cmd = np.array(self.u.value)
        self.prev_trajectory = np.array(self.x.value)

        return np.array(self.x.value), np.array(self.u.value)
