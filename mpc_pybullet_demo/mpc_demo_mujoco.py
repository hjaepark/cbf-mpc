import pathlib
import threading
import time

import matplotlib.pyplot as plt
import numpy as np

import mujoco
import mujoco.viewer
from cvxpy_mpc import MPC, VehicleModel
from cvxpy_mpc.utils import compute_path_from_wp, get_ref_trajectory

TARGET_VEL = 1.0
T = 5.0
DT = 0.2  # controller time step


# MPC and sim are on 2 threads
# avoids messy global vars
class SharedData:
    """Encapsulates all data shared between the Physics thread and MPC thread."""

    def __init__(self):
        self.lock = threading.Lock()

        # Core control & state
        self.ctrl = np.zeros(2)  # [steering_rad, target_speed_mps]
        self.state = np.zeros(4)  # [x, y, speed, yaw]
        self.goal_reached = False
        self.is_active = True

        # Telemetry & visualization
        self.x_hist = []
        self.y_hist = []
        self.x_mpc_world = None

        # HUD stats
        self.mpc_elapsed = 0.0
        self.mpc_accel = 0.0
        self.mpc_steer = 0.0


def controller_loop(mpc, path, shared):
    """Runs continuously in the background at ~5Hz"""
    # Assuming L is the vehicle wheelbase, grab it from the model object
    L = mpc.vehicle.wheelbase

    while True:
        start_time = time.time()

        # (safely) grab the latest state from the simulation
        with shared.lock:
            if not shared.is_active or shared.goal_reached:
                break
            current_state = shared.state.copy()  # Global [X, Y, V, Theta]
            last_control = (shared.mpc_accel, shared.mpc_steer)  # [Accel, Steer]
            elapsed = shared.mpc_elapsed

        # Check goal using absolute global coordinates
        goal_dist = np.sqrt(
            (current_state[0] - path[0, -1]) ** 2
            + (current_state[1] - path[1, -1]) ** 2
        )
        if goal_dist < 0.2:
            with shared.lock:
                shared.goal_reached = True
            break

        # Add delay compensation
        # ok why we need this in practice? the optimiser takes some time
        # to compute the next command. The mpc should compute the command for t+delay because that is
        # when it will be applied. the actual delay is the expected computation time (assumed from last one)
        pred_state = current_state.copy()
        v = pred_state[2]
        theta = pred_state[3]
        a = last_control[0]
        delta = last_control[1]

        # Integrate physics forward in global space
        pred_state[0] += v * np.cos(theta) * elapsed
        pred_state[1] += v * np.sin(theta) * elapsed
        pred_state[2] += a * elapsed
        pred_state[3] += (v * np.tan(delta) / L) * elapsed

        # Get reference trajectory (already matches global coordinates)
        target = get_ref_trajectory(pred_state, path, TARGET_VEL, T, DT)

        # Integrate physics forward in ego space
        pred_ego_state = [0.0, 0.0, v, 0.0]
        pred_ego_state[0] += v * np.cos(theta) * elapsed
        pred_ego_state[1] += v * np.sin(theta) * elapsed
        pred_ego_state[2] += a * elapsed
        pred_ego_state[3] += (v * np.tan(delta) / L) * elapsed
        x_mpc, u_mpc = mpc.step(pred_ego_state, target, verbose=False)

        # Extract the immediate next optimal control actions
        control = (u_mpc[0, 0], u_mpc[1, 0])
        elapsed = time.time() - start_time

        # Format control values for MuJoCo actuator expectations
        new_ctrl = np.array(
            [
                control[1],  # Target steer angle (rad)
                current_state[2] + control[0] * DT,  # Target velocity (m/s)
            ]
        )

        # Safely push results back to the shared object
        with shared.lock:
            shared.ctrl[:] = new_ctrl
            shared.mpc_accel = control[0]
            shared.mpc_steer = control[1]
            shared.mpc_elapsed = elapsed
            shared.x_mpc_world = ego_to_global(pred_state, x_mpc)
            shared.x_hist.append(current_state[0])
            shared.y_hist.append(current_state[1])

        # Enforce loop frequency
        elapsed_total = time.time() - start_time
        sleep_time = max(0.0, DT - elapsed_total)
        time.sleep(sleep_time)


def body_id(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if i == -1:
        raise ValueError(f"Body '{name}' not found")
    return i


def get_state(data, bid):
    rot = data.xmat[bid].reshape(3, 3)
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    speed = np.linalg.norm(data.qvel[0:2])
    return np.array([data.xpos[bid][0], data.xpos[bid][1], speed, yaw])


def ego_to_global(state, x_mpc):
    traj = x_mpc[:2, :].copy()
    ct, st = np.cos(state[3]), np.sin(state[3])
    R = np.array([[ct, -st], [st, ct]])
    traj = R @ traj
    traj[0, :] += state[0]
    traj[1, :] += state[1]
    return traj


def draw_path(viewer, path):
    """Draws the reference path"""
    for i in range(path.shape[1] - 1):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break

        # next geometry slot
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]

        p1 = np.array([path[0, i], path[1, i], 0.03], dtype=np.float64)
        p2 = np.array([path[0, i + 1], path[1, i + 1], 0.03], dtype=np.float64)

        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array(
                [0.008, 0.0, 0.0], dtype=np.float64
            ),  # [radius, unused, unused]
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([0, 0.6, 1, 1], dtype=np.float32),
        )

        # mujoco handles the vector math to stretch it between p1 and p2
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.008, p1, p2)

        viewer.user_scn.ngeom += 1


def draw_trail(viewer, x_hist, y_hist):
    """Draws breadcrumbs using strict arrays."""
    step = max(1, len(x_hist) // 40)
    for i in range(0, len(x_hist), step):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break

        # next geometry slot
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        alpha = (i + 1) / len(x_hist) * 0.8

        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.025, 0.0, 0.0], dtype=np.float64),
            pos=np.array([x_hist[i], y_hist[i], 0.005], dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=np.array([1, 0, 0, alpha], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1


def draw_mpc_preview(viewer, x_mpc_world):
    """Draws predicted horizon points using strict arrays."""
    for i in range(x_mpc_world.shape[1]):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            break

        # next geometry slot
        g = viewer.user_scn.geoms[viewer.user_scn.ngeom]

        mujoco.mjv_initGeom(
            g,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.03, 0.0, 0.0], dtype=np.float64),
            pos=np.array(
                [x_mpc_world[0, i], x_mpc_world[1, i], 0.01], dtype=np.float64
            ),
            mat=np.eye(3).ravel(),
            rgba=np.array([0, 1, 0, 0.6], dtype=np.float32),
        )
        viewer.user_scn.ngeom += 1


def plot_results(path, x_hist, y_hist):
    plt.style.use("ggplot")
    plt.figure()
    plt.title("MPC Tracking Results")
    plt.plot(
        path[0, :], path[1, :], c="tab:orange", marker=".", label="reference track"
    )
    plt.plot(
        x_hist, y_hist, c="tab:blue", marker=".", alpha=0.5, label="vehicle trajectory"
    )
    plt.axis("equal")
    plt.legend()
    plt.show()


# here we run the sim loop
def main():
    model_path = pathlib.Path(__file__).parent / "models" / "mushr" / "mush_nano.xml"
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    bid = body_id(m, "buddy")

    d.qpos[0] = 0.0
    d.qpos[1] = 0.3
    d.qpos[2] = 0.1
    d.qpos[3] = 1.0
    mujoco.mj_forward(m, d)

    path = compute_path_from_wp(
        [0, 3, 4, 6, 10, 11, 12, 6, 1, 0],
        [0, 0, 2, 4, 3, 3, -1, -6, -2, -2],
        0.05,
    )

    state_cost = [
        1.0,
        80.0,
        10.0,
        20.0,
    ]  # [Along-track, Cross-track, Velocity, Heading]
    actuation_cost = [10.0, 10.0]
    mpc = MPC(
        VehicleModel(),
        T,
        DT,
        state_cost,
        state_cost,
        actuation_cost,
        actuation_cost,
    )

    steer_jnt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "buddy_steering_wheel")
    steer_qaddr = m.jnt_qposadr[steer_jnt]

    # Instantiate the shared data object
    shared = SharedData()

    # Pass the shared object to the background controller thread
    t = threading.Thread(target=controller_loop, args=(mpc, path, shared), daemon=True)
    t.start()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -45

        fps = 60.0
        render_dt = 1.0 / fps
        input("\033[92mPress Enter to continue...\033[0m")

        sim_start_time = time.perf_counter()
        try:
            while viewer.is_running() and not shared.goal_reached:

                # step the physics of mujoco
                elapsed_real_time = time.perf_counter() - sim_start_time
                while d.time < elapsed_real_time:

                    # Apply controls right before the step
                    with shared.lock:
                        # this value is held constant between MPC updates. (Zero order hold)
                        d.ctrl[:] = shared.ctrl[:]

                    mujoco.mj_step(m, d)

                # sync data with MPC thread
                with shared.lock:
                    # TODO: this would come from a proper state estimator
                    shared.state[:] = get_state(d, bid)

                    mpc_elapsed = shared.mpc_elapsed
                    mpc_accel = shared.mpc_accel
                    mpc_steer = shared.mpc_steer
                    x_mpc_world = shared.x_mpc_world
                    local_x_hist = list(shared.x_hist)
                    local_y_hist = list(shared.y_hist)
                    current_speed = shared.state[2]

                # Update viz
                viewer.user_scn.ngeom = 0
                draw_path(viewer, path)
                draw_trail(viewer, local_x_hist, local_y_hist)
                if x_mpc_world is not None:
                    draw_mpc_preview(viewer, x_mpc_world)

                actual_steer = np.degrees(d.qpos[steer_qaddr])
                goal_dist = np.sqrt(
                    (d.xpos[bid][0] - path[0, -1]) ** 2
                    + (d.xpos[bid][1] - path[1, -1]) ** 2
                )

                viewer.set_texts(
                    [
                        (
                            None,
                            None,
                            f"MPC Demo\n"
                            f"state:  v {current_speed:.2f} m/s  |  steer {actual_steer:.1f} deg\n"
                            f"MPC:    accel {mpc_accel:.2f} m/s^2  |  steer {np.degrees(mpc_steer):.1f} deg  |  {mpc_elapsed*1000:.0f} ms\n"
                            f"goal:   {goal_dist:.2f} m",
                            "",
                        )
                    ]
                )

                viewer.sync()

                # Sleep just enough to hit 60 FPS
                time_until_next_frame = render_dt - (
                    time.perf_counter() - elapsed_real_time - sim_start_time
                )
                if time_until_next_frame > 0:
                    time.sleep(time_until_next_frame)

            # Show end state
            if shared.goal_reached:
                viewer.set_texts([(None, None, "GOAL REACHED", "")])
                viewer.sync()
                time.sleep(1.5)
        finally:
            # handles user closing the GUI window
            with shared.lock:
                shared.is_active = False

        viewer.clear_texts()

    # Final plot (grab the history one last time)
    with shared.lock:
        final_x_hist = list(shared.x_hist)
        final_y_hist = list(shared.y_hist)

    plot_results(path, final_x_hist, final_y_hist)


if __name__ == "__main__":
    main()
