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
L = 0.3
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
    control = np.zeros(2)

    while True:
        start_time = time.time()

        # 1. Safely grab the latest state from the simulation
        with shared.lock:
            if shared.goal_reached:
                break
            current_state = shared.state.copy()

        # 2. Check goal condition
        goal_dist = np.sqrt(
            (current_state[0] - path[0, -1]) ** 2
            + (current_state[1] - path[1, -1]) ** 2
        )
        if goal_dist < 0.2:
            with shared.lock:
                shared.goal_reached = True
            break

        # 3. Heavy MPC Math (Done OUTSIDE the lock to avoid blocking physics)
        target = get_ref_trajectory(current_state, path, TARGET_VEL, T, DT)

        # Add 1 time-step delay compensation
        ego_state = np.array([0.0, 0.0, current_state[2], 0.0])
        ego_state[0] += ego_state[2] * np.cos(ego_state[3]) * DT
        ego_state[1] += ego_state[2] * np.sin(ego_state[3]) * DT
        ego_state[2] += control[0] * DT
        ego_state[3] += control[0] * np.tan(control[1]) / L * DT

        x_mpc, u_mpc = mpc.step(ego_state, target, control, verbose=False)

        control[0] = u_mpc[0, 0]
        control[1] = u_mpc[1, 0]
        elapsed = time.time() - start_time

        new_ctrl = np.array(
            [
                control[1],  # steer (rad)
                current_state[2] + control[0] * DT,  # target speed (m/s)
            ]
        )

        x_mpc_world = ego_to_global(current_state, x_mpc) if x_mpc is not None else None

        # 4. Safely push results back to the shared object
        with shared.lock:
            shared.ctrl[:] = new_ctrl
            shared.mpc_accel = control[0]
            shared.mpc_steer = control[1]
            shared.mpc_elapsed = elapsed
            shared.x_mpc_world = x_mpc_world
            shared.x_hist.append(current_state[0])
            shared.y_hist.append(current_state[1])

        # Enforce ~5Hz loop
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


def _add_sphere(scn, radius, pos, rgba):
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0], dtype=np.float64),
        np.array(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def draw_path(scn, path):
    for i in range(path.shape[1] - 1):
        x1, y1 = path[0, i], path[1, i]
        x2, y2 = path[0, i + 1], path[1, i + 1]
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([0.008, length / 2, 0], dtype=np.float64),
            np.array([cx, cy, 0.03], dtype=np.float64),
            np.array([ux, uy, 0, -uy, ux, 0, 0, 0, 1], dtype=np.float64),
            np.array([0, 0.6, 1, 1], dtype=np.float32),
        )
        scn.ngeom += 1


def draw_trail(scn, x_hist, y_hist):
    step = max(1, len(x_hist) // 40)
    for i in range(0, len(x_hist), step):
        alpha = (i + 1) / len(x_hist) * 0.8
        _add_sphere(scn, 0.025, [x_hist[i], y_hist[i], 0.005], [1, 0, 0, alpha])


def draw_mpc_preview(scn, x_mpc_world):
    for i in range(x_mpc_world.shape[1]):
        _add_sphere(
            scn, 0.03, [x_mpc_world[0, i], x_mpc_world[1, i], 0.01], [0, 1, 0, 0.6]
        )


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

    mpc = MPC(
        VehicleModel(), T, DT, [20, 20, 10, 20], [30, 30, 30, 30], [10, 10], [10, 10]
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

        input("\033[92mPress Enter to continue...\033[0m")

        while viewer.is_running():
            step_start = time.time()

            # sync data with MPC thread
            with shared.lock:
                if shared.goal_reached:
                    break

                # Push physics state TO controller
                shared.state[:] = get_state(d, bid)

                # Pull controller inputs FROM controller
                d.ctrl[:] = shared.ctrl[:]

                # Extract copy of render vars to avoid holding lock during drawing
                mpc_elapsed = shared.mpc_elapsed
                mpc_accel = shared.mpc_accel
                mpc_steer = shared.mpc_steer
                x_mpc_world = shared.x_mpc_world
                local_x_hist = list(shared.x_hist)
                local_y_hist = list(shared.y_hist)
                current_speed = shared.state[2]

            # Step physics
            mujoco.mj_step(m, d)

            # Update viz
            viewer.user_scn.ngeom = 0
            draw_path(viewer.user_scn, path)
            draw_trail(viewer.user_scn, local_x_hist, local_y_hist)
            if x_mpc_world is not None:
                draw_mpc_preview(viewer.user_scn, x_mpc_world)

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

            # Pace to real-time
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        # Show end state
        if shared.goal_reached:
            viewer.set_texts([(None, None, "GOAL REACHED", "")])
            viewer.sync()
            time.sleep(1.5)

        viewer.clear_texts()

    # Final plot (grab the history one last time)
    with shared.lock:
        final_x_hist = list(shared.x_hist)
        final_y_hist = list(shared.y_hist)

    plot_results(path, final_x_hist, final_y_hist)


if __name__ == "__main__":
    main()
