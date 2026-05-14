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
DT = 0.2
RENDER_HZ = 30.0


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


def set_ctrl(data, current_speed, acceleration, steering):
    target_speed = current_speed + acceleration * DT
    data.ctrl[0] = steering
    data.ctrl[1] = target_speed


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
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0, 0], dtype=np.float64),
        np.array(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).ravel(),
        np.array(rgba, dtype=np.float32),
    )
    scn.ngeom += 1


def draw_path(scn, path):
    step = max(1, path.shape[1] // 80)
    for i in range(0, path.shape[1], step):
        _add_sphere(scn, 0.04, [path[0, i], path[1, i], 0.005], [0, 0, 1, 0.5])


def draw_trail(scn, x_hist, y_hist):
    step = max(1, len(x_hist) // 40)
    for i in range(0, len(x_hist), step):
        alpha = (i + 1) / len(x_hist) * 0.8
        _add_sphere(scn, 0.025, [x_hist[i], y_hist[i], 0.005], [1, 0, 0, alpha])


def draw_mpc_preview(scn, x_mpc_world):
    for i in range(x_mpc_world.shape[1]):
        _add_sphere(scn, 0.03, [x_mpc_world[0, i], x_mpc_world[1, i], 0.01], [0, 1, 0, 0.6])


def plot_results(path, x_hist, y_hist):
    plt.style.use("ggplot")
    plt.figure()
    plt.title("MPC Tracking Results")
    plt.plot(path[0, :], path[1, :], c="tab:orange", marker=".", label="reference track")
    plt.plot(x_hist, y_hist, c="tab:blue", marker=".", alpha=0.5, label="vehicle trajectory")
    plt.axis("equal")
    plt.legend()
    plt.show()


def main():
    model_path = pathlib.Path(__file__).parent / "models" / "mushr" / "mush_nano.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    bid = body_id(model, "buddy")

    data.qpos[0] = 0.0
    data.qpos[1] = 0.3
    data.qpos[2] = 0.1
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)

    path = compute_path_from_wp(
        [0, 3, 4, 6, 10, 11, 12, 6, 1, 0],
        [0, 0, 2, 4, 3, 3, -1, -6, -2, -2],
        0.05,
    )

    mpc = MPC(
        VehicleModel(), T, DT, [20, 20, 10, 20], [30, 30, 30, 30], [10, 10], [10, 10]
    )

    control = np.zeros(2)
    x_hist, y_hist = [], []
    x_mpc_world = None
    mpc_elapsed = 0.0
    mpc_rtf = 0.0
    goal_reached = False
    lock = threading.Lock()

    def physics_loop(viewer):
        nonlocal control, x_hist, y_hist, x_mpc_world, mpc_elapsed, mpc_rtf, goal_reached
        n_substeps = int(DT / model.opt.timestep)

        while viewer.is_running() and not goal_reached:
            with viewer.lock():
                state = get_state(data, bid)

            goal_dist = np.sqrt(
                (state[0] - path[0, -1]) ** 2 + (state[1] - path[1, -1]) ** 2
            )
            if goal_dist < 0.2:
                goal_reached = True
                break

            target = get_ref_trajectory(state, path, TARGET_VEL, T, DT)
            ego_state = np.array([0.0, 0.0, state[2], 0.0])
            ego_state[0] += ego_state[2] * np.cos(ego_state[3]) * DT
            ego_state[1] += ego_state[2] * np.sin(ego_state[3]) * DT
            ego_state[2] += control[0] * DT
            ego_state[3] += control[0] * np.tan(control[1]) / L * DT

            start = time.time()
            x_mpc, u_mpc = mpc.step(ego_state, target, control, verbose=False)
            control[0] = u_mpc[0, 0]
            control[1] = u_mpc[1, 0]
            elapsed = time.time() - start
            rtf = DT / elapsed if elapsed > 0 else 0

            with viewer.lock():
                set_ctrl(data, state[2], control[0], control[1])
                for _ in range(n_substeps):
                    mujoco.mj_step(model, data)
                x_hist.append(state[0])
                y_hist.append(state[1])

            x_mpc_world = ego_to_global(state, x_mpc) if x_mpc is not None else None

            with lock:
                mpc_elapsed = elapsed
                mpc_rtf = rtf

            if DT - elapsed > 0:
                time.sleep(DT - elapsed)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = -90
        viewer.cam.elevation = -45

        input("\033[92mPress Enter to continue...\033[0m")

        thread = threading.Thread(target=physics_loop, args=(viewer,), daemon=True)
        thread.start()

        while viewer.is_running() and not goal_reached:
            state = get_state(data, bid)

            goal_dist = np.sqrt(
                (state[0] - path[0, -1]) ** 2 + (state[1] - path[1, -1]) ** 2
            )

            with lock:
                elapsed = mpc_elapsed
                rtf = mpc_rtf

            viewer.user_scn.ngeom = 0
            draw_path(viewer.user_scn, path)
            draw_trail(viewer.user_scn, x_hist, y_hist)
            if x_mpc_world is not None:
                draw_mpc_preview(viewer.user_scn, x_mpc_world)
            viewer.set_texts([
                (None, None,
                 f"MPC Demo\n"
                 f"v: {state[2]:.2f} m/s  |  steer: {np.degrees(control[1]):.1f} deg\n"
                 f"mpc: {elapsed*1000:.0f} ms  |  RTF: {rtf:.1f}x  |  goal: {goal_dist:.2f} m",
                 ""),
            ])

            viewer.sync()
            time.sleep(1.0 / RENDER_HZ)

        viewer.set_texts([
            (None, None, "GOAL REACHED", ""),
        ])
        viewer.sync()
        viewer.clear_texts()

    plot_results(path, x_hist, y_hist)


if __name__ == "__main__":
    main()
