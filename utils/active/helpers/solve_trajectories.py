import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from odefunc import odefunc
from mee2rv import mee2rv

def custom_rk45_batch(f, t_eval, y0_batch):
    N, D = y0_batch.shape
    T = len(t_eval)
    dt = t_eval[1] - t_eval[0]
    Y = np.zeros((N, T, D))
    Y[:, 0, :] = y0_batch

    for i in range(1, T):
        t = t_eval[i - 1]
        y = Y[:, i - 1, :]

        k1 = f(t, y)
        k2 = f(t + dt/4, y + dt/4 * k1)
        k3 = f(t + 3*dt/8, y + dt * (3/32 * k1 + 9/32 * k2))
        k4 = f(t + 12/13*dt, y + dt * (1932/2197 * k1 - 7200/2197 * k2 + 7296/2197 * k3))
        k5 = f(t + dt, y + dt * (439/216 * k1 - 8 * k2 + 3680/513 * k3 - 845/4104 * k4))
        k6 = f(t + dt/2, y + dt * (-8/27 * k1 + 2 * k2 - 3544/2565 * k3 + 1859/4104 * k4 - 11/40 * k5))

        Y[:, i, :] = y + dt * (16/135 * k1 + 6656/12825 * k3 + 28561/56430 * k4 - 9/50 * k5 + 2/55 * k6)

    return Y


def _solve_single_segment(bundle_idx, time_idx, time_steps, sigmas_combined, new_lam_bundles, m_b,
                          mu, F, c, m0, g0, Wm, Wc, substeps=5, evals_per_substep=10):
    t0 = time_steps[time_idx]
    t1 = time_steps[time_idx + 1]
    num_sigmas = sigmas_combined.shape[1]
    total_eval_points = substeps * evals_per_substep

    r_hist = np.zeros((num_sigmas, total_eval_points, 3))
    v_hist = np.zeros((num_sigmas, total_eval_points, 3))
    mass_hist = np.zeros((num_sigmas, total_eval_points))
    X_rows, y_rows = [], []

    lam0 = new_lam_bundles[time_idx, :, bundle_idx]
    lam1 = new_lam_bundles[time_idx + 1, :, bundle_idx]
    control_segments = np.linspace(0, 1, substeps)

    state = sigmas_combined[bundle_idx, :, :, time_idx]  # (num_sigmas, 7)

    for s_idx in range(substeps):
        tau0 = t0 + (t1 - t0) * (s_idx / substeps)
        tau1 = t0 + (t1 - t0) * ((s_idx + 1) / substeps)
        t_eval = np.linspace(tau0, tau1, evals_per_substep)

        lam = (1 - control_segments[s_idx]) * lam0 + control_segments[s_idx] * lam1

        def f(t, x_batch):
            return np.array([odefunc(t, x_i, mu, F, c, m0, g0, lam) for x_i in x_batch])

        result = custom_rk45_batch(f, t_eval, state)
        state = result[:, -1, :]

        p, f_, g, h, k, L = result[:, :, 0], result[:, :, 1], result[:, :, 2], result[:, :, 3], result[:, :, 4], result[:, :, 5]
        r_eval, v_eval = mee2rv(p.flatten(), f_.flatten(), g.flatten(), h.flatten(), k.flatten(), L.flatten(), mu)
        r_eval = r_eval.reshape(num_sigmas, evals_per_substep, 3)
        v_eval = v_eval.reshape(num_sigmas, evals_per_substep, 3)

        idx_start = s_idx * evals_per_substep
        idx_end = (s_idx + 1) * evals_per_substep

        r_hist[:, idx_start:idx_end] = r_eval
        v_hist[:, idx_start:idx_end] = v_eval
        mass_hist[:, idx_start:idx_end] = result[:, :, 6]

        for sigma_idx in range(num_sigmas):
            X_rows.append(np.hstack([tau0, state[sigma_idx, :7], bundle_idx, sigma_idx]))
            y_rows.append(lam)

    r_final = r_hist[:, -1, :]
    v_final = v_hist[:, -1, :]
    m_final = mass_hist[:, -1]
    means = np.sum(Wm[:, None] * np.hstack([r_final, v_final, m_final[:, None]]), axis=0)
    deviations = np.hstack([r_final, v_final, m_final[:, None]]) - means
    P = np.einsum("i,ij,ik->jk", Wc, deviations, deviations)

    trajs = np.hstack([r_hist, v_hist, mass_hist[:, :, None]])
    return np.array(X_rows), np.array(y_rows), P, means, trajs


def _solve_single_segment_wrapper(args):
    return _solve_single_segment(*args)


def solve_trajectories_with_covariance_parallel_with_progress(
    backTspan, time_steps, num_time_steps, num_bundles,
    sigmas_combined, new_lam_bundles, m_b, mu, F, c, m0, g0,
    Wm, Wc, num_workers=4
):
    X_list, y_list, P_list, means_list, traj_list = [], [], [], [], []

    for b_idx in tqdm(range(num_bundles), desc="Bundles", position=0):
        jobs = [
            (b_idx, t_idx, time_steps, sigmas_combined, new_lam_bundles, m_b,
             mu, F, c, m0, g0, Wm, Wc)
            for t_idx in range(num_time_steps - 1)
        ]

        with Pool(processes=num_workers) as pool:
            for result in tqdm(pool.imap_unordered(_solve_single_segment_wrapper, jobs),
                               total=len(jobs), desc=f"Bundle {b_idx:03d}", position=1, leave=False, mininterval=0.1):
                X_i, y_i, P_i, means_i, traj_i = result
                X_list.append(X_i)
                y_list.append(y_i)
                P_list.append(P_i)
                means_list.append(means_i)
                traj_list.append(traj_i)

    X = np.vstack(X_list)
    y = np.vstack(y_list)
    P_combined_history = list(P_list)
    means_history = list(means_list)
    trajectories = np.array(traj_list)

    return trajectories, P_combined_history, means_history, X, y
