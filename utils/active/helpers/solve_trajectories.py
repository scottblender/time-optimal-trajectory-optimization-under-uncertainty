import numpy as np
import scipy.integrate
import traceback
from multiprocessing import Pool, Manager, Process
from tqdm import tqdm

from rv2mee import rv2mee
from mee2rv import mee2rv
from odefunc import odefunc


def sample_within_bounds(mean, cov, max_tries=100):
    for _ in range(max_tries):
        sample = np.random.multivariate_normal(mean, cov)
        z_score = np.abs(sample - mean) / np.sqrt(np.diag(cov))
        if np.all(z_score <= 3):
            return sample
    return mean


def listener(q, total):
    with tqdm(total=total, desc="Total integration progress") as pbar:
        completed = 0
        while completed < total:
            msg = q.get()
            if isinstance(msg, int) and msg > 0:
                pbar.update(msg)
                completed += msg
            elif msg == -1:
                completed += 1
                pbar.write("[WARN] Subprocess failed")


def _solve_single_bundle(args):
    (bundle_index, backTspan, time_steps, num_time_steps, sigmas_combined,
     new_lam_bundles, mass_bundles, mu, F, c, m0, g0, Wm, Wc, progress_queue) = args

    try:
        forwardTspan = backTspan[::-1]
        time = [forwardTspan[time_steps[i]] for i in range(num_time_steps)]
        bundle_trajectories = []
        bundle_P_combined_history = []
        bundle_means_history = []

        num_sigma = sigmas_combined.shape[1]
        substeps = 10
        evals_per_substep = 20
        steps_per_segment = substeps * evals_per_substep
        num_segments = num_time_steps - 1
        total_rows = num_sigma * steps_per_segment * num_segments

        X_rows = np.zeros((total_rows, 17))  # [time, MEE(6), mass, diag cov(7), bundle, sigma]
        y_rows = np.zeros((total_rows, 7))   # control vector
        row_idx = 0

        for j in range(num_segments):
            tstart, tend = time[j], time[j + 1]
            sigma_combined = sigmas_combined[bundle_index, :, :, j]
            idx_start = time_steps[j]
            new_lam = new_lam_bundles[idx_start, :, bundle_index]
            P_lam = np.eye(7) * 0.001
            sub_times = np.linspace(tstart, tend, substeps + 1)

            full_state_sigma_points = []
            cartesian_sigma_points = []
            sigma_point_trajectories = []

            for sigma_idx in range(num_sigma):
                r0, v0, mass = sigma_combined[sigma_idx, :3], sigma_combined[sigma_idx, 3:6], sigma_combined[sigma_idx, 6]
                initial_state = rv2mee(np.array([r0]), np.array([v0]), mu).flatten()
                S = np.append(np.append(initial_state, mass), new_lam)
                full_states = []
                time_values = []
                prev_lam_mean = new_lam.copy()

                for k in range(substeps):
                    tsub_start, tsub_end = sub_times[k], sub_times[k + 1]
                    lam = new_lam if k == 0 else sample_within_bounds(prev_lam_mean, P_lam)
                    prev_lam_mean = lam.copy()
                    S[-7:] = lam

                    func = lambda t, x: odefunc(t, x, mu, F, c, m0, g0)
                    t_eval = np.linspace(tsub_start, tsub_end, evals_per_substep)
                    sol = scipy.integrate.solve_ivp(func, [tsub_start, tsub_end], S, method='RK45',
                                                    t_eval=t_eval, rtol=1e-6, atol=1e-8)

                    if not sol.success:
                        raise RuntimeError("Integration failed.")

                    full_states.append(sol.y.T)
                    time_values.append(sol.t)
                    S = sol.y[:, -1]

                    if progress_queue is not None:
                        progress_queue.put(len(sol.t))  # track actual rows added to X

                full_states = np.vstack(full_states)
                time_values = np.hstack(time_values)
                full_state_sigma_points.append(full_states[:, :7])

                r_eval, v_eval = mee2rv(full_states[:, 0], full_states[:, 1], full_states[:, 2],
                                        full_states[:, 3], full_states[:, 4], full_states[:, 5], mu)
                cartesian = np.hstack((r_eval, v_eval, full_states[:, 6:7]))
                cartesian_sigma_points.append(cartesian)
                sigma_point_trajectories.append(cartesian)

                y_rows[row_idx:row_idx + full_states.shape[0], :] = full_states[:, -7:]
                for step in range(full_states.shape[0]):
                    X_rows[row_idx + step, :] = np.hstack([
                        time_values[step],
                        full_states[step, :7],
                        np.zeros(7),  # placeholder for diag cov
                        bundle_index,
                        sigma_idx
                    ])
                row_idx += full_states.shape[0]

            full_state_sigma_points = np.array(full_state_sigma_points)
            cartesian_sigma_points = np.array(cartesian_sigma_points)

            mean_state = np.sum(Wm[:, None, None] * full_state_sigma_points, axis=0)
            deviations = full_state_sigma_points - mean_state[None, :, :]
            P_combined = np.einsum("i,ijk,ijl->jkl", Wc, deviations, deviations)
            P_combined_diag = np.array([np.diag(np.diag(P)) for P in P_combined])

            mean_cartesian = np.sum(Wm[:, None, None] * cartesian_sigma_points, axis=0)
            deviations_cartesian = cartesian_sigma_points - mean_cartesian[None, :, :]
            P_combined_cartesian = np.einsum("i,ijk,ijl->jkl", Wc, deviations_cartesian, deviations_cartesian)
            P_combined_cartesian_diag = np.array([np.diag(np.diag(P)) for P in P_combined_cartesian])

            for t in range(len(P_combined_diag)):
                cov_diag = np.diag(P_combined_diag[t])
                for sigma_idx in range(num_sigma):
                    index = (t * num_sigma) + sigma_idx + (j * num_sigma * steps_per_segment)
                    X_rows[index, 8:15] = cov_diag

            bundle_trajectories.append(sigma_point_trajectories)
            bundle_P_combined_history.append(P_combined_cartesian_diag)
            bundle_means_history.append(mean_cartesian)

        return bundle_trajectories, bundle_P_combined_history, bundle_means_history, X_rows[:row_idx], y_rows[:row_idx]

    except Exception:
        if progress_queue:
            progress_queue.put(-1)
        print(f"[ERROR] Bundle {bundle_index} failed:")
        traceback.print_exc()
        return None


def solve_trajectories_with_covariance_parallel_with_progress(
    backTspan, time_steps, num_time_steps, num_bundles,
    sigmas_combined, new_lam_bundles, mass_bundles, mu, F, c, m0, g0, Wm, Wc,
    num_workers=4
):
    num_sigma = sigmas_combined.shape[1]
    substeps = 10
    evals_per_substep = 20
    total_steps = num_bundles * (num_time_steps - 1) * num_sigma * substeps * evals_per_substep

    manager = Manager()
    progress_queue = manager.Queue()
    listener_process = Process(target=listener, args=(progress_queue, total_steps))
    listener_process.start()

    job_args = [
        (i, backTspan, time_steps, num_time_steps, sigmas_combined, new_lam_bundles, mass_bundles,
         mu, F, c, m0, g0, Wm, Wc, progress_queue)
        for i in range(num_bundles)
    ]

    with Pool(processes=num_workers) as pool:
        results = pool.map(_solve_single_bundle, job_args)

    listener_process.join()
    results = [r for r in results if r is not None]
    if not results:
        raise RuntimeError("No bundle results returned successfully.")

    trajectories, P_combined_history, means_history, X_list, y_list = zip(*results)
    X = np.vstack(X_list)
    y = np.vstack(y_list)

    sort_indices = np.lexsort((X[:, 0], X[:, -1], X[:, -2]))  # sort by time, sigma, bundle
    X_sorted = X[sort_indices]
    y_sorted = y[sort_indices]

    return np.array(trajectories), np.array(P_combined_history), np.array(means_history), X_sorted, y_sorted
