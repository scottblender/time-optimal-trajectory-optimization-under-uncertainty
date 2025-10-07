import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from multiprocessing import Pool, Manager, Process
import traceback

from rv2mee import rv2mee
from mee2rv import mee2rv
from odefunc import odefunc

def sample_within_bounds(mean, cov, max_tries=100):
    for _ in range(max_tries):
        sample = np.random.multivariate_normal(mean, cov)
        z = np.abs(sample - mean) / np.sqrt(np.diag(cov))
        if np.all(z <= 3):
            return sample
    return mean

def listener(q, total):
    with tqdm(total=total, desc="Monte Carlo progress") as pbar:
        completed = 0
        while completed < total:
            msg = q.get()
            if isinstance(msg, int) and msg > 0:
                pbar.update(msg)
                completed += msg
            elif msg == -1:
                completed += 1
                pbar.write("[WARN] Subprocess failed")

def _solve_mc_single_bundle(args):
    (bundle_idx_global, time_steps, num_time_steps, sigmas_combined, new_lam_bundles,
     mu, F, c, m0, g0, num_samples, P_init, P_control, backTspan, queue, bundle_idx_local) = args

    try:
        forwardTspan = backTspan[::-1]
        time = forwardTspan[time_steps]
        substeps = 10
        evals_per_substep = 20
        steps_per_segment = substeps * evals_per_substep
        num_segments = num_time_steps - 1
        total_steps = num_samples * steps_per_segment * num_segments

        X_rows = np.zeros((total_steps, 7))
        y_rows = np.zeros((total_steps, 7))
        cov_rows = np.zeros((total_steps, 7))
        t_rows = np.zeros((total_steps,))
        b_rows = np.full((total_steps,), bundle_idx_global)
        s_rows = np.zeros((total_steps,))

        P_combined_history = []
        means_history = []
        trajectories = []

        row_idx = 0

        for j in range(num_segments):
            tstart, tend = time[j], time[j + 1]
            sigma_state = sigmas_combined[bundle_idx_local, 0, :, j]
            lam_nominal = new_lam_bundles[time_steps[j], :, bundle_idx_local]
            sub_times = np.linspace(tstart, tend, substeps + 1)
            cartesian_samples = []

            for sample_idx in range(num_samples):
                sample_state = sigma_state if sample_idx == 0 else sample_within_bounds(sigma_state, P_init)
                r0, v0, mass = sample_state[:3], sample_state[3:6], sample_state[6]
                mee = rv2mee(np.array([r0]), np.array([v0]), mu).flatten()
                S = np.concatenate([mee, [mass], lam_nominal])

                full_states = []
                time_vals = []

                prev_lam_mean = lam_nominal
                for k in range(substeps):
                    if k == 0:
                        lam = prev_lam_mean
                    else:
                        lam = sample_within_bounds(prev_lam_mean, P_control)
                    prev_lam_mean = lam.copy()
                    S[-7:] = lam
                    t0, t1 = sub_times[k], sub_times[k + 1]
                    t_eval = np.linspace(t0, t1, evals_per_substep)
                    func = lambda t, x: odefunc(t, x, mu, F, c, m0, g0)
                    sol = solve_ivp(func, [t0, t1], S, t_eval=t_eval, rtol=1e-6, atol=1e-8)

                    if sol.success:
                        full_states.append(sol.y.T)
                        time_vals.append(sol.t)
                        S = sol.y[:, -1]
                        if queue is not None:
                            queue.put(len(sol.t))

                if not full_states:
                    continue

                full_states = np.vstack(full_states)
                t_eval = np.hstack(time_vals)
                mee_state = full_states[:, :7]
                ctrl_state = full_states[:, -7:]

                r_eval, v_eval = mee2rv(mee_state[:, 0], mee_state[:, 1], mee_state[:, 2],
                                        mee_state[:, 3], mee_state[:, 4], mee_state[:, 5], mu)
                cartesian = np.hstack((r_eval, v_eval, mee_state[:, 6:7]))

                cartesian_samples.append(cartesian)
                trajectories.append(cartesian)

                N = len(t_eval)
                X_rows[row_idx:row_idx+N] = mee_state
                y_rows[row_idx:row_idx+N] = ctrl_state
                t_rows[row_idx:row_idx+N] = t_eval
                s_rows[row_idx:row_idx+N] = sample_idx
                row_idx += N

            cartesian_samples = np.array(cartesian_samples)
            mean_cartesian = np.mean(cartesian_samples, axis=0)
            deviations = cartesian_samples - mean_cartesian[np.newaxis, :, :]
            P_combined = np.einsum("ijk,ijl->jkl", deviations, deviations) / num_samples
            P_diag = np.array([np.diag(np.diag(P)) for P in P_combined])

            for t in range(P_diag.shape[0]):
                if row_idx - P_diag.shape[0] + t < total_steps:
                    cov_rows[row_idx - P_diag.shape[0] + t] = np.diag(P_diag[t])

            P_combined_history.append(P_diag)
            means_history.append(mean_cartesian)

        return (
            np.array(trajectories).reshape(num_segments, num_samples, -1, 7),
            P_combined_history, means_history,
            X_rows[:row_idx], y_rows[:row_idx],
            cov_rows[:row_idx], t_rows[:row_idx],
            b_rows[:row_idx], s_rows[:row_idx]
        )

    except Exception:
        if queue is not None:
            queue.put(-1)
        traceback.print_exc()
        return None

def generate_monte_carlo_trajectories_parallel(
    backTspan, time_steps, num_time_steps,
    sigmas_combined, new_lam_bundles, mu, F, c, m0, g0,
    global_bundle_indices, num_samples=1000, num_workers=4
):
    DU_km = 696340.0  # Sun radius in km
    g0_s = 9.81/1000
    TU = np.sqrt(DU_km / g0_s)
    VU_kms = DU_km / TU # Convert m/s to km/s using g0

    # Set desired physical covariances (e.g., 0.1 km², 1e-4 (km/s)², 1 kg²)
    P_pos_km2 = np.eye(3) * 0.01        # km²
    P_vel_kms2 = np.eye(3) * 1e-10      # (km/s)²
    P_mass_kg2 = np.array([[1e-3]])     # kg²

    # Convert to non-dimensional units for input
    P_pos = P_pos_km2 / (DU_km**2)
    P_vel = P_vel_kms2 / (VU_kms**2)
    P_mass = P_mass_kg2 / (4000**2)
    P_init = np.block([
        [P_pos, np.zeros((3, 3)), np.zeros((3, 1))],
        [np.zeros((3, 3)), P_vel, np.zeros((3, 1))],
        [np.zeros((1, 3)), np.zeros((1, 3)), P_mass]
    ])
    P_control = np.eye(7) * 1e-16

    substeps = 10
    evals_per_substep = 20
    total = len(global_bundle_indices) * (num_time_steps - 1) * num_samples * substeps * evals_per_substep

    manager = Manager()
    q = manager.Queue()
    listener_process = Process(target=listener, args=(q, total))
    listener_process.start()

    args_list = [
        (global_idx, time_steps, num_time_steps, sigmas_combined, new_lam_bundles,
         mu, F, c, m0, g0, num_samples, P_init, P_control, backTspan, q, local_idx)
        for local_idx, global_idx in enumerate(global_bundle_indices)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(_solve_mc_single_bundle, args_list)

    listener_process.join()

    trajectories, P_hist, means_hist = [], [], []
    X_rows, y_rows, cov_rows, time_rows, b_rows, s_rows = [], [], [], [], [], []

    for res in results:
        if res is None:
            continue
        traj, P, M, X, y, cov, t, b, s = res
        trajectories.append(traj)
        P_hist.append(P)
        means_hist.append(M)
        X_rows.append(X)
        y_rows.append(y)
        cov_rows.append(cov)
        time_rows.append(t)
        b_rows.append(b)
        s_rows.append(s)

    X = np.hstack((
        np.concatenate(time_rows).reshape(-1, 1),
        np.concatenate(X_rows),
        np.concatenate(cov_rows),
        np.concatenate(b_rows).reshape(-1, 1),
        np.concatenate(s_rows).reshape(-1, 1)
    ))
    y = np.concatenate(y_rows)

    sort_indices = np.lexsort((X[:, 0], X[:, -1], X[:, -2]))
    X_sorted = X[sort_indices]
    y_sorted = y[sort_indices]

    return (
        np.array(trajectories),
        np.array(P_hist),
        np.array(means_hist),
        X_sorted,
        y_sorted
    )