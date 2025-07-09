import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm
from multiprocessing import Pool, Manager, Process
from numpy.linalg import inv
import traceback
import rv2mee
import mee2rv
import odefunc


def _listener(q, total):
    with tqdm(total=total, desc="Monte Carlo progress") as pbar:
        completed = 0
        while completed < total:
            msg = q.get()
            if msg == 1:
                pbar.update(1)
                completed += 1
            elif msg == -1:
                completed += 1
                pbar.write("[WARN] Subprocess failed")


def sample_within_bounds(mean, cov, max_tries=100):
    for _ in range(max_tries):
        sample = np.random.multivariate_normal(mean, cov)
        z_score = np.abs(sample - mean) / np.sqrt(np.diag(cov))
        if np.all(z_score <= 3):
            return sample
    return mean


def _solve_mc_single_bundle(args):
    (bundle_idx, time_steps, num_time_steps, sigmas_combined, new_lam_bundles,
     mu, F, c, m0, g0, num_samples, P_init, P_control, backTspan, queue) = args

    try:
        forwardTspan = backTspan[::-1]
        time = forwardTspan[time_steps]
        print(time)
        num_updates = 10

        bundle_trajectories = []
        state_history = []
        control_state_history = []
        covariance_history = []
        time_history = []
        bundle_index_history = []
        sample_index_history = []

        P_combined_history = []
        means_history = []

        for j in range(num_time_steps - 1):
            tstart, tend = time[j], time[j + 1]
            sigma_state = sigmas_combined[bundle_idx, 0, :, j]
            new_lam_nominal = new_lam_bundles[j, :, bundle_idx]
            sub_times = np.linspace(tstart, tend, num_updates + 1)

            full_state_samples = []
            cartesian_samples = []

            for sample_idx in range(num_samples):
                sample_state = sigma_state if sample_idx == 0 else sample_within_bounds(sigma_state, P_init)
                r0 = sample_state[:3]
                v0 = sample_state[3:6]
                mass = sample_state[6]

                mee_state = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu).flatten()
                S = np.concatenate([mee_state, [mass], new_lam_nominal])

                full_states = []
                time_values = []

                for k in range(num_updates):
                    tsub_start, tsub_end = sub_times[k], sub_times[k + 1]
                    lam_k = new_lam_nominal if k == 0 else sample_within_bounds(new_lam_nominal, P_control)
                    S[-7:] = lam_k

                    func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)
                    tspan = np.linspace(tsub_start, tsub_end, 20)

                    Sf = solve_ivp(func, [tsub_start, tsub_end], S, t_eval=tspan,
                                   rtol=1e-3, atol=1e-6)
                    if Sf.success:
                        full_states.append(Sf.y.T)
                        time_values.append(Sf.t)
                        S = Sf.y[:, -1]

                full_states = np.vstack(full_states)
                time_values = np.hstack(time_values)
                full_state_samples.append(full_states[:, :7])

                r_new, v_new = mee2rv.mee2rv(full_states[:, 0], full_states[:, 1], full_states[:, 2],
                                             full_states[:, 3], full_states[:, 4], full_states[:, 5], mu)
                cartesian_state = np.hstack((r_new, v_new, full_states[:, 6].reshape(-1, 1)))
                cartesian_samples.append(cartesian_state)

                bundle_trajectories.append(cartesian_state)
                control_state_history.append(full_states[:, -7:])

                # ✅ Update progress per sample
                if queue is not None:
                    queue.put(1)

            full_state_samples = np.array(full_state_samples)
            cartesian_samples = np.array(cartesian_samples)

            mean_cartesian = np.mean(cartesian_samples, axis=0)
            deviations = cartesian_samples - mean_cartesian[np.newaxis, :, :]
            P_combined_sample = np.einsum('ijk,ijl->jkl', deviations, deviations) / num_samples
            P_combined_diag = np.array([np.diag(np.diag(P)) for P in P_combined_sample])

            P_combined_history.append(P_combined_diag)
            means_history.append(mean_cartesian)

            for sample_idx in range(full_state_samples.shape[0]):
                for step in range(full_state_samples.shape[1]):
                    state_history.append(full_state_samples[sample_idx, step])
                    covariance_history.append(np.diagonal(P_combined_diag[step]))
                    time_history.append(time_values[step])
                    bundle_index_history.append(bundle_idx)
                    sample_index_history.append(sample_idx)

        return (np.array(bundle_trajectories).reshape(num_time_steps - 1, num_samples, -1, 7),
                P_combined_history, means_history,
                state_history, control_state_history,
                covariance_history, time_history,
                bundle_index_history, sample_index_history)

    except Exception:
        if queue is not None:
            queue.put(-1)
        traceback.print_exc()
        return None


def generate_monte_carlo_trajectories_parallel(
    backTspan, time_steps, num_time_steps, num_bundles,
    sigmas_combined, new_lam_bundles, mu, F, c, m0, g0,
    num_samples=1000, num_workers=4
):
    P_pos = np.eye(3) * 0.01
    P_vel = np.eye(3) * 0.0001
    P_mass = np.array([[0.0001]])
    P_init = np.block([
        [P_pos, np.zeros((3, 3)), np.zeros((3, 1))],
        [np.zeros((3, 3)), P_vel, np.zeros((3, 1))],
        [np.zeros((1, 3)), np.zeros((1, 3)), P_mass]
    ])
    P_control = np.eye(7) * 0.001

    # ✅ Show progress per sample
    total = num_bundles * (num_time_steps - 1) * num_samples

    manager = Manager()
    q = manager.Queue()
    listener = Process(target=_listener, args=(q, total))
    listener.start()

    args_list = [
        (i, time_steps, num_time_steps, sigmas_combined, new_lam_bundles,
         mu, F, c, m0, g0, num_samples, P_init, P_control, backTspan, q)
        for i in range(num_bundles)
    ]

    with Pool(num_workers) as pool:
        results = pool.map(_solve_mc_single_bundle, args_list)

    listener.join()

    # Unpack and merge
    trajectories, P_hist, means_hist = [], [], []
    X_rows, y_rows, cov_rows, time_rows, b_rows, s_rows = [], [], [], [], [], []

    for res in results:
        if res is None:
            continue
        traj, P, M, X, y, cov, t, b, s = res
        trajectories.append(traj)
        P_hist.append(P)
        means_hist.append(M)
        X_rows.extend(X)
        y_rows.extend(y)
        cov_rows.extend(cov)
        time_rows.extend(t)
        b_rows.extend(b)
        s_rows.extend(s)

    X = np.hstack((
        np.array(time_rows).reshape(-1, 1),
        np.array(X_rows).reshape(-1, 7),
        np.array(cov_rows).reshape(-1, 7),
        np.array(b_rows).reshape(-1, 1),
        np.array(s_rows).reshape(-1, 1)
    ))

    y = np.vstack(y_rows)

    mee_state_subset = X[:, 1:7]
    _, unique_indices = np.unique(mee_state_subset, axis=0, return_index=True)
    X_unique = X[unique_indices]
    y_unique = y[unique_indices]

    sort_indices = np.lexsort((X_unique[:, 0], X_unique[:, -1], X_unique[:, -2]))
    X_sorted = X_unique[sort_indices]
    y_sorted = y_unique[sort_indices]

    return (
        np.array(trajectories),
        np.array(P_hist),
        np.array(means_hist),
        X_sorted,
        y_sorted
    )
