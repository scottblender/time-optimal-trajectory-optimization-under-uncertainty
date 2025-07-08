import os
import numpy as np
from multiprocessing import Pool, Manager, Process
import traceback
from tqdm import tqdm
import joblib
import sys
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from odefunc import odefunc
from mee2rv import mee2rv
from rv2mee import rv2mee


def listener(q, total):
    with tqdm(total=total, desc="Total integration progress") as pbar:
        completed = 0
        while completed < total:
            msg = q.get()
            if msg == 1:
                pbar.update(1)
                completed += 1
            elif msg == -1:
                completed += 1
                pbar.write("[WARN] Subprocess reported failure.")


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


def sample_within_bounds(mean, cov, max_tries=100):
    for _ in range(max_tries):
        sample = np.random.multivariate_normal(mean, cov)
        z_score = np.abs(sample - mean) / np.sqrt(np.diag(cov))
        if np.all(z_score <= 3):
            return sample
    return mean


def _solve_entire_bundle_to_disk(bundle_idx, time_steps, sigmas_combined, new_lam_bundles, m_b,
                                 mu, F, c, m0, g0, Wm, Wc, backTspan, out_dir,
                                 substeps=5, evals_per_substep=10, progress_queue=None):
    try:
        forwardTspan = backTspan[::-1]
        num_time_steps = len(time_steps)
        num_sigmas = sigmas_combined.shape[1]

        X_rows, y_rows = [], []
        all_cartesian = [[] for _ in range(num_sigmas)]

        for time_idx in range(num_time_steps - 1):
            t0_idx = time_steps[time_idx]
            t1_idx = time_steps[time_idx + 1]
            t0 = forwardTspan[t0_idx]
            t1 = forwardTspan[t1_idx]

            P_lam = np.eye(7) * 0.001
            lam0 = new_lam_bundles[t0_idx, :, bundle_idx]
            prev_lam_mean = lam0.copy()

            eci_state = sigmas_combined[bundle_idx, :, :, time_idx]
            r = eci_state[:, :3]
            v = eci_state[:, 3:6]
            mass = eci_state[:, 6]
            mee = rv2mee(r, v, mu)
            state = np.hstack([mee, mass[:, None], np.tile(lam0, (num_sigmas, 1))])

            for s_idx in range(substeps):
                tau0 = t0 + (t1 - t0) * s_idx / substeps
                tau1 = t0 + (t1 - t0) * (s_idx + 1) / substeps
                t_eval = np.linspace(tau0, tau1, evals_per_substep)

                lam = lam0 if s_idx == 0 else sample_within_bounds(prev_lam_mean, P_lam)
                prev_lam_mean = lam.copy()
                state[:, 7:] = lam

                def f(t, x_batch):
                    return np.array([odefunc(t, x_i, mu, F, c, m0, g0) for x_i in x_batch])

                result = custom_rk45_batch(f, t_eval, state)
                state = result[:, -1, :]

                for sigma_idx in range(num_sigmas):
                    p, f_, g, h, k, L = result[sigma_idx, :, 0], result[sigma_idx, :, 1], result[sigma_idx, :, 2], \
                                        result[sigma_idx, :, 3], result[sigma_idx, :, 4], result[sigma_idx, :, 5]
                    r_eval, v_eval = mee2rv(p, f_, g, h, k, L, mu)
                    mass_eval = result[sigma_idx, :, 6]
                    cartesian = np.hstack([r_eval, v_eval, mass_eval[:, None]])
                    all_cartesian[sigma_idx].append(cartesian)

                    for step in range(evals_per_substep):
                        t_val = t_eval[step]
                        state_val = result[sigma_idx, step, :7]
                        lam_val = result[sigma_idx, step, 7:]
                        X_rows.append(np.hstack([t_val, state_val, np.zeros(7), bundle_idx, sigma_idx]))
                        y_rows.append(lam_val)

                if progress_queue is not None:
                    progress_queue.put(1)

        all_cartesian = [np.vstack(steps) for steps in all_cartesian]
        all_cartesian = np.stack(all_cartesian)
        total_steps = all_cartesian.shape[1]

        means = np.sum(Wm[:, None, None] * all_cartesian, axis=0)
        deviations = all_cartesian - means[None, :, :]
        P_all = np.einsum("i,ijk,ijl->jkl", Wc, deviations, deviations)
        P_diag = np.array([np.diag(np.diag(P)) for P in P_all])
        cov_diag = np.array([np.diag(P) for P in P_diag])
        cov_diag_repeated = np.repeat(cov_diag, num_sigmas, axis=0)

        X_rows_np = np.array(X_rows)
        y_rows_np = np.array(y_rows)
        X_rows_np[:, 8:15] = cov_diag_repeated
        time_vector = np.array([[row[0]] for row in X_rows])  

        joblib.dump({
            "trajectories": all_cartesian,
            "P_combined_history": P_diag,
            "means_history": means,
            "X": X_rows_np,
            "y": y_rows_np,
            "time_history": time_vector  
        }, os.path.join(out_dir, f"bundle_{bundle_idx:03d}.pkl"))

        return "ok"
    except Exception as e:
        if progress_queue is not None:
            progress_queue.put(-1)
        tqdm.write(f"[ERROR] in _solve_entire_bundle_to_disk | bundle={bundle_idx}")
        traceback.print_exc()
        return None


def _solve_entire_bundle_to_disk_wrapper(args):
    return _solve_entire_bundle_to_disk(*args)


def solve_trajectories_with_covariance_parallel_with_progress(
    backTspan, time_steps, num_time_steps, num_bundles,
    sigmas_combined, new_lam_bundles, m_b, mu, F, c, m0, g0,
    Wm, Wc, num_workers=4, output_dir="bundle_outputs",
    substeps=5, evals_per_substep=10
):
    import shutil

    if os.path.exists(output_dir):
        for f in glob.glob(os.path.join(output_dir, "*")):
            try:
                os.remove(f)
            except IsADirectoryError:
                shutil.rmtree(f)
    else:
        os.makedirs(output_dir)

    tqdm.write("Launching trajectory propagation tasks...")

    total_steps = (num_time_steps - 1) * num_bundles * substeps
    manager = Manager()
    progress_queue = manager.Queue()

    listener_process = Process(target=listener, args=(progress_queue, total_steps))
    listener_process.start()

    job_args = [
        (b_idx, time_steps, sigmas_combined, new_lam_bundles, m_b,
         mu, F, c, m0, g0, Wm, Wc, backTspan, output_dir, substeps, evals_per_substep, progress_queue)
        for b_idx in range(num_bundles)
    ]

    batch_size = 10
    for i in range(0, len(job_args), batch_size):
        job_batch = job_args[i:i + batch_size]
        with Pool(processes=min(num_workers, batch_size)) as pool:
            pool.map(_solve_entire_bundle_to_disk_wrapper, job_batch)

    listener_process.join()
    tqdm.write(f"All bundles saved to: {output_dir}")

    all_trajectories, all_P_hist, all_means_hist, all_X, all_y, all_times = [], [], [], [], [], []

    for file in sorted(glob.glob(os.path.join(output_dir, "bundle_*.pkl"))):
        data = joblib.load(file)
        all_trajectories.append(data["trajectories"])
        all_P_hist.append(data["P_combined_history"])
        all_means_hist.append(data["means_history"])
        all_X.append(data["X"])
        all_y.append(data["y"])
        all_times.append(data["time_history"])

        X = np.vstack(all_X)
    y = np.vstack(all_y)
    time_vector = np.vstack(all_times)

    # === Append sigma 0 at t_{k+1} for bundle 0 ===
    if sigmas_combined.shape[3] > 1:
        bundle_idx = 0
        sigma_idx = 0
        t_idx = time_steps[1]
        forwardTspan = backTspan[::-1]
        time_val = forwardTspan[t_idx]

        sigma0_cartesian = sigmas_combined[bundle_idx, sigma_idx, :, t_idx]
        r0 = sigma0_cartesian[:3].reshape(1, -1)
        v0 = sigma0_cartesian[3:6].reshape(1, -1)
        m0_val = m_b[t_idx, bundle_idx]
        lam_val = new_lam_bundles[t_idx, :, bundle_idx]
        mee_state = rv2mee(r0, v0, mu).flatten()
        mee_full = np.hstack((mee_state, m0_val))

        X_extra = np.hstack([time_val, mee_full, np.zeros(7), bundle_idx, 0])
        y_extra = lam_val

        X = np.vstack([X, X_extra])
        y = np.vstack([y, y_extra])

        time_vector = np.vstack([time_vector, np.array([[time_val]])])

    return (
        np.stack(all_trajectories),
        np.stack(all_P_hist),
        np.stack(all_means_hist),
        X,
        y,
        time_vector
    )
