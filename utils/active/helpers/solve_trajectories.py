import os
import numpy as np
import glob
import traceback
import zarr
from tqdm import tqdm
from multiprocessing import Pool, Manager, Process
from scipy.integrate import solve_ivp

from odefunc import odefunc
from mee2rv import mee2rv
from rv2mee import rv2mee


def sample_within_bounds(mean, cov, max_tries=100):
    for _ in range(max_tries):
        sample = np.random.multivariate_normal(mean, cov)
        z = np.abs(sample - mean) / np.sqrt(np.diag(cov))
        if np.all(z <= 3):
            return sample
    return mean


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
                pbar.write("[WARN] Subprocess failed")


def _solve_entire_bundle_to_disk(bundle_idx, time_steps, sigmas_combined, new_lam_bundles, m_b,
                                 mu, F, c, m0, g0, Wm, Wc, backTspan, out_dir,
                                 substeps=10, evals_per_substep=20, progress_queue=None):
    try:
        forwardTspan = backTspan[::-1]
        num_time_steps = len(time_steps)
        num_sigmas = sigmas_combined.shape[1]

        X_rows, y_rows = [], []
        full_cartesian = [[] for _ in range(num_sigmas)]

        for time_idx in range(num_time_steps - 1):
            t0_idx = time_steps[time_idx]
            t1_idx = time_steps[time_idx + 1]
            t0 = forwardTspan[t0_idx]
            t1 = forwardTspan[t1_idx]

            sub_times = np.linspace(t0, t1, substeps + 1)
            P_lam = np.eye(7) * 0.001
            lam0 = new_lam_bundles[t0_idx, :, bundle_idx]

            eci_state = sigmas_combined[bundle_idx, :, :, time_idx]
            r = eci_state[:, :3]
            v = eci_state[:, 3:6]
            mass = eci_state[:, 6]
            mee = rv2mee(r, v, mu)
            state = np.hstack([mee, mass[:, None]])

            for sigma_idx in range(num_sigmas):
                S = np.hstack([state[sigma_idx], lam0])
                full_states, full_times = [], []
                prev_lam_mean = lam0.copy()

                for s_idx in range(substeps):
                    tau0 = sub_times[s_idx]
                    tau1 = sub_times[s_idx + 1]
                    t_eval = np.linspace(tau0, tau1, evals_per_substep)

                    lam = lam0 if s_idx == 0 else sample_within_bounds(prev_lam_mean, P_lam)
                    prev_lam_mean = lam.copy()
                    S[-7:] = lam

                    func = lambda t, x: odefunc(t, x, mu, F, c, m0, g0)
                    sol = solve_ivp(func, (tau0, tau1), S, method='RK45',
                                    t_eval=t_eval, rtol=1e-3, atol=1e-6)

                    if sol.success:
                        full_states.append(sol.y.T)
                        full_times.append(sol.t)
                        S = sol.y[:, -1]
                    else:
                        raise RuntimeError("solve_ivp failed")

                    if progress_queue is not None:
                        progress_queue.put(1)

                full_states = np.vstack(full_states)
                t_hist = np.hstack(full_times)

                pfg = full_states[:, 0:6].T
                mass_eval = full_states[:, 6]
                r_eval, v_eval = mee2rv(*pfg, mu)
                cartesian = np.hstack([r_eval, v_eval, mass_eval[:, None]])
                full_cartesian[sigma_idx].append(cartesian)

                for step in range(len(t_hist)):
                    X_rows.append(np.hstack([t_hist[step], full_states[step, :7], np.zeros(7), bundle_idx, sigma_idx]))
                    y_rows.append(full_states[step, 7:])

        full_cartesian = [np.vstack(c) for c in full_cartesian]
        full_cartesian = np.stack(full_cartesian, axis=0)

        mean_cartesian = np.sum(Wm[:, None, None] * full_cartesian, axis=0)
        deviations = full_cartesian - mean_cartesian[None, :, :]
        P_combined = np.einsum("i,ijk,ijl->jkl", Wc, deviations, deviations)
        P_diag = np.array([np.diag(np.diag(P)) for P in P_combined])
        cov_diag = np.array([np.diag(P) for P in P_diag])
        cov_diag_repeated = np.repeat(cov_diag, num_sigmas, axis=0)

        X_np = np.array(X_rows)
        y_np = np.array(y_rows)
        X_np[:, 8:15] = cov_diag_repeated

        zarr_path = os.path.join(out_dir, f"bundle_{bundle_idx:03d}.zarr")
        root = zarr.open_group(zarr_path, mode="w")
        compressor = zarr.Blosc(cname="zstd", clevel=3)

        root.create_dataset("trajectories", data=full_cartesian, compressor=compressor, chunks=True)
        root.create_dataset("P_combined_history", data=P_diag, compressor=compressor)
        root.create_dataset("means_history", data=mean_cartesian, compressor=compressor)
        root.create_dataset("X", data=X_np, compressor=compressor)
        root.create_dataset("y", data=y_np, compressor=compressor)

        return "ok"
    except Exception:
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
    substeps=10, evals_per_substep=20
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

    all_trajectories, all_P_hist, all_means_hist, all_X, all_y = [], [], [], [], []

    for file in sorted(glob.glob(os.path.join(output_dir, "bundle_*.zarr"))):
        root = zarr.open(file, mode="r")
        all_trajectories.append(np.array(root["trajectories"]))
        all_P_hist.append(np.array(root["P_combined_history"]))
        all_means_hist.append(np.array(root["means_history"]))
        all_X.append(np.array(root["X"]))
        all_y.append(np.array(root["y"]))

    X = np.vstack(all_X)
    y = np.vstack(all_y)

    _, unique_indices = np.unique(X[:, [0, -1]], axis=0, return_index=True)
    X_unique = X[unique_indices]
    y_unique = y[unique_indices]

    sort_indices = np.lexsort((X_unique[:, 0], X_unique[:, -1], X_unique[:, -2]))
    X_sorted = X_unique[sort_indices]
    y_sorted = y_unique[sort_indices]

    if sigmas_combined.shape[3] > 1:
        forwardTspan = backTspan[::-1]
        t_idx = time_steps[1]
        time_val = forwardTspan[t_idx]

        for bundle_idx in range(num_bundles):
            sigma0_cartesian = sigmas_combined[bundle_idx, 0, :, t_idx]
            r0 = sigma0_cartesian[:3].reshape(1, -1)
            v0 = sigma0_cartesian[3:6].reshape(1, -1)
            m0_val = m_b[t_idx, bundle_idx]
            lam_val = new_lam_bundles[t_idx, :, bundle_idx]
            mee_state = rv2mee(r0, v0, mu).flatten()
            mee_full = np.hstack((mee_state, m0_val))

            X_extra = np.hstack([time_val, mee_full, np.zeros(7), bundle_idx, 0])
            y_extra = lam_val

            X_sorted = np.vstack([X_sorted, X_extra])
            y_sorted = np.vstack([y_sorted, y_extra])

    return (
        np.stack(all_trajectories),
        np.stack(all_P_hist),
        np.stack(all_means_hist),
        X_sorted,
        y_sorted,
        X_sorted[:, 0].reshape(-1, 1)
    )
