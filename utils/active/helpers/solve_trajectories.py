import numpy as np
import scipy.integrate
import rv2mee
import mee2rv
import odefunc

def solve_trajectories_with_covariance(
    backTspan, time_steps, num_time_steps, num_bundles, sigmas_combined,
    new_lam_bundles, mass_bundles, mu, F, c, m0, g0, Wm, Wc
):
    forwardTspan = backTspan[::-1]
    time = [forwardTspan[time_steps[i]] for i in range(num_time_steps)]
    trajectories = []

    P_combined_history = []
    means_history = []
    state_history = []
    control_state_history = []
    covariance_history = []
    time_history = []
    bundle_index_history = []
    sigma_point_index_history = []

    def sample_within_bounds(mean, cov, max_tries=100):
        for _ in range(max_tries):
            sample = np.random.multivariate_normal(mean, cov)
            z_score = np.abs(sample - mean) / np.sqrt(np.diag(cov))
            if np.all(z_score <= 3):
                return sample
        return mean  # fallback

    for i in range(num_bundles):
        bundle_trajectories = []
        for j in range(num_time_steps - 1):
            tstart, tend = time[j], time[j + 1]
            sigma_combined = sigmas_combined[i, :, :, j]
            sigma_point_trajectories = []

            idx_start = np.where(forwardTspan == tstart)[0][0]
            new_lam = new_lam_bundles[idx_start, :, i]
            P_lam = np.eye(7) * 0.001

            full_state_sigma_points = []
            cartesian_sigma_points = []

            num_updates = 10
            sub_times = np.linspace(tstart, tend, num_updates + 1)

            full_P_combined = []
            full_means = []

            for sigma_idx in range(sigma_combined.shape[0]):
                time_values = []

                r0, v0, mass = sigma_combined[sigma_idx, :3], sigma_combined[sigma_idx, 3:6], sigma_combined[sigma_idx, 6]
                initial_state = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu)
                initial_state = np.append(initial_state, mass)
                S = np.append(initial_state, new_lam)

                full_states = []
                cartesian_states = []

                prev_lam_mean = new_lam.copy()

                for k in range(num_updates):
                    tsub_start, tsub_end = sub_times[k], sub_times[k + 1]

                    if k == 0:
                        new_lam_current = new_lam
                    else:
                        new_lam_current = sample_within_bounds(prev_lam_mean, P_lam)

                    prev_lam_mean = new_lam_current.copy()
                    S[-7:] = new_lam_current

                    func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)
                    tspan = np.linspace(tsub_start, tsub_end, 20)

                    Sf = scipy.integrate.solve_ivp(func, [tsub_start, tsub_end], S, method='RK45',
                                                   rtol=1e-6, atol=1e-8, t_eval=tspan)

                    if Sf.success:
                        full_states.append(Sf.y.T)
                        time_values.append(Sf.t)
                        S = Sf.y[:, -1]

                full_states = np.vstack(full_states)
                full_state_sigma_points.append(full_states[:, :7])
                time_values = np.hstack(time_values)

                r_new, v_new = mee2rv.mee2rv(full_states[:, 0], full_states[:, 1], full_states[:, 2],
                                             full_states[:, 3], full_states[:, 4], full_states[:, 5], mu)
                cartesian_state = np.hstack((r_new, v_new, full_states[:, 6].reshape(-1, 1)))
                cartesian_states.append(cartesian_state)

                sigma_point_trajectories.append(cartesian_state)
                cartesian_sigma_points.append(cartesian_state)
                control_state_history.append(full_states[:, -7:])

            full_state_sigma_points = np.array(full_state_sigma_points)
            cartesian_sigma_points = np.array(cartesian_sigma_points)

            mean_state = np.sum(Wm[:, np.newaxis, np.newaxis] * full_state_sigma_points, axis=0)
            deviations = full_state_sigma_points - mean_state[np.newaxis, :, :]
            P_combined = np.einsum('i,ijk,ijl->jkl', Wc, deviations, deviations)
            P_combined_diag = np.array([np.diag(np.diag(P)) for P in P_combined])

            mean_cartesian = np.sum(Wm[:, np.newaxis, np.newaxis] * cartesian_sigma_points, axis=0)
            deviations_cartesian = cartesian_sigma_points - mean_cartesian[np.newaxis, :, :]
            P_combined_cartesian = np.einsum('i,ijk,ijl->jkl', Wc, deviations_cartesian, deviations_cartesian)
            P_combined_cartesian_diag = np.array([np.diag(np.diag(P)) for P in P_combined_cartesian])

            full_P_combined.append(P_combined_cartesian_diag)
            full_means.append(mean_cartesian)

            # print(f"\n=== Position Comparison at Initial and Final Step — Segment {j} for Bundle {i} ===")
            # for sigma_idx, cartesian_state in enumerate(cartesian_sigma_points):
            #     r_start = cartesian_state[1, :3]
            #     r_end = cartesian_state[-1, :3]
            #     mee_start = full_state_sigma_points[sigma_idx][1, :3]  # p/f/g
            #     mee_end = full_state_sigma_points[sigma_idx][-1, :3]
            #     print(f"Sigma {sigma_idx:2d}:\n  Step +1: Cartesian r = {r_start}, MEE p/f/g = {mee_start}\n  Step -1: Cartesian r = {r_end}, MEE p/f/g = {mee_end}")

            bundle_trajectories.append(sigma_point_trajectories)

            for sigma_idx in range(full_state_sigma_points.shape[0]):
                for step in range(full_state_sigma_points.shape[1]):
                    state_history.append(full_state_sigma_points[sigma_idx, step])
                    covariance_history.append(np.diagonal(P_combined_diag[step]))
                    time_history.append(time_values[step])
                    bundle_index_history.append(32)
                    sigma_point_index_history.append(sigma_idx)

            # print("\n=== Comparing Stored X Values to Propagated Values at Final Step ===")
            # for sigma_idx in range(15):
            #     propagated_final = full_state_sigma_points[sigma_idx][-1]
            #     X_match = [
            #         (len(state_history) - 1 - k, s)
            #         for k, s in enumerate(state_history[::-1])
            #         if sigma_point_index_history[-1 - k] == sigma_idx
            #     ]
            #     if X_match:
            #         _, x_state = X_match[0]
            #         diff = np.linalg.norm(propagated_final - x_state)
            #         print(f"Sigma {sigma_idx:2d}: Δ = {diff:.6e}")
            #     else:
            #         print(f"Sigma {sigma_idx:2d}: ❌ No match in X")

        P_combined_history.append(full_P_combined)
        means_history.append(full_means)
        trajectories.append(bundle_trajectories)

    time_history = np.hstack(time_history).reshape(-1, 1)
    state_history = np.vstack(state_history)
    control_state_history = np.vstack(control_state_history)
    covariance_history = np.vstack(covariance_history)
    bundle_index_history = np.array(bundle_index_history).reshape(-1, 1)
    sigma_point_index_history = np.array(sigma_point_index_history).reshape(-1, 1)

    X = np.hstack((time_history, state_history, covariance_history, bundle_index_history, sigma_point_index_history))
    y = control_state_history

    _, unique_indices = np.unique(X[:, [0, -1]], axis=0, return_index=True)
    X_unique = X[unique_indices]
    y_unique = y[unique_indices]

    sort_indices = np.lexsort((X_unique[:, 0], X_unique[:, -1], X_unique[:, -2]))
    X_sorted = X_unique[sort_indices]
    y_sorted = y_unique[sort_indices]

    if sigmas_combined.shape[3] > 1:
        sigma0_cartesian = sigmas_combined[0, 0, :, 1]
        r0 = sigma0_cartesian[:3].reshape(1, -1)
        v0 = sigma0_cartesian[3:6].reshape(1, -1)

        time_idx = time_steps[1]
        m0 = mass_bundles[time_idx, 0]
        mee_state = rv2mee.rv2mee(r0, v0, mu).flatten()
        mee_full = np.hstack((mee_state, m0))

        lam_val = new_lam_bundles[time_idx, :, 0]
        time_val = forwardTspan[time_idx]

        X_extra = np.hstack([
            time_val,
            mee_full,
            np.zeros(7),
            32,
            0
        ])
        y_extra = lam_val

        X_sorted = np.vstack([X_sorted, X_extra])
        y_sorted = np.vstack([y_sorted, y_extra])

    return np.array(trajectories), np.array(P_combined_history), np.array(means_history), X_sorted, y_sorted