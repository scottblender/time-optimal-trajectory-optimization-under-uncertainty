import numpy as np
import scipy.integrate
import rv2mee
import mee2rv
import odefunc

def generate_monte_carlo_trajectories(backTspan, time_steps, num_time_steps, num_bundles,
                                       sigmas_combined, new_lam_bundles, mu, F, c, m0, g0, num_samples):
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
    sample_index_history = []

    # === Define Initial Covariances ===
    P_pos = np.eye(3) * 0.01
    P_vel = np.eye(3) * 0.0001
    P_mass = np.array([[0.0001]])
    P_init = np.block([
        [P_pos, np.zeros((3, 3)), np.zeros((3, 1))],
        [np.zeros((3, 3)), P_vel, np.zeros((3, 1))],
        [np.zeros((1, 3)), np.zeros((1, 3)), P_mass]
    ])
    P_control = np.eye(7) * 0.001

    def sample_within_bounds(mean, cov, max_tries=100):
        for _ in range(max_tries):
            sample = np.random.multivariate_normal(mean, cov)
            z_score = np.abs(sample - mean) / np.sqrt(np.diag(cov))
            if np.all(z_score <= 3):
                return sample
        return mean

    for i in range(num_bundles):
        bundle_trajectories = []
        for j in range(num_time_steps - 1):
            tstart, tend = time[j], time[j + 1]
            sigma_state = sigmas_combined[i, 0, :, j]  # ECI + mass

            new_lam_nominal = new_lam_bundles[j, :, i]

            full_state_samples = []
            cartesian_samples = []

            num_updates = 10
            sub_times = np.linspace(tstart, tend, num_updates + 1)

            P_combined_t = []
            mean_cartesian_t = []

            for sample_idx in range(num_samples):
                if sample_idx == 0:
                    sample_state = sigma_state
                else:
                    sample_state = sample_within_bounds(sigma_state, P_init)

                r0 = sample_state[:3]
                v0 = sample_state[3:6]
                mass = sample_state[6]

                mee_state = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu).flatten()
                S = np.concatenate([mee_state, [mass], new_lam_nominal])

                full_states = []
                time_values = []

                for k in range(num_updates):
                    tsub_start, tsub_end = sub_times[k], sub_times[k + 1]
                    if k == 0:
                        new_lam_k = new_lam_nominal
                    else:
                        new_lam_k = sample_within_bounds(new_lam_nominal, P_control)

                    S[-7:] = new_lam_k
                    func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)
                    tspan = np.linspace(tsub_start, tsub_end, 20)

                    Sf = scipy.integrate.solve_ivp(func, [tsub_start, tsub_end], S, t_eval=tspan,
                                                   rtol=1e-6, atol=1e-8)
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

            full_state_samples = np.array(full_state_samples)
            cartesian_samples = np.array(cartesian_samples)

            mean_cartesian = np.mean(cartesian_samples, axis=0)
            deviations = cartesian_samples - mean_cartesian[np.newaxis, :, :]
            P_combined_sample = np.einsum('ijk,ijl->jkl', deviations, deviations) / num_samples
            P_combined_diag = np.array([np.diag(np.diag(P)) for P in P_combined_sample])

            P_combined_t.append(P_combined_diag)
            mean_cartesian_t.append(mean_cartesian)

            for sample_idx in range(full_state_samples.shape[0]):
                for step in range(full_state_samples.shape[1]):
                    state_history.append(full_state_samples[sample_idx, step])
                    covariance_history.append(np.diagonal(P_combined_diag[step]))
                    time_history.append(time_values[step])
                    bundle_index_history.append(32)
                    sample_index_history.append(sample_idx)

        P_combined_history.append(P_combined_t)
        means_history.append(mean_cartesian_t)

        bundle_trajectories = np.array(bundle_trajectories).reshape(num_time_steps - 1, num_samples, -1, 7)
        trajectories.append(np.transpose(bundle_trajectories, (0, 1, 2, 3)))

    time_history = np.hstack(time_history).reshape(-1, 1)
    state_history = np.vstack(state_history)
    control_state_history = np.vstack(control_state_history)
    covariance_history = np.vstack(covariance_history)
    bundle_index_history = np.array(bundle_index_history).reshape(-1, 1)
    sample_index_history = np.array(sample_index_history).reshape(-1, 1)

    X = np.hstack((time_history, state_history, covariance_history, bundle_index_history, sample_index_history))
    y = control_state_history

    mee_state_subset = X[:, 1:7]
    _, unique_indices = np.unique(mee_state_subset, axis=0, return_index=True)
    X_unique = X[unique_indices]
    y_unique = y[unique_indices]

    sort_indices = np.lexsort((X_unique[:, 0], X_unique[:, -1], X_unique[:, -2]))
    X_sorted = X_unique[sort_indices]
    y_sorted = y_unique[sort_indices]

    return np.array(trajectories), np.array(P_combined_history), np.array(means_history), X_sorted, y_sorted
