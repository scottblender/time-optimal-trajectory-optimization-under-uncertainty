import numpy as np
import scipy.integrate
import rv2mee
import mee2rv
import odefunc

def solve_trajectories_with_covariance(backTspan, time_steps, num_time_steps, num_bundles, sigmas_combined, 
                                       new_lam_bundles, mass_bundles, mu, F, c, m0, g0, Wm, Wc):
    forwardTspan = backTspan[::-1]
    time = [forwardTspan[time_steps[i]] for i in range(num_time_steps)]  # Ensure exact indices
    trajectories = []

    # Storage for covariance, means, and training data
    P_combined_history = []
    means_history = []
    state_history = []
    control_state_history = []
    covariance_history = []
    mean_state_history = []
    time_history = []  # Store time information
    time_match_indices = []  # Store matching indices

    for i in range(num_bundles):
        bundle_trajectories = []
        for j in range(num_time_steps - 1):
            tstart, tend = time[j], time[j + 1]
            sigma_combined = sigmas_combined[i, :, :, j]
            sigma_point_trajectories = []

            idx_start = np.where(forwardTspan == tstart)[0][0]
            new_lam = new_lam_bundles[idx_start, :, i]  # Get control state at exact time index

            full_state_sigma_points = []
            cartesian_sigma_points = []

            num_updates = 10  # Number of control updates
            sub_times = np.linspace(tstart, tend, num_updates + 1)

            full_P_combined = []
            full_means = []

            for sigma_idx in range(sigma_combined.shape[0]):
                # Initialize time_values for each sigma point inside the loop
                time_values = []

                r0, v0, mass = sigma_combined[sigma_idx, :3], sigma_combined[sigma_idx, 3:6], sigma_combined[sigma_idx, 6]
                initial_state = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu)
                initial_state = np.append(initial_state, mass)  # Append mass to MEE state
                S = np.append(initial_state, new_lam)  # Append control states

                full_states = []  # Store state trajectory for this sigma point
                cartesian_states = []  # Store Cartesian state trajectory

                for k in range(num_updates):
                    tsub_start, tsub_end = sub_times[k], sub_times[k + 1]

                    if k == 0:
                        new_lam_current = new_lam
                    else:
                        new_lam_current = np.random.multivariate_normal(new_lam, np.eye(7) * 0.001)

                    S[-7:] = new_lam_current
                    func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)
                    tspan = np.linspace(tsub_start, tsub_end, 20)

                    Sf = scipy.integrate.solve_ivp(func, [tsub_start, tsub_end], S, method='RK45', rtol=1e-6, atol=1e-8, t_eval=tspan)
                    
                    if Sf.success:
                        full_states.append(Sf.y.T)  # Store state trajectory
                        time_values.append(Sf.t)  # Store time steps for this sigma point
                        S = Sf.y[:, -1]  # Use last state for next step

                full_states = np.vstack(full_states)  # Shape: (total time steps, 14)
                full_state_sigma_points.append(full_states[:, :7])  # Store first 7 elements (MEE + mass)

                # Convert time list to NumPy array AFTER the loop
                time_values = np.hstack(time_values)  # Convert list of arrays into a single 1D NumPy array

                # Convert to Cartesian for storage
                r_new, v_new = mee2rv.mee2rv(full_states[:, 0], full_states[:, 1], full_states[:, 2], 
                                             full_states[:, 3], full_states[:, 4], full_states[:, 5], mu)
                cartesian_state = np.hstack((r_new, v_new, full_states[:, 6].reshape(-1, 1)))  # Append mass
                cartesian_states.append(cartesian_state)

                sigma_point_trajectories.append(cartesian_state)
                cartesian_sigma_points.append(cartesian_state)

                control_state_history.append(full_states[:, -7:])  # Store control states

            full_state_sigma_points = np.array(full_state_sigma_points)  # Shape: (15, total_time_steps, 7)
            cartesian_sigma_points = np.array(cartesian_sigma_points)  # Shape: (15, total_time_steps, 7)

            # Compute mean for MEE-based full state
            mean_state = np.sum(Wm[:, np.newaxis, np.newaxis] * full_state_sigma_points, axis=0)

            # Compute covariance for MEE-based full state
            deviations = full_state_sigma_points - mean_state[np.newaxis, :, :]
            P_combined = np.einsum('i,ijk,ijl->jkl', Wc, deviations, deviations)

            # Compute mean for Cartesian state
            mean_cartesian = np.sum(Wm[:, np.newaxis, np.newaxis] * cartesian_sigma_points, axis=0)

            # Compute covariance for Cartesian state
            deviations_cartesian = cartesian_sigma_points - mean_cartesian[np.newaxis, :, :]
            P_combined_cartesian = np.einsum('i,ijk,ijl->jkl', Wc, deviations_cartesian, deviations_cartesian)

            full_P_combined.append(P_combined_cartesian)
            full_means.append(mean_cartesian)
            bundle_trajectories.append(sigma_point_trajectories)

            # Store history
            for sigma_idx in range(full_state_sigma_points.shape[0]):  
                for step in range(full_state_sigma_points.shape[1]):  
                    state_history.append(full_state_sigma_points[sigma_idx, step])  
                    mean_state_history.append(mean_state[step])  
                    covariance_history.append(np.diagonal(P_combined[step], axis1=0, axis2=1))  
                    time_history.append(time_values[step])  # Store the correct time for each sigma point

        P_combined_history.append(full_P_combined)
        means_history.append(full_means)
        trajectories.append(bundle_trajectories)

    time_history = np.hstack(time_history).reshape(-1, 1)  # Convert to column vector
    time_match_indices = np.where(np.isin(time_history.flatten(), forwardTspan))[0]

    # Convert training data into numpy arrays
    state_history = np.vstack(state_history)  # (N_samples, 7) (MEE-based)
    control_state_history = np.vstack(control_state_history)  # (N_samples, 7)
    covariance_history = np.vstack(covariance_history)  # (N_samples, 7) - Only diagonal elements
    mean_state_history = np.vstack(mean_state_history)  # (N_samples, 7)

    X = np.hstack((time_history, state_history, covariance_history, mean_state_history))
    y = control_state_history  

    return np.array(trajectories), np.array(P_combined_history), np.array(means_history), X, y, time_match_indices
