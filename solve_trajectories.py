import numpy as np
import scipy.integrate
import rv2mee
import mee2rv
import odefunc

def solve_trajectories_with_covariance(backTspan, time_steps, num_time_steps, num_bundles, sigmas_combined, 
                                       new_lam_bundles, mass_bundles, mu, F, c, m0, g0, Wm, Wc):
    forwardTspan = backTspan[::-1]
    time = [forwardTspan[time_steps[i]] for i in range(num_time_steps)] # Ensure exact indices
    trajectories = []
    P_combined_history = np.zeros((num_bundles, num_time_steps, 10000, 7, 7))  # Increase storage size
    means_history = np.zeros((num_bundles, num_time_steps, 10000, 7))  # Increase storage size
    
    for i in range(num_bundles):
        bundle_trajectories = []
        for j in range(num_time_steps - 1):
            tstart, tend = time[j], time[j + 1]  # Get exact time points from backTspan
            sigma_combined = sigmas_combined[i, :, :, j]
            sigma_point_trajectories = []
            
            # Get the exact indices for tstart and tend
            idx_start = np.where(forwardTspan == tstart)[0][0]
            idx_end = np.where(forwardTspan == tend)[0][0]

            # Get control state and mass at exact time index
            new_lam = new_lam_bundles[idx_start, :, i]
            mass = mass_bundles[idx_start, i]  # Retrieve mass from mass_bundles

            sigma_points_final = []
            num_updates = 10  # Number of control updates
            sub_times = np.linspace(tstart, tend, num_updates + 1)  # Create 10 sub-intervals
            
            for sigma_idx in range(sigma_combined.shape[0]):
                r0, v0 = sigma_combined[sigma_idx, :3], sigma_combined[sigma_idx, 3:6]
                initial_state = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu)
                initial_state = np.append(initial_state, mass)  # Use retrieved mass
                S = np.append(initial_state, new_lam)  # Append control state
                
                full_tspan = []  # Store time values
                full_states = []  # Store all state values
                mass_history = []  # Store mass values

                for k in range(num_updates):
                    tsub_start, tsub_end = sub_times[k], sub_times[k + 1]

                    # Sample new control state from a Gaussian
                    new_lam = np.random.multivariate_normal(new_lam, np.eye(7) * 0.001)

                    # Retrieve updated mass from mass_bundles
                    mass = mass_bundles[idx_start, i]  # Ensure proper time step indexing

                    # Update control state and mass in the system state vector
                    S[-7:] = new_lam
                    S[6] = mass  # Update mass in state

                    func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)
                    tspan = np.linspace(tsub_start, tsub_end, 100)  # More refined time steps

                    try:
                        Sf = scipy.integrate.solve_ivp(func, [tsub_start, tsub_end], S, method='RK45', rtol=1e-6, atol=1e-8, t_eval=tspan)
                        if Sf.success:
                            full_tspan.extend(Sf.t)  # Store time values
                            full_states.append(Sf.y.T)  # Store state values
                            mass_history.append(Sf.y[6, :])  # Directly store mass from integration

                            S = Sf.y[:, -1]  # Use last state as next initial condition
                            new_lam = Sf.y[-7:, -1]  # Update new_lam from last control states
                    except Exception:
                        continue

                # Convert accumulated states into a trajectory
                full_states = np.vstack(full_states)  # Shape: (total time steps, 7+7)
                mass_history = np.concatenate(mass_history)  # Flatten mass array

                r_new, v_new = mee2rv.mee2rv(full_states[:, 0], full_states[:, 1], full_states[:, 2], 
                                             full_states[:, 3], full_states[:, 4], full_states[:, 5], mu)

                trajectory = np.hstack((r_new, v_new, mass_history.reshape(-1, 1)))  # Append mass to trajectory
                sigma_point_trajectories.append(trajectory)
                sigma_points_final.append(trajectory)

            sigma_points_final = np.array(sigma_points_final)  # Shape: (13, total_time_steps, 7)
            mean_state = np.sum(Wm[:, np.newaxis, np.newaxis] * sigma_points_final, axis=0)  # Compute mean
            deviations = sigma_points_final - mean_state[np.newaxis, :, :]  # Compute deviations
            
            # Compute weighted covariance
            P_combined = np.einsum('i,ijk,ijl->jkl', Wc, deviations, deviations)

            P_combined_history[i, j, :len(full_tspan), :, :] = P_combined[:len(full_tspan)]
            means_history[i, j, :len(full_tspan), :] = mean_state[:len(full_tspan)]
            bundle_trajectories.append(sigma_point_trajectories)
        
        trajectories.append(bundle_trajectories)

    return np.array(trajectories), P_combined_history, means_history
