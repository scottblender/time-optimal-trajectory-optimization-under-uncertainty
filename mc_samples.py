import numpy as np
import scipy.integrate
import rv2mee
import mee2rv
import odefunc

def monte_carlo_sub_trajectories(num_samples, backTspan, time_steps, num_time_steps, num_bundles, sigmas_combined, P_combined, new_lam_bundles, mu, F, c, m0, g0):
    """
    Perform Monte Carlo sampling using the original mean (sigma point 0) and covariance (P_combined) 
    to generate new sub-trajectories and compute mean and covariance at each step.
    """
    time = [backTspan[time_steps[i]] for i in range(num_time_steps)][::-1]
    mc_trajectories = np.zeros((num_bundles, num_samples, num_time_steps, 1000, 6))  # (bundles, samples, timesteps, integration points, states)
    mc_means_history = np.zeros((num_bundles, num_time_steps, 1000, 6))
    mc_covariances_history = np.zeros((num_bundles, num_time_steps, 1000, 6, 6))
    
    for i in range(num_bundles):
        time_index = np.argmin(np.abs(backTspan - time[0]))  # Use first time step index
        new_lam = new_lam_bundles[time_index, :, i]

        # Extract the original mean from sigma point 0 at the first integration point
        original_mean = sigmas_combined[i, 0, :, 0]  # Shape: (6,)

        # Sample from original covariance at the first integration point
        initial_samples = np.random.multivariate_normal(original_mean, P_combined, num_samples)  # Shape: (num_samples, 6)

        for k in range(num_samples):
            sampled_state = initial_samples[k]  # Shape: (6,)
            r0, v0 = sampled_state[:3], sampled_state[3:]
            initial_state = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu)
            initial_state = np.append(initial_state, np.array([1]))  # Append a perturbation indicator
            S = np.append(initial_state, new_lam)  # Append control parameters

            func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)
            
            # Propagate trajectory through all time steps
            for j in range(num_time_steps - 1):
                tstart, tend = time[j], time[j + 1]
                tspan = np.linspace(tstart, tend, 1000)

                try:
                    Sf = scipy.integrate.solve_ivp(func, [tstart, tend], S, method='RK45', rtol=1e-6, atol=1e-8, t_eval=tspan)
                    if Sf.success:
                        r_new, v_new = mee2rv.mee2rv(Sf.y[0, :], Sf.y[1, :], Sf.y[2, :], Sf.y[3, :], Sf.y[4, :], Sf.y[5, :], mu)
                        mc_trajectories[i, k, j, :, :] = np.hstack((r_new, v_new))
                        S = Sf.y[:, -1]  # Update state for next time step
                except Exception:
                    continue
        
        # Compute mean and covariance at each integration point
        for j in range(num_time_steps - 1):
            for t in range(1000):  # Loop over integration points
                states_at_t = mc_trajectories[i, :, j, t, :]  # Shape: (num_samples, 6)
                mean_t = np.mean(states_at_t, axis=0)  # Compute mean
                deviations = states_at_t - mean_t
                covariance_t = np.einsum('ni,nj->ij', deviations, deviations) / (num_samples - 1)  # Compute covariance
                
                mc_means_history[i, j, t, :] = mean_t
                mc_covariances_history[i, j, t, :, :] = covariance_t

    return mc_trajectories, mc_means_history, mc_covariances_history
