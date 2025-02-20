import numpy as np
import scipy.integrate
import rv2mee
import mee2rv
import odefunc

def solve_trajectories_with_covariance(backTspan, time_steps, num_time_steps, num_bundles, sigmas_combined, new_lam_bundles, mu, F, c, m0, g0, P_combined_initial):
    # Select times based on time_steps
    time = [backTspan[time_steps[i]] for i in range(num_time_steps)]
    time = time[::-1]

    # Initialize an empty list to store trajectories for all bundles
    trajectories = []
    
    # Initialize an array to store P_combined at each time step for each bundle
    P_combined_history = np.zeros((num_bundles, num_time_steps, P_combined_initial.shape[0], P_combined_initial.shape[1]))

    # Loop over each bundle and each time step to solve the IVP
    for i in range(num_bundles):  # Loop over each bundle
        # List to store the sigma points trajectories for the current bundle
        bundle_trajectories = []

        # Loop over the time steps and initialize the covariance for each time step
        for j in range(num_time_steps-1):  # Loop over time steps (ensure range is valid)
            # Define the start and end times for the integration
            tstart = time[j]
            tend = time[j + 1]

            # Extract the current sigma points for position and velocity (combined state)
            sigma_combined = sigmas_combined[i, :, :, j]  # Shape: (13, 6)

            # List to store the trajectories for all sigma points at the current time step
            sigma_point_trajectories = []
            
            # Find the closest index in backTspan for the current time step
            time_index = np.argmin(np.abs(backTspan - time[j]))

            # Extract the lamperture values (new_lam) at the current time step for the current bundle
            new_lam = new_lam_bundles[time_index, :, i]  # Extract 7 elements for lamperture at time step j for bundle i

            # Propagate sigma points and compute covariance at this time step
            sigma_points_final = []

            # Loop over the sigma points
            for sigma_idx in range(sigma_combined.shape[0]):  # Loop through the 13 sigma points
                # Extract position and velocity from the combined sigma point
                r0 = sigma_combined[sigma_idx, :3]  # First 3 elements are position
                v0 = sigma_combined[sigma_idx, 3:]  # Last 3 elements are velocity

                # Convert the position and velocity to modified equinoctial elements
                initial_state = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu)
                initial_state = np.append(initial_state, np.array([1]), axis=0)  # Append the 1 element for perturbation
                S = np.append(initial_state, new_lam)  # Append the lamperture values (new_lam)

                # Define the ODE function for integration
                func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)

                # Define the time span for the ODE solver
                tspan = np.linspace(tstart, tend, 1000)

                try:
                    # Solve the ODE using RK45 method
                    Sf = scipy.integrate.solve_ivp(func, [tstart, tend], S, method='RK45', rtol=1e-6, atol=1e-8, t_eval=tspan)

                    # If the solution is successful, convert the MEE back to RV
                    if Sf.success:
                        r_new, v_new = mee2rv.mee2rv(Sf.y[0, :], Sf.y[1, :], Sf.y[2, :], Sf.y[3, :], Sf.y[4, :], Sf.y[5, :], mu)

                        # Store both position and velocity for the current sigma point at each time step
                        trajectory = np.hstack((r_new, v_new))  # Combine position (r_new) and velocity (v_new)
                        sigma_point_trajectories.append(trajectory)

                        # Store the propagated sigma points for covariance calculation
                        sigma_points_final.append(np.hstack((r_new, v_new)))  # Save final propagated sigma point

                except Exception as e:
                    continue  # In case of error, continue with the next sigma point

            # Compute the covariance matrix using the deviations from the nominal (first) sigma point
            sigma_points_final = np.array(sigma_points_final)
            nominal_trajectory = sigma_points_final[0]  # The first sigma point is the nominal trajectory
            deviations = sigma_points_final[1:] - nominal_trajectory  # Deviations of other sigma points from nominal
            
            # Reshape deviations to (num_samples, num_states) where num_samples = num_sigma_points * num_time_steps
            deviations_reshaped = deviations.reshape(-1, deviations.shape[-1])

            # Compute the covariance matrix (with n-1 because the nominal trajectory is not used for the covariance)
            P_combined = np.cov(deviations_reshaped.T)  # Covariance matrix based on deviations from the nominal trajectory
            
            # Store P_combined for this time step and bundle
            P_combined_history[i, j, :, :] = P_combined

            # Append the trajectories of all sigma points at this time step to the bundle
            bundle_trajectories.append(sigma_point_trajectories)

        # Append the trajectories for the entire bundle
        trajectories.append(bundle_trajectories)

    # Convert trajectories to a numpy array
    return np.array(trajectories), P_combined_history  # Return both trajectories and the time-history of P_combined
