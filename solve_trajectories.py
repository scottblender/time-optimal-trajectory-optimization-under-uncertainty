import numpy as np
import scipy.integrate
import rv2mee
import mee2rv
import odefunc
from filterpy.kalman import MerweScaledSigmaPoints

def solve_trajectories_with_covariance(backTspan, time_steps, num_time_steps, num_bundles, sigmas_combined, new_lam_bundles, mu, F, c, m0, g0, Wm, Wc):
    time = [backTspan[time_steps[i]] for i in range(num_time_steps)][::-1]
    trajectories = []
    P_combined_history = np.zeros((num_bundles, num_time_steps, 1000, 6, 6))
    means_history = np.zeros((num_bundles, num_time_steps, 1000, 6))
    
    for i in range(num_bundles):
        bundle_trajectories = []
        for j in range(num_time_steps - 1):
            tstart, tend = time[j], time[j + 1]
            sigma_combined = sigmas_combined[i, :, :, j]
            sigma_point_trajectories = []
            time_index = np.argmin(np.abs(backTspan - time[j]))
            new_lam = new_lam_bundles[time_index, :, i]
            sigma_points_final = []
            
            for sigma_idx in range(sigma_combined.shape[0]):
                r0, v0 = sigma_combined[sigma_idx, :3], sigma_combined[sigma_idx, 3:]
                initial_state = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu)
                initial_state = np.append(initial_state, np.array([1]), axis=0)
                S = np.append(initial_state, new_lam)
                func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)
                tspan = np.linspace(tstart, tend, 1000)
                
                try:
                    Sf = scipy.integrate.solve_ivp(func, [tstart, tend], S, method='RK45', rtol=1e-6, atol=1e-8, t_eval=tspan)
                    if Sf.success:
                        r_new, v_new = mee2rv.mee2rv(Sf.y[0, :], Sf.y[1, :], Sf.y[2, :], Sf.y[3, :], Sf.y[4, :], Sf.y[5, :], mu)
                        trajectory = np.hstack((r_new, v_new))
                        sigma_point_trajectories.append(trajectory)
                        sigma_points_final.append(np.hstack((r_new, v_new)))
                except Exception:
                    continue
            
            sigma_points_final = np.array(sigma_points_final)  # Shape: (13, 1000, 6)
            mean_state = np.sum(Wm[:, np.newaxis, np.newaxis] * sigma_points_final, axis=0)  # Weighted mean, shape: (1000, 6)
            deviations = sigma_points_final - mean_state[np.newaxis, :, :]  # Deviation, shape: (13, 1000, 6)
            
            P_combined = np.einsum('i,ijk,ijl->jkl', Wc, deviations, deviations)  # Weighted covariance, shape: (1000, 6, 6)
            
            P_combined_history[i, j, :, :, :] = P_combined
            means_history[i, j, :, :] = mean_state
            bundle_trajectories.append(sigma_point_trajectories)
        
        trajectories.append(bundle_trajectories)
    
    return np.array(trajectories), P_combined_history, means_history
