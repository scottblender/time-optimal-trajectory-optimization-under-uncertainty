import numpy as np
import scipy.integrate
import pandas as pd
import rv2mee
import mee2rv
import odefunc

def propagate_sensitivity_with_evolving_mean(
    sigmas_combined, y_sorted,
    backTspan, time_steps, num_time_steps,
    mu, F, c, m0, g0,
    lam_std=100
):
    forwardTspan = backTspan[::-1]
    time = [forwardTspan[time_steps[i]] for i in range(num_time_steps)]
    results = []

    for j in range(num_time_steps - 1):
        tstart, tend = time[j], time[j + 1]
        sub_times = np.linspace(tstart, tend, 11)  # 10 subintervals
        sigma_combined = sigmas_combined[0, :, :, j]  # bundle 0

        for sigma_idx in range(sigma_combined.shape[0]):
            # Get initial Cartesian state
            r0 = sigma_combined[sigma_idx, :3]
            v0 = sigma_combined[sigma_idx, 3:6]
            mass0 = sigma_combined[sigma_idx, 6]

            # Look up initial lambda from y_sorted
            initial_mask = (np.abs(y_sorted[:, 0] - sub_times[0]) < 1e-8) & (y_sorted[:, 16] == sigma_idx)
            initial_lam_row = y_sorted[initial_mask]
            if initial_lam_row.shape[0] == 0:
                raise ValueError(f"No initial control data found for time {sub_times[0]} and sigma {sigma_idx}")
            lam_mean = initial_lam_row[0, -7:]

            # Build initial state in MEE for each branch independently
            mee = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu).flatten()
            state_mean = np.concatenate([mee, [mass0]])
            state_plus3 = state_mean.copy()
            state_minus3 = state_mean.copy()

            lam_mean_prev = lam_mean.copy()
            lam_plus_prev = lam_mean + 3 * lam_std
            lam_minus_prev = lam_mean - 3 * lam_std

            for k in range(10):
                t0, t1 = sub_times[k], sub_times[k + 1]
                t_eval = np.linspace(t0, t1, 20)

                for label in ["mean", "plus3", "minus3"]:
                    if label == "mean":
                        state = state_mean.copy()
                        lam = lam_mean_prev
                    elif label == "plus3":
                        state = state_plus3.copy()
                        lam = lam_plus_prev
                    else:
                        state = state_minus3.copy()
                        lam = lam_minus_prev

                    state_full = np.concatenate([state, lam])

                    func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)
                    Sf = scipy.integrate.solve_ivp(func, [t0, t1], state_full, method='RK45',
                                                   t_eval=t_eval, rtol=1e-6, atol=1e-8)

                    if Sf.success:
                        mee = Sf.y[:6, :].T
                        mass = Sf.y[6, :].reshape(-1, 1)
                        r, v = mee2rv.mee2rv(mee[:, 0], mee[:, 1], mee[:, 2],
                                             mee[:, 3], mee[:, 4], mee[:, 5], mu)
                        full = np.hstack([r, v, mass])

                        for idx in range(full.shape[0]):
                            results.append([
                                j, k, sigma_idx, label, Sf.t[idx],
                                *full[idx],
                                *lam,
                                *lam_mean
                            ])

                        # Final Cartesian state from this subinterval
                        r_final = full[-1, :3]
                        v_final = full[-1, 3:6]
                        m_final = full[-1, 6]
                        mee_final = rv2mee.rv2mee(np.array([r_final]), np.array([v_final]), mu).flatten()

                        if label == "mean":
                            state_mean = np.concatenate([mee_final, [m_final]])
                            lam_mean_prev = Sf.y[-7:, -1]
                        elif label == "plus3":
                            state_plus3 = np.concatenate([mee_final, [m_final]])
                            lam_plus_prev = Sf.y[-7:, -1]
                        else:
                            state_minus3 = np.concatenate([mee_final, [m_final]])
                            lam_minus_prev = Sf.y[-7:, -1]

    return pd.DataFrame(results, columns=[
        "interval", "subinterval", "sigma_idx", "lam_type", "time",
        "x", "y", "z", "vx", "vy", "vz", "mass",
        "sampled_lam_0", "sampled_lam_1", "sampled_lam_2", "sampled_lam_3",
        "sampled_lam_4", "sampled_lam_5", "sampled_lam_6",
        "mean_lam_0", "mean_lam_1", "mean_lam_2", "mean_lam_3",
        "mean_lam_4", "mean_lam_5", "mean_lam_6"
    ])
