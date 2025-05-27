import numpy as np
import scipy.integrate
import pandas as pd
import rv2mee
import mee2rv
import odefunc

def propagate_sensitivity_with_evolving_mean(
    sigmas_combined, y_sorted,
    backTspan, time_steps, num_time_steps,
    mu, F, c, m0, g0
):
    forwardTspan = backTspan[::-1]
    time = [forwardTspan[time_steps[i]] for i in range(num_time_steps)]
    lam_std = np.sqrt(0.001)
    results = []

    for j in range(num_time_steps - 1):
        tstart, tend = time[j], time[j + 1]
        sub_times = np.linspace(tstart, tend, 11)  # 10 subintervals
        sigma_combined = sigmas_combined[0, :, :, j]  # bundle 0

        for sigma_idx in range(sigma_combined.shape[0]):
            # Get shared position, velocity, and mass (only reused at j=0, k=0)
            r0 = sigma_combined[sigma_idx, :3]
            v0 = sigma_combined[sigma_idx, 3:6]
            mass0 = sigma_combined[sigma_idx, 6]
            shared_initial_state = rv2mee.rv2mee(np.array([r0]), np.array([v0]), mu).flatten()
            shared_initial_state = np.append(shared_initial_state, mass0)

            for k in range(10):
                t0, t1 = sub_times[k], sub_times[k + 1]

                # Look up control vector from y_sorted
                mask = (np.abs(y_sorted[:, 0] - t0) < 1e-8) & (y_sorted[:, 16] == sigma_idx)
                lam_row = y_sorted[mask]
                if lam_row.shape[0] == 0:
                    raise ValueError(f"No control data found for time {t0} and sigma {sigma_idx}")
                lam_mean = lam_row[0, -7:]

                lam_plus = lam_mean + 3 * lam_std
                lam_minus = lam_mean - 3 * lam_std

                for label, lam in zip(["mean", "plus3", "minus3"], [lam_mean, lam_plus, lam_minus]):
                    # Use shared initial state only for the first interval and first subinterval
                    if j == 0 and k == 0:
                        S = np.concatenate([shared_initial_state, lam])
                    else:
                        # Continue from last propagation step of this branch
                        if label == "mean":
                            prev_state = S_mean_final
                        elif label == "plus3":
                            prev_state = S_plus3_final
                        else:
                            prev_state = S_minus3_final
                        S = np.concatenate([prev_state[:7], lam])

                    func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)
                    t_eval = np.linspace(t0, t1, 20)

                    Sf = scipy.integrate.solve_ivp(func, [t0, t1], S, method='RK45',
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

                        # Store final state for next subinterval of each control type
                        if label == "mean":
                            S_mean_final = Sf.y[:, -1].copy()
                        elif label == "plus3":
                            S_plus3_final = Sf.y[:, -1].copy()
                        else:
                            S_minus3_final = Sf.y[:, -1].copy()

    return pd.DataFrame(results, columns=[
        "interval", "subinterval", "sigma_idx", "lam_type", "time",
        "x", "y", "z", "vx", "vy", "vz", "mass",
        "sampled_lam_0", "sampled_lam_1", "sampled_lam_2", "sampled_lam_3",
        "sampled_lam_4", "sampled_lam_5", "sampled_lam_6",
        "mean_lam_0", "mean_lam_1", "mean_lam_2", "mean_lam_3",
        "mean_lam_4", "mean_lam_5", "mean_lam_6"
    ])
