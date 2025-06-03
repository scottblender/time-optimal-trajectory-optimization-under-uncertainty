import numpy as np
import scipy.integrate
import pandas as pd
import rv2mee
import mee2rv
import odefunc

def propagate_sensitivity_from_initial_lam_only(
    X_sorted, y_sorted, trajectories, backTspan,
    stride, lam_std=0.01
):
    mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81
    bundle_idx = int(np.unique(X_sorted[:, -2])[0])
    sigma_indices = np.unique(X_sorted[:, -1].astype(int))

    records = []

    for sigma_idx in sigma_indices:
        mask = X_sorted[:, -1] == sigma_idx
        X_sigma = X_sorted[mask]
        y_sigma = y_sorted[mask]
        sort_idx = np.argsort(X_sigma[:, 0])
        X_sigma = X_sigma[sort_idx]
        y_sigma = y_sigma[sort_idx]

        for label in ["mean", "plus3", "minus3"]:
            # === Fix initial lambda once ===
            lam_initial = y_sigma[0]
            if label == "plus3":
                lam_used = lam_initial + 3 * lam_std
            elif label == "minus3":
                lam_used = lam_initial - 3 * lam_std
            else:
                lam_used = lam_initial.copy()

            print(f"SIGMA {sigma_idx} [{label.upper()}] initial λ (t = {X_sigma[0, 0]:.2f}):")
            print(np.array2string(lam_used, formatter={'float_kind': lambda x: f"{x:.6f}"}))

            for i in range(len(X_sigma) - 1):
                t0, t1 = X_sigma[i, 0], X_sigma[i + 1, 0]
                seg_times = np.linspace(t0, t1, 5)
                mee = X_sigma[i, 1:8]

                S = np.concatenate([mee, lam_used])
                Sf = scipy.integrate.solve_ivp(
                    lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                    [t0, t1], S, t_eval=seg_times
                )

                mee_matrix = Sf.y.T[:, :6]
                r_xyz, v_xyz = mee2rv.mee2rv(
                    mee_matrix[:, 0],  # p
                    mee_matrix[:, 1],  # f
                    mee_matrix[:, 2],  # g
                    mee_matrix[:, 3],  # h
                    mee_matrix[:, 4],  # k
                    mee_matrix[:, 5],  # L
                    mu
                )

                for k in range(len(seg_times)):
                    lam_k = Sf.y[7:, k]
                    records.append({
                        "time": seg_times[k],
                        "sigma_idx": sigma_idx,
                        "lam_type": label,
                        "x": r_xyz[k, 0], "y": r_xyz[k, 1], "z": r_xyz[k, 2],
                        "vx": v_xyz[k, 0], "vy": v_xyz[k, 1], "vz": v_xyz[k, 2],
                        **{f"sampled_lam_{j}": lam_k[j] for j in range(7)}
                    })

    return pd.DataFrame(records)