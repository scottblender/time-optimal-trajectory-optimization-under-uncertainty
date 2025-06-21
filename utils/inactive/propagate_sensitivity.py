import numpy as np
import pandas as pd
import scipy.integrate
import rv2mee
import mee2rv
import odefunc

def propagate_sensitivity_from_initial_lam_only(
    X_sorted, y_sorted, trajectories, backTspan,
    stride
):
    mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81
    bundle_idx = int(np.unique(X_sorted[:, -2])[0])
    sigma_indices = np.unique(X_sorted[:, -1].astype(int))

    # === Empirical std devs ===
    lam_std_vec = np.array([0.004881, 0.006194, 0.004179, 0.004418, 0.005339, 0.003779, 0.003987])

    # === Time check ===
    t_min = X_sorted[:, 0].min()
    t_max = X_sorted[:, 0].max()
    print(f"\n=== Propagation Time Span ===")
    print(f"Start Time: {t_min:.5f}")
    print(f"End Time:   {t_max:.5f}")
    print(f"Duration:   {t_max - t_min:.5f} (nondimensional)")

    records = []

    for sigma_idx in sigma_indices:
        mask = X_sorted[:, -1] == sigma_idx
        X_sigma = X_sorted[mask]
        y_sigma = y_sorted[mask]
        sort_idx = np.argsort(X_sigma[:, 0])
        X_sigma = X_sigma[sort_idx]
        y_sigma = y_sigma[sort_idx]

        for label in ["mean", "plus3", "minus3"]:
            lam0 = y_sigma[0]

            if label == "plus3":
                lam_used = lam0 + 3 * lam_std_vec
            elif label == "minus3":
                lam_used = lam0 - 3 * lam_std_vec
            else:
                lam_used = lam0.copy()

            # Initial state
            mee0 = X_sigma[0, 1:8]
            r0, v0 = mee2rv.mee2rv(
                np.array([mee0[0]]), np.array([mee0[1]]), np.array([mee0[2]]),
                np.array([mee0[3]]), np.array([mee0[4]]), np.array([mee0[5]]),
                mu
            )
            print(f"\nSIGMA {sigma_idx} [{label.upper()}] Initial Conditions at t = {X_sigma[0, 0]:.2f}")
            print("Position [x y z] (km):")
            print(np.array2string(r0[0], formatter={'float_kind': lambda x: f"{x:.6f}"}))
            print("Velocity [vx vy vz] (km/s):")
            print(np.array2string(v0[0], formatter={'float_kind': lambda x: f"{x:.6f}"}))
            print("Initial λ:")
            print(np.array2string(lam_used, formatter={'float_kind': lambda x: f"{x:.6f}"}))

            # Propagate each segment with same lam_used
            for i in range(len(X_sigma) - 1):
                t0, t1 = X_sigma[i, 0], X_sigma[i + 1, 0]
                seg_times = np.linspace(t0, t1, 5)

                state = X_sigma[i, 1:8]  # MEE + mass
                S = np.concatenate([state, lam_used])

                Sf = scipy.integrate.solve_ivp(
                    lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                    [t0, t1], S, t_eval=seg_times
                )

                mee_matrix = Sf.y.T[:, :6]
                r_xyz, v_xyz = mee2rv.mee2rv(
                    mee_matrix[:, 0], mee_matrix[:, 1], mee_matrix[:, 2],
                    mee_matrix[:, 3], mee_matrix[:, 4], mee_matrix[:, 5], mu
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

                # Update lam_used to most recent propagated value
                lam_used = Sf.y[7:, -1]  

    return pd.DataFrame(records)
