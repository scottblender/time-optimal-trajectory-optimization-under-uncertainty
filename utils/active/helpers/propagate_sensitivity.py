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
    # === Constants ===
    mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81
    bundle_idx = int(np.unique(X_sorted[:, -2])[0])  # assumes fixed bundle
    sigma_indices = np.unique(X_sorted[:, -1].astype(int))

    results = []

    for sigma_idx in sigma_indices:
        mask = X_sorted[:, -1] == sigma_idx
        X_sigma = X_sorted[mask]
        y_sigma = y_sorted[mask]
        sort_idx = np.argsort(X_sigma[:, 0])
        X_sigma = X_sigma[sort_idx]
        y_sigma = y_sigma[sort_idx]

        for label in ["mean", "plus3", "minus3"]:
            r_all, v_all = [], []

            for i in range(len(X_sigma) - 1):
                t0, t1 = X_sigma[i, 0], X_sigma[i + 1, 0]
                seg_times = np.linspace(t0, t1, 5)
                mee = X_sigma[i, 1:8]
                lam_original = y_sigma[i]

                # Apply ±3σ only at i == 0
                if i == 0:
                    if label == "plus3":
                        lam_used = lam_original + 3 * lam_std
                    elif label == "minus3":
                        lam_used = lam_original - 3 * lam_std
                    else:
                        lam_used = lam_original.copy()

                    print(f"SIGMA {sigma_idx} [{label.upper()}] initial λ (t = {t0:.2f}):")
                    print(np.array2string(lam_used, formatter={'float_kind': lambda x: f"{x:.6f}"}))
                else:
                    lam_used = lam_original  # exactly as in the original propagation

                S = np.concatenate([mee, lam_used])
                Sf = scipy.integrate.solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                                               [t0, t1], S, t_eval=seg_times)

                r_xyz, v_xyz = mee2rv.mee2rv(Sf.y.T[:, 0], Sf.y.T[:, 1], Sf.y.T[:, 2],
                                             Sf.y.T[:, 3], Sf.y.T[:, 4], Sf.y.T[:, 5], mu)
                r_all.append(r_xyz)
                v_all.append(v_xyz)

            r_all = np.vstack(r_all)
            v_all = np.vstack(v_all)
            r_actual = trajectories[0][0][sigma_idx][:, :3]
            v_actual = trajectories[0][0][sigma_idx][:, 3:6]

            r_pred_interp = r_all[:r_actual.shape[0]]
            v_pred_interp = v_all[:v_actual.shape[0]]

            print("Final Predicted Position:")
            print(np.array2string(r_pred_interp[-1], formatter={'float_kind': lambda x: f"{x:.6f}"}))
            print("Final Actual Position:")
            print(np.array2string(r_actual[-1], formatter={'float_kind': lambda x: f"{x:.6f}"}))

            mse_r = np.mean((r_actual - r_pred_interp) ** 2, axis=0)
            mse_v = np.mean((v_actual - v_pred_interp) ** 2, axis=0)
            final_dev = np.linalg.norm(r_actual[-1] - r_pred_interp[-1])

            results.append({
                "sigma": sigma_idx,
                "label": label,
                "stride": stride,
                "x mse": mse_r[0],
                "y mse": mse_r[1],
                "z mse": mse_r[2],
                "vx mse": mse_v[0],
                "vy mse": mse_v[1],
                "vz mse": mse_v[2],
                "final position deviation": final_dev
            })

    return pd.DataFrame(results)
