import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from tqdm import tqdm
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['LGBM_LOGLEVEL'] = '2'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import mee2rv
import rv2mee
import odefunc

# === Constants ===
mu, F, c, m0, g0 = 27.8996, 0.33, 4.4246, 4000, 9.81
P_pos = np.eye(3) * 0.01
P_vel = np.eye(3) * 0.0001
P_mass = np.array([[0.0001]])
P_init = np.block([
    [P_pos, np.zeros((3, 3)), np.zeros((3, 1))],
    [np.zeros((3, 3)), P_vel, np.zeros((3, 1))],
    [np.zeros((1, 3)), np.zeros((1, 3)), P_mass]
])

# === Helper functions ===
def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    ranges = [abs(lim[1] - lim[0]) for lim in [x_limits, y_limits, z_limits]]
    centers = [np.mean(lim) for lim in [x_limits, y_limits, z_limits]]
    max_range = max(ranges) / 2
    ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
    ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
    ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])
    ax.set_box_aspect([1.25, 1, 0.75])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.2, scale=3.0):
    cov = 0.5 * (cov + cov.T) + np.eye(3) * 1e-10
    vals, vecs = eigh(cov)
    if np.any(vals <= 0): return
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    radii = scale * np.sqrt(vals)
    u, v = np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 40)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    ellipsoid = np.stack((x, y, z), axis=-1) @ vecs.T + mean
    ax.plot_surface(ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2],
                    rstride=1, cstride=1, color=color, alpha=alpha, linewidth=0)
    ax.plot_wireframe(ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2],
                      rstride=5, cstride=5, color='k', alpha=0.2, linewidth=0.3)

def compute_kl_divergence(mu1, sigma1, mu2, sigma2):
    k = mu1.shape[0]
    sigma2_inv = inv(sigma2)
    trace_term = np.trace(sigma2_inv @ sigma1)
    diff = mu2 - mu1
    quadratic_term = diff.T @ sigma2_inv @ diff
    sign1, logdet1 = np.linalg.slogdet(sigma1)
    sign2, logdet2 = np.linalg.slogdet(sigma2)
    if sign1 <= 0 or sign2 <= 0:
        return 0.0
    log_det_term = logdet2 - logdet1
    kl_div = 0.5 * (trace_term + quadratic_term - k + log_det_term)
    return max(kl_div, 0.0)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def monte_carlo_segment(r0, v0, m0_val, lam, t0, t1, num_samples=500):
    mean_eci = np.hstack([r0.flatten(), v0.flatten(), m0_val])
    samples = np.random.multivariate_normal(mean_eci, P_init, size=num_samples)
    r_trajs = []
    for s in tqdm(samples, desc=f"[MC] Segment t={t0:.1f}→{t1:.1f}", leave=False):
        r_s, v_s = s[:3].reshape(1, 3), s[3:6].reshape(1, 3)
        m_s = s[6]
        state_mee = np.hstack([rv2mee.rv2mee(r_s, v_s, mu).flatten(), m_s])
        S = np.hstack([state_mee, lam])
        sol = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                        [t0, t1], S, t_eval=np.linspace(t0, t1, 20))
        r, _ = mee2rv.mee2rv(*sol.y[:6], mu)
        r_trajs.append(r)
    return np.array(r_trajs)

def plot_bar_dual_axis(df_label, label):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    df_label = df_label.copy()
    df_label["id"] = df_label.apply(lambda row: f"B{int(row['bundle_idx']):02d}_S{int(row['seg_idx']):02d}", axis=1)
    x = np.arange(len(df_label))
    ids = df_label["id"]

    os.makedirs("eval_lightgbm_outputs", exist_ok=True)

    # === Position MSE Plot ===
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()
    bar1 = ax1.bar(x, df_label["mse_x"] + df_label["mse_y"] + df_label["mse_z"],
                   width=0.5, color="tab:blue", label="Position MSE")
    line1, = ax2.plot(x, df_label["control_mse"], color="tab:orange", marker='o', label='Control MSE', linewidth=2.0)

    ax1.set_ylabel("Position MSE (log scale)")
    ax2.set_ylabel("Control MSE")
    ax1.set_xlabel("Bundle-Segment")
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ids, rotation=90, fontsize=7)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # Combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper left', fontsize=9)

    fig1.tight_layout()
    plt.savefig(f"eval_lightgbm_outputs/barplot_position_control_mse_{label}.pdf", dpi=600)
    plt.close()

    # === Velocity MSE Plot ===
    fig2, ax3 = plt.subplots(figsize=(14, 5))
    ax4 = ax3.twinx()
    bar2 = ax3.bar(x, df_label["mse_vx"] + df_label["mse_vy"] + df_label["mse_vz"],
                   width=0.5, color="tab:green", label="Velocity MSE")
    line2, = ax4.plot(x, df_label["control_mse"], color="tab:orange", marker='o', label='Control MSE', linewidth=2.0)

    ax3.set_ylabel("Velocity MSE (log scale)")
    ax4.set_ylabel("Control MSE")
    ax3.set_xlabel("Bundle-Segment")
    ax3.set_yscale("log")
    ax3.set_xticks(x)
    ax3.set_xticklabels(ids, rotation=90, fontsize=7)
    ax3.grid(True, linestyle='--', alpha=0.4)

    # Combine legends from both axes
    handles3, labels3 = ax3.get_legend_handles_labels()
    handles4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(handles3 + handles4, labels3 + labels4, loc='upper left', fontsize=9)

    fig2.tight_layout()
    plt.savefig(f"eval_lightgbm_outputs/barplot_velocity_control_mse_{label}.pdf", dpi=600)
    plt.close()

def evaluate_and_plot(X, y, model, scaler, Wm, Wc, label, bundle_idx):
    X_b = X[X[:, -2] == bundle_idx]
    y_b = y[X[:, -2] == bundle_idx]
    times = np.unique(X_b[:, 0])
    segments = list(zip(times[:-1], times[1:]))

    X_df = pd.DataFrame(X_b[:, :-2])
    X_scaled = scaler.transform(X_df)
    y_pred = model.predict(X_scaled)

    r_sigmas, control_profile = [], []
    sigma_indices = np.unique(X_b[:, -1]).astype(int)
    metrics, maha_rows = [], []

    for sigma_idx in sigma_indices:
        traj = []
        for i, (t0, t1) in enumerate(segments):
            row = X_b[(X_b[:, 0] == t0) & (X_b[:, -1] == sigma_idx)][1]
            lam = y_pred[(X_b[:, 0] == t0) & (X_b[:, -1] == sigma_idx)][1]
            if sigma_idx == 0:
                control_profile.append(lam)
            S = np.concatenate([row[1:8], lam])
            sol = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                            [t0, t1], S, t_eval=np.linspace(t0, t1, 20))
            r, _ = mee2rv.mee2rv(*sol.y[:6], mu)
            traj.append(r)
        r_sigmas.append(traj)

    r_sigmas = np.array(r_sigmas)
    out_dir = f"eval_lightgbm_outputs/{label}/bundle_{bundle_idx}"
    os.makedirs(out_dir, exist_ok=True)

    for seg_idx, (t0, t1) in enumerate(segments):
        sigma0_rows = X_b[(X_b[:, 0] == t0) & (X_b[:, -1] == 0) & np.all(np.isclose(X_b[:, 8:15], 0.0, atol=1e-12), axis=1)]
        if len(sigma0_rows) != 1:
            print(f"[WARN] Missing σ₀ row at t0={t0:.6f} for bundle {bundle_idx}")
            continue
        row_sigma0 = sigma0_rows[0]
        mee0 = row_sigma0[1:8]
        r0, v0 = mee2rv.mee2rv(*[np.array([val]) for val in mee0[:6]], mu)
        m0_val = mee0[6]
        lam = control_profile[seg_idx]

        mc_traj = monte_carlo_segment(r0, v0, m0_val, lam, t0, t1, num_samples=500)
        mc_end = np.array([traj[-1] for traj in mc_traj])
        mu_mc = np.mean(mc_end, axis=0)
        cov_mc = np.cov(mc_end.T)

        r_sigma_start = np.array([traj[seg_idx][0] for traj in r_sigmas])
        r_sigma_end = np.array([traj[seg_idx][-1] for traj in r_sigmas])
        mu_start = np.sum(Wm[:, None] * r_sigma_start, axis=0)
        mu_end = np.sum(Wm[:, None] * r_sigma_end, axis=0)
        cov_start = np.einsum("i,ij,ik->jk", Wc, r_sigma_start - mu_start, r_sigma_start - mu_start)
        cov_end = np.einsum("i,ij,ik->jk", Wc, r_sigma_end - mu_end, r_sigma_end - mu_end)

        # Propagate actual trajectories to t1
        r_actual, v_actual, m_actual = [], [], []
        for sigma_idx in sigma_indices:
            rows = X_b[(X_b[:, 0] == t0) & (X_b[:, -1] == sigma_idx)]
            if len(rows) == 1:
                row = rows[0]
            elif len(rows) > 1:
                row = rows[1]
            else:
                continue

            y_actual = y_b[(X_b[:, 0] == t0) & (X_b[:, -1] == sigma_idx)]
            if len(y_actual) == 1:
                lam_actual = y_actual[0]
            elif len(y_actual) > 1:
                lam_actual = y_actual[1]
            else:
                continue

            S = np.concatenate([row[1:8], lam_actual])
            t_eval = np.linspace(t0, t1, 20)
            sol = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                            [t0, t1], S, t_eval=t_eval)
            r_i, v_i = mee2rv.mee2rv(*sol.y[:6], mu)
            r_actual.append(r_i[-1])
            v_actual.append(v_i[-1])
            m_actual.append(sol.y[6, -1])

        mu_act = np.sum(Wm[:, None] * r_actual, axis=0)
        cov_act = np.einsum("i,ij,ik->jk", Wc, r_actual - mu_act, r_actual - mu_act)

        sigma0_rows_t1 = X_b[(X_b[:, 0] == t1) & (X_b[:, -1] == 0) & np.all(np.isclose(X_b[:, 8:15], 0.0, atol=1e-12), axis=1)]
        if len(sigma0_rows_t1) != 1:
            print(f"[WARN] Missing σ₀ row at t1={t1:.6f} for bundle {bundle_idx}")
            continue
        mee_ref = sigma0_rows_t1[0][1:8]
        _, ref_v = mee2rv.mee2rv(*[np.array([val]) for val in mee_ref[:6]], mu)
        ref_r = r0.flatten()
        ref_v = ref_v.flatten()
        ref_m = mee_ref[6]

        r_pred, v_pred, m_pred = [], [], []
        for sigma_idx in sigma_indices:
            row = X_b[(X_b[:, 0] == t0) & (X_b[:, -1] == sigma_idx)][1]
            lam_i = y_pred[(X_b[:, 0] == t0) & (X_b[:, -1] == sigma_idx)][1]
            S = np.concatenate([row[1:8], lam_i])
            t_eval = np.linspace(t0, t1, 20)
            sol = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                            [t0, t1], S, t_eval=t_eval)
            r_i, v_i = mee2rv.mee2rv(*sol.y[:6], mu)
            r_pred.append(r_i[-1])
            v_pred.append(v_i[-1])
            m_pred.append(sol.y[6, -1])

        r_pred = np.array(r_pred)
        r_actual = np.array(r_actual)
        v_pred = np.array(v_pred)
        v_actual = np.array(v_actual)
        m_pred = np.array(m_pred)
        m_actual = np.array(m_actual)

        mse_xyz = np.mean((r_pred - ref_r) ** 2, axis=0)
        mse_vxyz = np.mean((v_pred - ref_v) ** 2, axis=0)
        mse_mass = np.mean((m_pred - ref_m) ** 2)

        mse_xyz_act = np.mean((r_actual - ref_r) ** 2, axis=0)
        mse_vxyz_act = np.mean((v_actual - ref_v) ** 2, axis=0)
        mse_mass_act = np.mean((m_actual - ref_m) ** 2)

        maha_vals = [mahalanobis(r, mu_mc, inv(cov_mc + 1e-10*np.eye(3))) for r in r_pred]
        maha_act = [mahalanobis(r, mu_mc, inv(cov_mc + 1e-10*np.eye(3))) for r in r_actual]

        for i, m in enumerate(maha_vals):
            maha_rows.append({
                "label": label, "bundle_idx": bundle_idx, "seg_idx": seg_idx,
                "sigma_idx": i, "mahalanobis": m,
                "cosine_similarity": cosine_similarity(
                    lam, y_b[(X_b[:, 0] == t0) & (X_b[:, -1] == i)][1]),
                "type": "predicted"
            })

        for i, m in enumerate(maha_act):
            rows = y_b[(X_b[:, 0] == t1) & (X_b[:, -1] == i)]
            if len(rows) == 1:
                y_actual = rows[0]
            elif len(rows) > 1:
                y_actual = rows[1]
            else:
                continue
            maha_rows.append({
                "label": label, "bundle_idx": bundle_idx, "seg_idx": seg_idx,
                "sigma_idx": i, "mahalanobis": m,
                "cosine_similarity": cosine_similarity(
                    y_actual,
                    y_b[(X_b[:, 0] == t0) & (X_b[:, -1] == 0)][1]
                ),
                "type": "actual"
            })

        metrics.append({
            "label": label, "bundle_idx": bundle_idx, "seg_idx": seg_idx,
            "mse_x": mse_xyz[0], "mse_y": mse_xyz[1], "mse_z": mse_xyz[2],
            "mse_vx": mse_vxyz[0], "mse_vy": mse_vxyz[1], "mse_vz": mse_vxyz[2],
            "mse_mass": mse_mass,
            "kl": compute_kl_divergence(mu_end, cov_end, mu_mc, cov_mc),
            "cos_sim": cosine_similarity(lam, y_b[(X_b[:, 0] == t0) & (X_b[:, -1] == 0)][1]),
            "control_mse": np.mean((lam - y_b[(X_b[:, 0] == t0) & (X_b[:, -1] == 0)][1]) ** 2),
            "kl_pred_vs_actual": compute_kl_divergence(mu_end, cov_end, mu_act, cov_act),
            "mse_dx": np.mean((r_pred - r_actual) ** 2, axis=0)[0],
            "mse_dy": np.mean((r_pred - r_actual) ** 2, axis=0)[1],
            "mse_dz": np.mean((r_pred - r_actual) ** 2, axis=0)[2],
            "mse_dvx": np.mean((v_pred - v_actual) ** 2, axis=0)[0],
            "mse_dvy": np.mean((v_pred - v_actual) ** 2, axis=0)[1],
            "mse_dvz": np.mean((v_pred - v_actual) ** 2, axis=0)[2],
            "mse_dmass": np.mean((np.array(m_pred) - m_actual) ** 2),
            "actual_mse_x": mse_xyz_act[0], "actual_mse_y": mse_xyz_act[1], "actual_mse_z": mse_xyz_act[2],
            "actual_mse_vx": mse_vxyz_act[0], "actual_mse_vy": mse_vxyz_act[1], "actual_mse_vz": mse_vxyz_act[2],
            "actual_mse_mass": mse_mass_act
        })

    return {
        "summary": pd.DataFrame(metrics),
        "per_sigma": pd.DataFrame(maha_rows)
    }

def finalize_metrics(df_all):
    total_metrics = {
        "Total Position MSE": np.mean(df_all["mse_x"] + df_all["mse_y"] + df_all["mse_z"]),
        "Total Velocity MSE": np.mean(df_all["mse_vx"] + df_all["mse_vy"] + df_all["mse_vz"]),
        "Total Mass MSE": np.mean(df_all["mse_mass"]),
        "Total Control MSE": np.mean(df_all["control_mse"]),
        "Mean KL Divergence": np.mean(df_all["kl"]),
        "Mean KL Pred vs Actual": np.mean(df_all["kl_pred_vs_actual"]),
    }
    pd.DataFrame([total_metrics]).to_csv("eval_lightgbm_outputs/total_mse_summary.csv", index=False)

def main():
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    Wm = joblib.load("Wm.pkl")
    Wc = joblib.load("Wc.pkl")
    data_max = joblib.load("segment_max.pkl")
    data_min = joblib.load("segment_min.pkl")

    summary_list, sigma_list = [], []

    for label, data in [("max", data_max), ("min", data_min)]:
        for bundle_idx in tqdm(np.unique(data["X"][:, -2]).astype(int), desc=f"Evaluating {label}"):
            result = evaluate_and_plot(data["X"], data["y"], model, scaler, Wm, Wc, label, bundle_idx)
            df_summary = result["summary"]
            df_summary["label"] = label
            df_summary["bundle_idx"] = bundle_idx
            summary_list.append(df_summary)
            df_sigma = result["per_sigma"]
            sigma_list.append(df_sigma)

    df_all = pd.concat(summary_list, ignore_index=True)
    df_sigma_all = pd.concat(sigma_list, ignore_index=True)

    os.makedirs("eval_lightgbm_outputs", exist_ok=True)
    df_all.to_csv("eval_lightgbm_outputs/summary_metrics_all.csv", index=False)
    df_sigma_all.to_csv("eval_lightgbm_outputs/mahal_cos_per_sigma.csv", index=False)

    for label in ["min", "max"]:
        df_label = df_all[df_all["label"] == label].copy()
        df_label.to_csv(f"eval_lightgbm_outputs/summary_metrics_{label}.csv", index=False)
        plot_bar_dual_axis(df_label, label)

    finalize_metrics(df_all)
    print("[DONE] All outputs saved.")

if __name__ == "__main__":
    main()
