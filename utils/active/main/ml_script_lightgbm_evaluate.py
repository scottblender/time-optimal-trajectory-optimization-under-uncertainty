# ml_script_lightgbm_evaluate.py

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.integrate import solve_ivp
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv, eigh
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
plt.rcParams.update({'font.size': 10})

# === Setup ===
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['LGBM_LOGLEVEL'] = '2'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
from odefunc import odefunc

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

# === Plotting Utilities ===
def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.2, scale=3.0):
    cov = 0.5 * (cov + cov.T) + np.eye(3) * 1e-10
    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.any(eigvals <= 0): return
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    radii = scale * np.sqrt(eigvals)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    ellipsoid = np.stack((x, y, z), axis=-1) @ eigvecs.T + mean
    ax.plot_surface(ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
                    rstride=1, cstride=1, color=color, alpha=alpha, linewidth=0)
    ax.plot_wireframe(ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
                      rstride=5, cstride=5, color='k', alpha=0.2, linewidth=0.3)

def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    ranges = [abs(lim[1] - lim[0]) for lim in [x_limits, y_limits, z_limits]]
    centers = [np.mean(lim) for lim in [x_limits, y_limits, z_limits]]
    max_range = max(ranges) / 2
    ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
    ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
    ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])
    ax.set_box_aspect([1.25, 1, 0.75])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

def set_max_ticks(fig, n=5):
    from matplotlib.ticker import MaxNLocator
    for ax in fig.get_axes():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n))
        if hasattr(ax, 'zaxis'):
            ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

# === Metric Utilities ===
def compute_kl_divergence(mu1, sigma1, mu2, sigma2):
    k = mu1.shape[0]
    sigma2_inv = inv(sigma2)
    trace_term = np.trace(sigma2_inv @ sigma1)
    diff = mu2 - mu1
    quad_term = diff.T @ sigma2_inv @ diff
    sign1, logdet1 = np.linalg.slogdet(sigma1)
    sign2, logdet2 = np.linalg.slogdet(sigma2)
    if sign1 <= 0 or sign2 <= 0:
        return 0.0
    return 0.5 * (trace_term + quad_term - k + logdet2 - logdet1)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# === MC Segment (Single, Fast) ===
def monte_carlo_segment(r0, v0, m0_val, lam, t0, t1, num_samples=500):
    mean = np.hstack([r0.flatten(), v0.flatten(), m0_val])
    samples = np.random.multivariate_normal(mean, P_init, size=num_samples)
    r_trajs = []
    for s in tqdm(samples, desc=f"[MC] {t0:.2f} → {t1:.2f}", leave=False):
        r_s, v_s, m_s = s[:3], s[3:6], s[6]
        mee_state = np.hstack([rv2mee(r_s.reshape(1, 3), v_s.reshape(1, 3), mu).flatten(), m_s])
        state = np.hstack([mee_state, lam])
        sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F, c, m0, g0),
                        [t0, t1], state, t_eval=np.linspace(t0, t1, 20))
        r, _ = mee2rv(*sol.y[:6], mu)
        r_trajs.append(r)
    return np.array(r_trajs)

def evaluate_and_plot_segment(X, y, model, scaler, Wm, Wc, label, bundle_idx):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from scipy.integrate import solve_ivp
    from scipy.spatial.distance import mahalanobis
    from numpy.linalg import inv
    import os

    X_b = X[X[:, -2] == bundle_idx]
    y_b = y[X[:, -2] == bundle_idx]
    times = np.unique(X_b[:, 0])
    if len(times) < 2:
        return None
    t0, t1 = times[0], times[1]

    sigma_indices = np.unique(X_b[:, -1]).astype(int)
    r_pred, v_pred, m_pred = [], [], []
    r_pred_start = []

    r_actual, v_actual, m_actual = [], [], []
    r_actual_traj = []

    def get_row(t, sigma_idx, is_y=False):
        mask = np.isclose(X_b[:, 0], t, atol=1e-12) & (X_b[:, -1] == sigma_idx)
        rows = y_b[mask] if is_y else X_b[mask]
        return rows[1] if len(rows) > 1 else rows[0]

    for sigma_idx in sigma_indices:
        x_row = get_row(t0, sigma_idx)
        x_df = pd.DataFrame([x_row[:-2]])
        lam_pred = model.predict(scaler.transform(x_df))[0]
        S_pred = np.concatenate([x_row[1:8], lam_pred])
        sol_pred = solve_ivp(lambda t, x: odefunc(t, x, mu, F, c, m0, g0),
                             [t0, t1], S_pred, t_eval=np.linspace(t0, t1, 20))
        r_p, v_p = mee2rv(*sol_pred.y[:6], mu)
        r_pred_start.append(r_p[0])
        r_pred.append(r_p[-1])
        v_pred.append(v_p[-1])
        m_pred.append(sol_pred.y[6, -1])

        lam_actual = get_row(t0, sigma_idx, is_y=True)
        S_actual = np.concatenate([x_row[1:8], lam_actual])
        sol_actual = solve_ivp(lambda t, x: odefunc(t, x, mu, F, c, m0, g0),
                               [t0, t1], S_actual, t_eval=np.linspace(t0, t1, 20))
        r_a, v_a = mee2rv(*sol_actual.y[:6], mu)
        r_actual.append(r_a[-1])
        v_actual.append(v_p[-1])
        m_actual.append(sol_actual.y[6, -1])
        r_actual_traj.append(r_a)

    r_pred = np.array(r_pred)
    v_pred = np.array(v_pred)
    m_pred = np.array(m_pred)
    r_pred_start = np.array(r_pred_start)
    r_actual = np.array(r_actual)
    v_actual = np.array(v_actual)
    m_actual = np.array(m_actual)

    ref_mask = (np.isclose(X_b[:, 0], t1, atol=1e-12)) & (X_b[:, -1] == 0) & np.all(np.abs(X_b[:, 8:15]) < 1e-8, axis=1)
    if not np.any(ref_mask):
        print(f"[WARN] No sigma₀ row found at t1 for bundle {bundle_idx}")
        return None
    row_sigma0 = X_b[ref_mask][0]
    mee_ref = row_sigma0[1:8]
    r_ref, v_ref = mee2rv(*[np.array([val]) for val in mee_ref[:6]], mu)
    m_ref = mee_ref[6]
    r_ref = r_ref.flatten()
    v_ref = v_ref.flatten()

    row_sigma0_start = get_row(t0, 0)
    mee0 = row_sigma0_start[1:8]
    r0, v0 = mee2rv(*[np.array([val]) for val in mee0[:6]], mu)

    lam0 = get_row(t0, 0, is_y=True)
    mc_traj = monte_carlo_segment(r0, v0, m_ref, lam0, t0, t1, num_samples=500)
    r_mc_end = np.array([traj[-1] for traj in mc_traj])
    mu_mc = np.mean(r_mc_end, axis=0)
    cov_mc = np.einsum("ni,nj->ij", r_mc_end - mu_mc, r_mc_end - mu_mc) / r_mc_end.shape[0]

    mu_pred = np.sum(Wm[:, None] * r_pred, axis=0)
    cov_pred = np.einsum("i,ij,ik->jk", Wc, r_pred - mu_pred, r_pred - mu_pred)
    mu_pred_start = np.sum(Wm[:, None] * r_pred_start, axis=0)
    cov_pred_start = np.einsum("i,ij,ik->jk", Wc, r_pred_start - mu_pred_start, r_pred_start - mu_pred_start)

    mu_act = np.sum(Wm[:, None] * r_actual, axis=0)
    cov_act = np.einsum("i,ij,ik->jk", Wc, r_actual - mu_act, r_actual - mu_act)

    maha_pred = [mahalanobis(r, mu_pred, inv(cov_pred + 1e-10*np.eye(3))) for r in r_pred]
    maha_act = [mahalanobis(r, mu_act, inv(cov_act + 1e-10*np.eye(3))) for r in r_actual]

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    # === Predicted sigma point trajectories ===
    for i in range(len(r_pred)):
        ax.plot([r_pred_start[i, 0], r_pred[i, 0]],
                [r_pred_start[i, 1], r_pred[i, 1]],
                [r_pred_start[i, 2], r_pred[i, 2]],
                color='gray' if i > 0 else 'black',
                linestyle='--' if i > 0 else '-',
                lw=0.8 if i > 0 else 2.2,
                alpha=1.0,
                zorder=5 if i == 0 else 4)
        ax.scatter(r_pred_start[i, 0], r_pred_start[i, 1], r_pred_start[i, 2],
                color='black', marker='o', s=8, zorder=1)
        ax.scatter(r_pred[i, 0], r_pred[i, 1], r_pred[i, 2],
                color='black', marker='X', s=8, zorder=1)

    # === Monte Carlo trajectories (subsampled) ===
    for j in range(0, len(mc_traj), 5):
        traj = mc_traj[j]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                linestyle=':', color='dimgray', lw=0.8, alpha=0.4, zorder=3)
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
                color='0.4', s=8, marker='o', alpha=0.3, zorder=1)
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
                color='0.4', s=8, marker='X', alpha=0.3, zorder=1)

    # === Ellipsoids ===
    plot_3sigma_ellipsoid(ax, mu_pred_start, cov_pred_start, color='0.5', alpha=0.15)
    plot_3sigma_ellipsoid(ax, mu_pred, cov_pred, color='0.5', alpha=0.25)

    # === Labels and legend ===
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    set_axes_equal(ax)
    set_max_ticks(fig)

    ax.legend(handles=[
        Line2D([0], [0], color='black', lw=2.2, label='Sub-nominal Mean State'),
        Line2D([0], [0], color='gray', lw=0.8, linestyle='--', label='Sigma Points'),
        Line2D([0], [0], color='0.4', lw=0.6, linestyle=':', label='Monte Carlo'),
        Line2D([0], [0], marker='o', color='black', linestyle='', label='Start', markersize=5),
        Line2D([0], [0], marker='X', color='black', linestyle='', label='End', markersize=6),
        Patch(facecolor='0.5', edgecolor='0.5', alpha=0.2, label='3-σ Ellipsoid')
    ], loc='upper left', bbox_to_anchor=(0.03, 1.07), frameon=True)


    os.makedirs(f"eval_lightgbm_outputs/{label}/bundle_{bundle_idx}", exist_ok=True)
    plt.savefig(f"eval_lightgbm_outputs/{label}/bundle_{bundle_idx}/sigma_mc_comparison.pdf",
                dpi=600, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    return {
        "label": label,
        "bundle_idx": bundle_idx,
        "seg_idx": 0,
        "pos_mse_pred": np.mean(np.sum((r_pred - r_ref) ** 2, axis=1)),
        "pos_mse_actual": np.mean(np.sum((r_actual - r_ref) ** 2, axis=1)),
        "pos_mse_diff": abs(np.mean(np.sum((r_pred - r_ref) ** 2, axis=1)) -
                            np.mean(np.sum((r_actual - r_ref) ** 2, axis=1))),
        "vel_mse_pred": np.mean(np.sum((v_pred - v_ref) ** 2, axis=1)),
        "vel_mse_actual": np.mean(np.sum((v_actual - v_ref) ** 2, axis=1)),
        "vel_mse_diff": abs(np.mean(np.sum((v_pred - v_ref) ** 2, axis=1)) -
                            np.mean(np.sum((v_actual - v_ref) ** 2, axis=1))),
        "mass_mse_pred": np.mean((m_pred - m_ref) ** 2),
        "mass_mse_actual": np.mean((m_actual - m_ref) ** 2),
        "mass_mse_diff": abs(np.mean((m_pred - m_ref) ** 2) - np.mean((m_actual - m_ref) ** 2)),
        "kl_pred_vs_mc": compute_kl_divergence(mu_pred, cov_pred, mu_mc, cov_mc),
        "kl_pred_vs_actual": compute_kl_divergence(mu_pred, cov_pred, mu_act, cov_act),
        "mahal_pred": np.mean(maha_pred),
        "mahal_actual": np.mean(maha_act),
        "mahal_diff": abs(np.mean(maha_pred) - np.mean(maha_act)),
    }

# === Main ===
def main():
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    Wm = joblib.load("Wm.pkl")
    Wc = joblib.load("Wc.pkl")
    data_max = joblib.load("segment_max.pkl")
    data_min = joblib.load("segment_min.pkl")

    results = []
    for label, data in [("max", data_max), ("min", data_min)]:
        for bundle_idx in tqdm(np.unique(data["X"][:, -2]).astype(int), desc=f"[{label}]"):
            res = evaluate_and_plot_segment(data["X"], data["y"], model, scaler, Wm, Wc, label, bundle_idx)
            if res: results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("eval_lightgbm_outputs/metrics_summary.csv", index=False)
    print(df)

    # === Plot bar plots ===
    for label in df["label"].unique():
        subset = df[df["label"] == label]
        x = np.arange(len(subset))
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax2 = ax1.twinx()
        ax1.bar(x - 0.2, subset["pos_mse_pred"], width=0.2, label="Pred Pos MSE")
        ax1.bar(x, subset["pos_mse_actual"], width=0.2, label="Actual Pos MSE")
        ax1.bar(x + 0.2, subset["pos_mse_diff"], width=0.2, label="|Diff|")
        ax1.set_yscale("log")
        ax1.set_ylabel("Position MSE")
        ax2.plot(x, subset["kl_pred_vs_mc"], color="black", marker="o", label="KL (Pred‖MC)")
        ax2.set_ylabel("KL Divergence")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"B{int(b)}S{int(s)}" for b, s in zip(subset.bundle_idx, subset.seg_idx)], rotation=90)
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"eval_lightgbm_outputs/barplot_position_{label}.pdf", dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
