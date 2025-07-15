# ml_script_lightgbm_evaluate.py

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

mu, F, c, m0, g0 = 27.8996, 0.33, 4.4246, 4000, 9.81
P_pos = np.eye(3) * 0.01
P_vel = np.eye(3) * 0.0001
P_mass = np.array([[0.0001]])
P_init = np.block([
    [P_pos, np.zeros((3, 3)), np.zeros((3, 1))],
    [np.zeros((3, 3)), P_vel, np.zeros((3, 1))],
    [np.zeros((1, 3)), np.zeros((1, 3)), P_mass]
])

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

def monte_carlo_segment(r0, v0, m0_val, lam, t0, t1, num_samples=500):
    from tqdm import tqdm
    mean_eci = np.hstack([r0.flatten(), v0.flatten(), m0_val])
    samples = np.random.multivariate_normal(mean_eci, P_init, size=num_samples)
    r_trajs = []

    for s in tqdm(samples, desc=f"[MC] Segment t={t0:.1f}→{t1:.1f}"):
        r_s, v_s = s[:3].reshape(1, 3), s[3:6].reshape(1, 3)
        m_s = s[6]
        state_mee = np.hstack([rv2mee.rv2mee(r_s, v_s, mu).flatten(), m_s])
        S = np.hstack([state_mee, lam])
        sol = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                        [t0, t1], S, t_eval=np.linspace(t0, t1, 20))
        r, _ = mee2rv.mee2rv(*sol.y[:6], mu)
        r_trajs.append(r)
    return np.array(r_trajs)

def evaluate_and_plot(X, y, model, scaler, Wm, Wc, label, bundle_idx):
    X_b = X[X[:, -2] == bundle_idx]
    y_b = y[X[:, -2] == bundle_idx]
    times = np.unique(X_b[:, 0])
    segments = list(zip(times[:-1], times[1:]))

    X_df = pd.DataFrame(X_b[:, :-2])
    X_scaled = scaler.transform(X_df)
    y_pred = model.predict(X_scaled)

    r_sigmas = []
    control_profile = []
    sigma_indices = np.unique(X_b[:, -1]).astype(int)

    for sigma_idx in sigma_indices:
        traj = []
        for i, (t0, t1) in enumerate(segments):
            row = X_b[(X_b[:, 0] == t0) & (X_b[:, -1] == sigma_idx)][1]
            lam = y_pred[(X_b[:, 0] == t0) & (X_b[:, -1] == sigma_idx)][0]
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
        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_subplot(111, projection="3d")

        seg_nominal = r_sigmas[0][seg_idx]
        ax.plot(*seg_nominal.T, color='black', lw=2.0, label='Nominal (σ₀)')
        ax.scatter(*seg_nominal[0], color='black', s=12, marker='o')
        ax.scatter(*seg_nominal[-1], color='black', s=20, marker='X')

        for k in range(1, r_sigmas.shape[0]):
            seg = r_sigmas[k][seg_idx]
            ax.plot(*seg.T, linestyle='--', color='gray', linewidth=0.8)
            ax.scatter(*seg[0], color='gray', s=6, marker='o')
            ax.scatter(*seg[-1], color='gray', s=6, marker='X')

        # Draw 3-sigma ellipsoid at start of segment
        r_sigma_start = np.array([traj[seg_idx][0] for traj in r_sigmas])
        r_sigma_end= np.array([traj[seg_idx][-1] for traj in r_sigmas])
        mean_r_start = np.sum(Wm[:, None] * r_sigma_start, axis=0)
        cov_start = np.einsum("i,ij,ik->jk", Wc, r_sigma_start - mean_r_start, r_sigma_start - mean_r_start)
        mean_r_end = np.sum(Wm[:, None] * r_sigma_end, axis=0)
        cov_end = np.einsum("i,ij,ik->jk", Wc, r_sigma_end - mean_r_end, r_sigma_end - mean_r_end)

        plot_3sigma_ellipsoid(ax, mean_r_start, cov_start, color='black', alpha=0.2)
        plot_3sigma_ellipsoid(ax, mean_r_end, cov_end, color='black', alpha=0.2)

        # Monte Carlo from fresh sigma₀ row
        row_sigma0 = X_b[(X_b[:, 0] == t0) & (X_b[:, -1] == 0)][1]
        mee0 = row_sigma0[1:8]
        r0, v0 = mee2rv.mee2rv(*[np.array([val]) for val in mee0[:6]], mu)
        m0_val = mee0[6]
        lam = control_profile[seg_idx]

        mc_traj = monte_carlo_segment(r0, v0, m0_val, lam, t0, t1, num_samples=500)
        for mc in mc_traj[::10]:
            ax.plot(*mc.T, linestyle=':', color='dimgray', lw=0.5, alpha=0.4)
            ax.scatter(*mc[0], color='dimgray', s=6, marker='o', alpha=0.3)
            ax.scatter(*mc[-1], color='dimgray', s=6, marker='X', alpha=0.3)

        ax.set_xlabel("X [km]")
        ax.set_ylabel("Y [km]")
        ax.set_zlabel("Z [km]")
        set_axes_equal(ax)
        ax.legend(handles=[
            Line2D([0], [0], color='black', lw=2.0, label='Nominal (σ₀)'),
            Line2D([0], [0], color='gray', linestyle='--', label='Sigma Points'),
            Line2D([0], [0], color='dimgray', linestyle=':', label='Monte Carlo'),
            Line2D([0], [0], marker='o', linestyle='', color='gray', label='Start'),
            Line2D([0], [0], marker='X', linestyle='', color='gray', label='End'),
            Patch(color='black', alpha=0.2, label='3σ Ellipsoid')
        ], fontsize=8, loc='upper left', bbox_to_anchor=(0.02, 0.98))
        plt.tight_layout()
        plt.savefig(f"{out_dir}/segment_{seg_idx:03d}.pdf", dpi=600)
        plt.close()

    return {"label": label, "bundle_idx": bundle_idx}

def main():
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    Wm = joblib.load("Wm.pkl")
    Wc = joblib.load("Wc.pkl")
    data_max = joblib.load("segment_max.pkl")
    data_min = joblib.load("segment_min.pkl")

    all_results = []
    for label, data in [("max", data_max), ("min", data_min)]:
        for bundle_idx in tqdm(np.unique(data["X"][:, -2]).astype(int), desc=f"Evaluating {label}"):
            result = evaluate_and_plot(data["X"], data["y"], model, scaler, Wm, Wc, label, bundle_idx)
            all_results.append(result)

    df = pd.DataFrame(all_results)
    os.makedirs("eval_lightgbm_outputs", exist_ok=True)
    df.to_csv("eval_lightgbm_outputs/summary_metrics.csv", index=False)
    print("[DONE] Saved summary_metrics.csv")

if __name__ == "__main__":
    main()
