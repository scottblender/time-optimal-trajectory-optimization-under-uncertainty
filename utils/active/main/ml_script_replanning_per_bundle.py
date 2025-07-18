import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.linalg import eigh
from tqdm import tqdm
from itertools import product
from filterpy.kalman import MerweScaledSigmaPoints

# === Suppress warnings ===
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['LGBM_LOGLEVEL'] = '2'

# === Helpers ===
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
from odefunc import odefunc

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

def run_case(model, scaler, Wm, Wc, t_vals, r_nom, v_nom, mass_nom, mu, F_nom, c, m0, g0,
             t_frac, F_scale, cov_id, P_pos, P_vel, P_mass, save_dir, summary_rows):

    t_k = t_frac * t_vals[-1]
    idx = np.argmin(np.abs(t_vals - t_k))
    r0, v0, m0_val = r_nom[idx], v_nom[idx], mass_nom[idx]
    state_nom = np.hstack([r0, v0, m0_val])

    P = np.block([
        [P_pos, np.zeros((3, 3)), np.zeros((3, 1))],
        [np.zeros((3, 3)), P_vel, np.zeros((3, 1))],
        [np.zeros((1, 3)), np.zeros((1, 3)), P_mass]
    ])

    nsd = 7
    sp = MerweScaledSigmaPoints(n=nsd, alpha=0.3, beta=2.0, kappa=0)
    sigma_states = sp.sigma_points(state_nom, P)
    Wm_sp, Wc_sp = sp.Wm, sp.Wc
    mc_samples = np.random.multivariate_normal(state_nom, P, size=100)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(r_nom[:, 0], r_nom[:, 1], r_nom[:, 2], color='black', lw=2.0, label="Nominal")
    ax.scatter(r_nom[idx, 0], r_nom[idx, 1], r_nom[idx, 2], color='black', s=20, label="Start", zorder=5)
    ax.scatter(r_nom[-1, 0], r_nom[-1, 1], r_nom[-1, 2], color='black', s=20, marker='X', label="End", zorder=5)

    sigma_trajs, mc_trajs = [], []

    for s in sigma_states:
        r_s, v_s, m_s = s[:3], s[3:6], s[6]
        mee = np.hstack([rv2mee(r_s.reshape(1,3), v_s.reshape(1,3), mu).flatten(), m_s])
        x_input = np.hstack([t_vals[idx], mee, np.zeros(7)])
        x_df = pd.DataFrame([x_input], columns=[
            't', 'p', 'f', 'g', 'h', 'L', 'mass', 'dummy1',
            'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'
        ])
        x_scaled = scaler.transform(x_df)
        lam = model.predict(x_scaled)[0]
        S = np.hstack([mee, lam])
        sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_nom * F_scale, c, m0, g0),
                        [t_vals[idx], t_vals[-1]], S,
                        t_eval=np.linspace(t_vals[idx], t_vals[-1], 200))
        r_out, _ = mee2rv(*sol.y[:6], mu)
        sigma_trajs.append(r_out)
        ax.plot(*r_out.T, linestyle='--', color='gray', lw=0.8, zorder=2)
        ax.scatter(*r_out[0], color='gray', s=6, marker='o', zorder=2)
        ax.scatter(*r_out[-1], color='gray', s=6, marker='X', zorder=2)

    for s in mc_samples:
        r_s, v_s, m_s = s[:3], s[3:6], s[6]
        mee = np.hstack([rv2mee(r_s.reshape(1,3), v_s.reshape(1,3), mu).flatten(), m_s])
        x_input = np.hstack([t_vals[idx], mee, np.zeros(7)])
        x_df = pd.DataFrame([x_input], columns=[
            't', 'p', 'f', 'g', 'h', 'L', 'mass', 'dummy1',
            'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'
        ])
        x_scaled = scaler.transform(x_df)
        lam = model.predict(x_scaled)[0]
        S = np.hstack([mee, lam])
        sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_nom * F_scale, c, m0, g0),
                        [t_vals[idx], t_vals[-1]], S,
                        t_eval=np.linspace(t_vals[idx], t_vals[-1], 200))
        r_out, _ = mee2rv(*sol.y[:6], mu)
        mc_trajs.append(r_out)
        ax.plot(*r_out.T, linestyle=':', color='0.6', lw=0.8, alpha=0.25, zorder=1)
        ax.scatter(*r_out[0], color='0.6', s=5, marker='o', alpha=0.25, zorder=1)
        ax.scatter(*r_out[-1], color='0.6', s=5, marker='X', alpha=0.25, zorder=1)

    r_sigma_end = np.array([traj[-1] for traj in sigma_trajs])
    mu_sigma = np.sum(Wm[:, None] * r_sigma_end, axis=0)
    cov_sigma = np.einsum("i,ij,ik->jk", Wc, r_sigma_end - mu_sigma, r_sigma_end - mu_sigma)
    plot_3sigma_ellipsoid(ax, mu_sigma, cov_sigma, color='gray', alpha=0.2)

    r_mc_end = np.array([traj[-1] for traj in mc_trajs])
    mu_mc = np.mean(r_mc_end, axis=0)
    cov_mc = np.einsum("ni,nj->ij", r_mc_end - mu_mc, r_mc_end - mu_mc) / r_mc_end.shape[0]
    plot_3sigma_ellipsoid(ax, mu_mc, cov_mc, color='0.6', alpha=0.15)

    sigma_dists = np.linalg.norm(r_sigma_end - r_nom[-1], axis=1)
    mc_dists = np.linalg.norm(r_mc_end - r_nom[-1], axis=1)
    summary_rows.append({
        't_frac': t_frac,
        'F_scale': F_scale,
        'cov_id': cov_id,
        'mean_sigma_dev': sigma_dists.mean(),
        'std_sigma_dev': sigma_dists.std(),
        'mean_mc_dev': mc_dists.mean(),
        'std_mc_dev': mc_dists.std(),
        'delta_sigma_mc_mean': sigma_dists.mean() - mc_dists.mean()
    })

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color='black', lw=2.0, label='Nominal'),
        Line2D([0], [0], linestyle='--', color='gray', lw=1.0, label='Sigma Points'),
        Line2D([0], [0], linestyle=':', color='0.6', lw=1.0, label='Monte Carlo'),
        Line2D([0], [0], marker='o', color='black', linestyle='', label='Start', markersize=5),
        Line2D([0], [0], marker='X', color='black', linestyle='', label='End', markersize=6),
        Patch(color='gray', alpha=0.2, label='3σ Ellipsoid (σ)'),
        Patch(color='0.6', alpha=0.15, label='3σ Ellipsoid (MC)')
    ], loc='upper left', fontsize=8, frameon=True)

    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    set_axes_equal(ax)
    plt.tight_layout()

    filename = f"tk{int(t_frac*100)}_F{int(F_scale*100)}_cov{cov_id}.pdf"
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', pad_inches=0.5, dpi=600)
    plt.close()
    print(f"[Saved] {filename}")

def main():
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    Wm = joblib.load("Wm.pkl")
    Wc = joblib.load("Wc.pkl")
    data = joblib.load("stride_4000min/bundle_data_4000min.pkl")

    r_nom = data["r_tr"]
    v_nom = data["v_tr"]
    mass_nom = data["mass_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]

    tk_vals = [0.8, 0.85, 0.9]
    F_scales = [0.8, 0.85, 0.9]
    cov_setups = {
        1: (np.eye(3)*0.01, np.eye(3)*0.0001, np.array([[0.0001]])),
        2: (np.eye(3)*0.05, np.eye(3)*0.0005, np.array([[0.0005]])),
        3: (np.eye(3)*0.1,  np.eye(3)*0.001,  np.array([[0.001]]))
    }

    os.makedirs("uncertainty_aware_outputs", exist_ok=True)
    summary_rows = []

    for t_frac, F_scale, (cov_id, (P_pos, P_vel, P_mass)) in tqdm(
        product(tk_vals, F_scales, cov_setups.items()), total=27, desc="Evaluating configs"
    ):
        run_case(model, scaler, Wm, Wc, t_vals, r_nom, v_nom, mass_nom,
                 mu, F_nom, c, m0, g0,
                 t_frac, F_scale, cov_id,
                 P_pos, P_vel, P_mass,
                 save_dir="uncertainty_aware_outputs",
                 summary_rows=summary_rows)

    df = pd.DataFrame(summary_rows)
    df.to_csv("uncertainty_aware_outputs/final_position_deviation_summary.csv", index=False)
    print("[Saved] final_position_deviation_summary.csv")

if __name__ == "__main__":
    main()
