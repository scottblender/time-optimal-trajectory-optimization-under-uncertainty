import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import eigh
from scipy.integrate import solve_ivp
plt.rcParams.update({'font.size': 10})

# === Setup ===
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({'font.size': 10})
os.environ['LGBM_LOGLEVEL'] = '2'

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
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.25, scale=3.0):
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

def main():
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    data = joblib.load("stride_4000min/bundle_data_4000min.pkl")
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    r_nom, v_nom, mass_nom = data["r_tr"], data["v_tr"], data["mass_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    tf = t_vals[-1]
    os.makedirs("uncertainty_aware_outputs", exist_ok=True)

    t_fracs = [0.90, 0.925, 0.95]
    diag_mins = np.array([9.098224e+01, 7.082445e-04, 6.788599e-04, 1.376023e-08, 2.346605e-08, 5.885859e-08, 1.000000e-04])
    diag_maxs = np.array([1.977489e+03, 6.013501e-03, 5.173225e-03, 4.284912e-04, 1.023625e-03, 6.818500e+00, 1.000000e-04])
    P_models = {'min': np.diag(diag_mins), 'max': np.diag(diag_maxs)}

    P_cart = np.block([
        [np.eye(3)*0.01,       np.zeros((3,3)), np.zeros((3,1))],
        [np.zeros((3,3)), np.eye(3)*0.0001,     np.zeros((3,1))],
        [np.zeros((1,3)), np.zeros((1,3)),      np.array([[0.0001]])]
    ])

    F_val = 0.9 * F_nom
    summary = []

    for t_frac in t_fracs:
        t_k = t_frac * tf
        idx = np.argmin(np.abs(t_vals - t_k))
        r0, v0, m0_val = r_nom[idx], v_nom[idx], mass_nom[idx]
        state_k = np.hstack([r0, v0, m0_val])

        # === Draw shared MC samples at this t_k ===
        shared_samples = np.random.multivariate_normal(state_k, P_cart, size=500)

        for level, P_model in P_models.items():
            print(f"[{t_frac:.2f}] Level: {level} — propagating shared MC set...")

            mc_endpoints = []
            fig = plt.figure(figsize=(6.5, 5.5))
            ax = fig.add_subplot(111, projection='3d')

            # Plot nominal trajectory
            ax.plot(r_nom[:, 0], r_nom[:, 1], r_nom[:, 2], color='black', lw=2.0, label="Nominal")
            ax.scatter(*r0, color='black', s=20, label="Start", zorder=5)
            ax.scatter(*r_nom[-1], color='black', s=20, marker='X', label="End", zorder=5)

            for s in shared_samples:
                try:
                    mee = np.hstack([rv2mee(s[:3].reshape(1, 3), s[3:6].reshape(1, 3), mu).flatten(), s[6]])
                    x_input = np.hstack([t_k, mee, np.diag(P_model)])
                    x_df = pd.DataFrame([x_input], columns=[
                        't', 'p', 'f', 'g', 'h', 'L', 'mass',
                        'dummy1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'
                    ])
                    lam = model.predict(scaler.transform(x_df))[0]
                    S = np.hstack([mee, lam])
                    sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0),
                                    [t_k, tf], S, t_eval=np.linspace(t_k, tf, 100))
                    r, _ = mee2rv(*sol.y[:6], mu)
                    mc_endpoints.append(r[-1])
                    ax.plot(r[:, 0], r[:, 1], r[:, 2],
                            linestyle=':', color='0.6', lw=0.8, alpha=0.3)
                    ax.scatter(*r[0], color='0.6', s=5, alpha=0.25)
                    ax.scatter(*r[-1], color='0.6', s=5, marker='X', alpha=0.25)
                except:
                    continue

            if len(mc_endpoints) == 0:
                print("[WARN] No valid trajectories.")
                continue

            mc_endpoints = np.array(mc_endpoints)
            mu_mc = np.mean(mc_endpoints, axis=0)
            cov_mc = np.cov(mc_endpoints.T)
            eigvals = np.maximum(np.linalg.eigvalsh(cov_mc), 0)
            volume = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals))

            plot_3sigma_ellipsoid(ax, mu_mc, cov_mc, color='gray', alpha=0.25)

            ax.set_xlabel("X [km]")
            ax.set_ylabel("Y [km]")
            ax.set_zlabel("Z [km]")
            set_axes_equal(ax)

            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            ax.legend(handles=[
                Line2D([0], [0], color='black', lw=2.0, label='Nominal'),
                Line2D([0], [0], linestyle=':', color='0.6', lw=1.0, label='Monte Carlo'),
                Line2D([0], [0], marker='o', color='black', linestyle='', label='Start', markersize=5),
                Line2D([0], [0], marker='X', color='black', linestyle='', label='End', markersize=6),
                Patch(color='gray', alpha=0.25, label='3σ Ellipsoid (MC)')
            ], loc='upper left', bbox_to_anchor=(0.03, 1.07), frameon=True)

            plt.tight_layout()
            fname = f"tk{int(t_frac*100)}_unc_{level}.pdf"
            plt.savefig(os.path.join("uncertainty_aware_outputs", fname), bbox_inches='tight', dpi=600,  pad_inches=0.5)
            plt.close()
            print(f"[Saved] {fname}")

            summary.append({
                "t_frac": t_frac,
                "uncertainty": level,
                "volume": volume,
                "n_samples": len(mc_endpoints)
            })

    df = pd.DataFrame(summary)
    df.to_csv("uncertainty_aware_outputs/mc_ellipsoid_volumes.csv", index=False)
    print("[Saved] mc_ellipsoid_volumes.csv")
    print(df)

if __name__ == "__main__":
    main()
