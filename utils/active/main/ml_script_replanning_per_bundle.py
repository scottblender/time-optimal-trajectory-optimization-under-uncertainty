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
from filterpy.kalman import MerweScaledSigmaPoints

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

def propagate(state, t0, tf, F_val, model, scaler, mu, c, m0, g0, P_model):
    try:
        mee = np.hstack([rv2mee(state[:3].reshape(1,3), state[3:6].reshape(1,3), mu).flatten(), state[6]])
        x_input = np.hstack([t0, mee, np.zeros(7)])
        cov_features = np.hstack([P_model[i, i] for i in range(7)])
        x_input[8:] = cov_features
        x_df = pd.DataFrame([x_input], columns=[
            't', 'p', 'f', 'g', 'h', 'L', 'mass',
            'dummy1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'
        ])
        lam = model.predict(scaler.transform(x_df))[0]
        print(f"[DEBUG] λ from P_model (diagonal): {np.round(cov_features, 6)} => λ = {np.round(lam, 6)}")
        S = np.hstack([mee, lam])
        sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0),
                        [t0, tf], S, t_eval=np.linspace(t0, tf, 200))
        r_out, _ = mee2rv(*sol.y[:6], mu)
        return r_out
    except Exception:
        return None

def main():
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    data = joblib.load("stride_4000min/bundle_data_4000min.pkl")
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    r_nom, v_nom, mass_nom = data["r_tr"], data["v_tr"], data["mass_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    os.makedirs("uncertainty_aware_outputs", exist_ok=True)

    t_frac_fixed = 0.9
    t_k = t_frac_fixed * t_vals[-1]
    idx = np.argmin(np.abs(t_vals - t_k))
    r0, v0, m0_val = r_nom[idx], v_nom[idx], mass_nom[idx]
    state_nom = np.hstack([r0, v0, m0_val])

    # === Realistic MEE+mass covariance diagonal ranges ===
    diag_mins = np.array([
        9.098224e+01,   # p
        7.082445e-04,   # f
        6.788599e-04,   # g
        1.376023e-08,   # h
        2.346605e-08,   # k
        5.885859e-08,   # L
        1.000000e-04    # mass
    ])

    diag_maxs = np.array([
        1.977489e+03,
        6.013501e-03,
        5.173225e-03,
        4.284912e-04,
        1.023625e-03,
        6.818500e+00,
        1.000000e-04
    ])

    cov_setups = {}
    num_levels = 4
    for i, alpha in enumerate(np.linspace(0.0, 1.0, num_levels)):
        diag_vals = diag_mins + alpha * (diag_maxs - diag_mins)
        P_model = np.diag(diag_vals)
        cov_setups[i+1] = P_model

    # === Base Cartesian covariance for sigmas and MC ===
    P_cart_base = np.block([
        [np.eye(3)*0.01,       np.zeros((3,3)), np.zeros((3,1))],
        [np.zeros((3,3)), np.eye(3)*0.0001,     np.zeros((3,1))],
        [np.zeros((1,3)), np.zeros((1,3)),      np.array([[0.0001]])]
    ])

    nsd = 7
    alpha = np.sqrt(9 / (nsd + (3 - nsd)))
    beta = 2
    kappa = 3 - nsd
    sp = MerweScaledSigmaPoints(n=nsd, alpha=alpha, beta=beta, kappa=kappa)
    sigma_states = sp.sigma_points(state_nom, P_cart_base)
    Wm, Wc = sp.Wm, sp.Wc
    mc_samples = np.random.multivariate_normal(state_nom, P_cart_base, size=100)

    thrust_vals = [0.85, 0.925, 1.0]
    summary_rows = []

    for thrust_scale in thrust_vals:
        for cov_id, P_model in cov_setups.items():
            fig = plt.figure(figsize=(6.5, 5.5))
            ax = fig.add_subplot(111, projection='3d')
            tf = t_vals[-1]
            F_val = F_nom * thrust_scale

            ax.plot(r_nom[:, 0], r_nom[:, 1], r_nom[:, 2], color='black', lw=2.0, label="Nominal")
            ax.scatter(*r_nom[idx], color='black', s=20, label="Start", zorder=5)
            ax.scatter(*r_nom[-1], color='black', s=20, marker='X', label="End", zorder=5)

            sigma_endpoints, mc_endpoints = [], []

            for s in sigma_states:
                traj = propagate(s, t_k, tf, F_val, model, scaler, mu, c, m0, g0, P_model)
                if traj is not None:
                    sigma_endpoints.append(traj[-1])
                    ax.plot(*traj.T, linestyle='--', color='gray', lw=0.8)
                    ax.scatter(*traj[0], color='gray', s=6)
                    ax.scatter(*traj[-1], color='gray', s=6, marker='X')

            for s in mc_samples:
                traj = propagate(s, t_k, tf, F_val, model, scaler, mu, c, m0, g0, P_model)
                if traj is not None:
                    mc_endpoints.append(traj[-1])
                    ax.plot(*traj.T, linestyle=':', color='0.6', lw=0.8, alpha=0.25)
                    ax.scatter(*traj[0], color='0.6', s=5, alpha=0.25)
                    ax.scatter(*traj[-1], color='0.6', s=5, marker='X', alpha=0.25)

            r_sigma_end = np.array(sigma_endpoints)
            mu_sigma = np.sum(Wm[:, None] * r_sigma_end, axis=0)
            cov_sigma = np.einsum("i,ij,ik->jk", Wc, r_sigma_end - mu_sigma, r_sigma_end - mu_sigma)
            plot_3sigma_ellipsoid(ax, mu_sigma, cov_sigma, color='gray', alpha=0.2)

            r_mc_end = np.array(mc_endpoints)
            mu_mc = np.mean(r_mc_end, axis=0)

            summary_rows.append({
                'thrust': thrust_scale,
                'cov_id': cov_id,
                'mean_sigma_dev': np.mean(np.linalg.norm(r_sigma_end - r_nom[-1], axis=1)),
                'mean_mc_dev': np.mean(np.linalg.norm(r_mc_end - r_nom[-1], axis=1))
            })

            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0], [0], color='black', lw=2.0, label='Nominal'),
                Line2D([0], [0], linestyle='--', color='gray', lw=1.0, label='Sigma Points'),
                Line2D([0], [0], linestyle=':', color='0.6', lw=1.0, label='Monte Carlo'),
                Line2D([0], [0], marker='o', color='black', linestyle='', label='Start', markersize=5),
                Line2D([0], [0], marker='X', color='black', linestyle='', label='End', markersize=6),
                Patch(color='gray', alpha=0.2, label='3σ Ellipsoid (σ)')
            ], loc='upper left', fontsize=8)

            ax.set_xlabel("X [km]")
            ax.set_ylabel("Y [km]")
            ax.set_zlabel("Z [km]")
            set_axes_equal(ax)
            plt.tight_layout()

            fname = f"tk{int(t_frac_fixed*100)}_F{int(thrust_scale*100)}_cov{cov_id}.pdf"
            plt.savefig(os.path.join("uncertainty_aware_outputs", fname), bbox_inches='tight', pad_inches=0.5)
            plt.close()
            print(f"[Saved] {fname}")

    df = pd.DataFrame(summary_rows)
    df.to_csv("uncertainty_aware_outputs/uncertainty_summary.csv", index=False)
    print("[Saved] uncertainty_summary.csv")

    df["uncertainty_level"] = df["cov_id"].map({k: np.trace(P) for k, P in cov_setups.items()})

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for thrust_scale in sorted(df["thrust"].unique()):
        df_t = df[df["thrust"] == thrust_scale]
        ax.plot(df_t["uncertainty_level"], df_t["mean_sigma_dev"],
                marker='o', label=f"σ Mean (F={thrust_scale})", linestyle='--', color='gray')
        ax.plot(df_t["uncertainty_level"], df_t["mean_mc_dev"],
                marker='x', label=f"MC Mean (F={thrust_scale})", linestyle=':', color='black')

    ax.set_xlabel("Total Initial Uncertainty Level (Trace of $P_0$)")
    ax.set_ylabel("Deviation from Nominal Final [km]")
    ax.set_title("Effect of Initial Uncertainty on Final Position Deviation")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig("uncertainty_aware_outputs/final_deviation_vs_uncertainty.pdf", bbox_inches='tight')
    plt.close()
    print("[Saved] final_deviation_vs_uncertainty.pdf")
    # === Lambda sensitivity plot ===
    lambda_base = None
    trace_vals, delta_lambdas = [], []

    for cov_id, P_model in cov_setups.items():
        mee = np.hstack([rv2mee(state_nom[:3].reshape(1,3), state_nom[3:6].reshape(1,3), mu).flatten(), state_nom[6]])
        x_input = np.hstack([t_k, mee, np.zeros(7)])
        cov_diag = np.diag(P_model)
        x_input[8:] = cov_diag
        x_df = pd.DataFrame([x_input], columns=[
            't', 'p', 'f', 'g', 'h', 'L', 'mass',
            'dummy1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'
        ])
        lam_pred = model.predict(scaler.transform(x_df))[0]
        if lambda_base is None:
            lambda_base = lam_pred
        delta = np.abs(lam_pred - lambda_base)
        delta_lambdas.append(delta)
        trace_vals.append(np.trace(P_model))

    delta_lambdas = np.array(delta_lambdas)
    trace_vals = np.array(trace_vals)
    rel_deltas = delta_lambdas / (np.abs(lambda_base) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(7):
        ax.plot(trace_vals, rel_deltas[:, i], label=f"$\lambda_{i+1}$", marker='o')
    ax.set_xlabel("Trace of Initial Covariance ($\mathrm{Tr}(P_0)$)")
    ax.set_ylabel("Relative Change in λ")
    ax.set_title("Sensitivity of Control λ to Initial Uncertainty")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("uncertainty_aware_outputs/lambda_sensitivity_plot.pdf", bbox_inches='tight')
    plt.close()
    print("[Saved] lambda_sensitivity_plot.pdf")

if __name__ == "__main__":
    main()
