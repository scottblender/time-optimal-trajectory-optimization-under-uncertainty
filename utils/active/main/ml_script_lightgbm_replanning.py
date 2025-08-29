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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({'font.size': 10})
warnings.filterwarnings("ignore", category=UserWarning)
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

def compute_thrust_direction(mu, F, mee, lam):
    p, f, g, h, k, L = mee[:-1]
    lam_p, lam_f, lam_g, lam_h, lam_k, lam_L = lam[:-1]
    lam_matrix = np.array([[lam_p, lam_f, lam_g, lam_h, lam_k, lam_L]]).T
    SinL, CosL = np.sin(L), np.cos(L)
    w = 1 + f * CosL + g * SinL
    if np.isclose(w, 0, atol=1e-10):
        return np.full(3, np.nan)
    s = 1 + h**2 + k**2
    C1 = np.sqrt(p / mu)
    C2 = 1 / w
    C3 = h * SinL - k * CosL
    A = np.array([
        [0, 2 * p * C2 * C1, 0],
        [C1 * SinL, C1 * C2 * ((w + 1) * CosL + f), -C1 * (g / w) * C3],
        [-C1 * CosL, C1 * C2 * ((w + 1) * SinL + g), C1 * (f / w) * C3],
        [0, 0, C1 * s * CosL * C2 / 2],
        [0, 0, C1 * s * SinL * C2 / 2],
        [0, 0, C1 * C2 * C3]
    ])
    mat = A.T @ lam_matrix
    return mat.flatten() / np.linalg.norm(mat)

def main():
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    data = joblib.load("stride_4000min/bundle_data_4000min.pkl")
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    r_nom, v_nom, mass_nom = data["r_tr"], data["v_tr"], data["mass_tr"]
    # Load nominal thrust vector from the updated pkl file
    del_t_nom = data["del_t_nom"]
    t_vals = np.asarray(data["backTspan"][::-1])
    tf = t_vals[-1]
    os.makedirs("uncertainty_aware_outputs", exist_ok=True)

    # === Load segment times from width file ===
    with open("stride_4000min/bundle_segment_widths.txt") as f:
        lines = f.readlines()[1:]
        times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
        time_vals = times_arr[:, 0]
        max_idx = int(np.argmax(times_arr[:, 1]))
    
    # Define replanning start and end times for a single event
    t_start_replan = time_vals[max(0, max_idx - 3)]
    t_end_replan = time_vals[max_idx + 3]

    # === Load covariance diagonals from file ===
    try:
        diag_mins = np.load("diag_mins.npy")
        diag_maxs = np.load("diag_maxs.npy")
        P_models = {'min': np.diag(diag_mins), 'max': np.diag(diag_maxs)}
    except FileNotFoundError:
        print("[ERROR] Covariance diagonal files not found. Using default hardcoded values.")
        diag_mins = np.array([1.319000e-08, 1.038710e-13, 6.880754e-14, 2.220821e-16, 2.220804e-16, 2.221686e-16, 6.250022e-11])
        diag_maxs = np.array([1.211294e-06, 5.092024e-12, 1.916482e-12, 7.411871e-14, 2.897651e-13, 1.786088e-15, 6.250022e-11])
        P_models = {'min': np.diag(diag_mins), 'max': np.diag(diag_maxs)}
    
    DU_km = 696340.0  # Sun radius in km
    g0_s = 9.81/1000
    TU = np.sqrt(DU_km / g0_s)
    VU_kms = DU_km / TU
    # Physical covariances (km / km/s / kg), then -> non-dimensional
    P_pos_km2  = np.eye(3) * 0.01
    P_vel_kms2 = np.eye(3) * 1e-10
    P_mass_kg2 = np.array([[1e-3]])
    P_pos  = P_pos_km2  / (DU_km**2)
    P_vel  = P_vel_kms2 / (VU_kms**2)
    P_mass = P_mass_kg2 / (4000**2)
    P_cart = np.block([
        [P_pos,       np.zeros((3,3)), np.zeros((3,1))],
        [np.zeros((3,3)), P_vel,     np.zeros((3,1))],
        [np.zeros((1,3)), np.zeros((1,3)),      P_mass]
    ])
    F_val = 0.9 * F_nom
    summary = []

    print(f"[t_start = {t_start_replan:.3f}] Replanning for window {t_start_replan:.2f} -> {t_end_replan:.2f} TU")
    
    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val = r_nom[idx_start], v_nom[idx_start], mass_nom[idx_start]
    state_k = np.hstack([r0, v0, m0_val])
    shared_samples = np.random.multivariate_normal(state_k, P_cart, size=1000)
    
    # Get the nominal thrust vector at the start of the replanning window
    nominal_plot_start_idx = np.argmin(np.abs(t_vals - t_start_replan))
    del_t_nom_start = del_t_nom[nominal_plot_start_idx]

    for level, P_model in P_models.items():
        print(f"  Level: {level} — propagating MC...")
        mc_endpoints = []
        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_subplot(111, projection='3d')

        nominal_plot_start_idx = np.argmin(np.abs(t_vals - t_start_replan))
        nominal_plot_end_idx = np.argmin(np.abs(t_vals - t_end_replan))
        ax.plot(r_nom[nominal_plot_start_idx:nominal_plot_end_idx, 0], r_nom[nominal_plot_start_idx:nominal_plot_end_idx, 1], r_nom[nominal_plot_start_idx:nominal_plot_end_idx, 2], color='black', lw=2.0, label="Nominal")
        
        ax.scatter(*r_nom[nominal_plot_start_idx], color='black', s=20, label="Start", zorder=5)
        ax.scatter(*r_nom[nominal_plot_end_idx], color='black', s=20, marker='X', label="End")

        # Get the nominal state at the start of the replanning window for thrust vector calculation
        mee0 = np.hstack([rv2mee(r_nom[nominal_plot_start_idx].reshape(1, 3), v_nom[nominal_plot_start_idx].reshape(1, 3), mu).flatten(), mass_nom[nominal_plot_start_idx]])

        # Get the predicted lambda from the model for the start of the window
        x_input_pred = np.hstack([t_start_replan, mee0, np.diag(P_model)])
        x_df_pred = pd.DataFrame([x_input_pred], columns=[
            't', 'p', 'f', 'g', 'h', 'L', 'mass',
            'dummy1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'
        ])
        lam_pred = model.predict(x_df_pred)[0]

        # Calculate the predicted thrust vector
        del_t_pred = compute_thrust_direction(mu, F_val, mee0, lam_pred)

        # Calculate the difference between predicted and nominal
        del_t_diff = del_t_pred - del_t_nom_start
        print(f"Predicted thrust vector for {level} covariance: {del_t_pred}")
        print(f"Nominal thrust vector: {del_t_nom_start}")
        print(f"Difference (Predicted - Nominal): {del_t_diff}")

        for i, s in enumerate(shared_samples):
            try:
                mee = np.hstack([rv2mee(s[:3].reshape(1, 3), s[3:6].reshape(1, 3), mu).flatten(), s[6]])
                x_input = np.hstack([t_start_replan, mee, np.diag(P_model)])
                x_df = pd.DataFrame([x_input], columns=[
                    't', 'p', 'f', 'g', 'h', 'L', 'mass',
                    'dummy1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'
                ])
                lam = model.predict(x_df)[0]
                S = np.hstack([mee, lam])
                sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0),
                                [t_start_replan, t_end_replan], S, t_eval=np.linspace(t_start_replan, t_end_replan, 100))
                r, _ = mee2rv(*sol.y[:6], mu)
                if i % 10 == 0:
                    ax.plot(r[:, 0], r[:, 1], r[:, 2], linestyle=':', color='red', lw=1.5, alpha=0.6)
                    ax.scatter(*r[-1], color='red', s=5, lw=1.5, marker='X', alpha=0.5)
                mc_endpoints.append(r[-1])
            except Exception as e:
                print(f"  [WARN] Sample {i} failed to propagate: {e}")
                continue

        mc_endpoints = np.array(mc_endpoints)
        mu_mc = np.mean(mc_endpoints, axis=0)
        
        print(f"MC - nominal:{mu_mc - r_nom[nominal_plot_end_idx]}")
        
        cov_mc = np.einsum("ni,nj->ij", mc_endpoints - mu_mc, mc_endpoints - mu_mc) / mc_endpoints.shape[0]
        eigvals = np.maximum(np.linalg.eigvalsh(cov_mc), 0)
        print(eigvals)
        volume = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals))
        plot_3sigma_ellipsoid(ax, mu_mc, cov_mc, color='gray', alpha=0.25)

        inset_ax = fig.add_axes([0.63, 0.73, 0.34, 0.34], projection='3d')
        for i, s in enumerate(shared_samples[::10]):
            try:
                mee = np.hstack([rv2mee(s[:3].reshape(1, 3), s[3:6].reshape(1, 3), mu).flatten(), s[6]])
                x_input = np.hstack([t_start_replan, mee, np.diag(P_model)])
                x_df = pd.DataFrame([x_input], columns=[
                    't', 'p', 'f', 'g', 'h', 'L', 'mass',
                    'dummy1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'
                ])
                lam = model.predict(x_df)[0]
                S = np.hstack([mee, lam])
                sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0),
                                [t_start_replan, t_end_replan], S, t_eval=np.linspace(t_start_replan, t_end_replan, 100))
                r, _ = mee2rv(*sol.y[:6], mu)
                inset_ax.scatter(*r[-1], color='0.6', s=5, marker='X', alpha=0.5)
            except:
                continue

        plot_3sigma_ellipsoid(inset_ax, mu_mc, cov_mc, color='gray', alpha=0.3)
        center = mu_mc
        radius = np.max(np.linalg.norm(mc_endpoints - center, axis=1)) * 1.2
        inset_ax.set_xlim(center[0] - radius, center[0] + radius)
        inset_ax.set_ylim(center[1] - radius, center[1] + radius)
        inset_ax.set_zlim(center[2] - radius, center[2] + radius)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_zticks([])
        inset_ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel("X [DU]")
        ax.set_ylabel("Y [DU]")
        ax.set_zlabel("Z [DU]")
        set_axes_equal(ax)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Line2D([0], [0], color='black', lw=2.0, label='Nominal'),
            Line2D([0], [0], linestyle=':', color='red', lw=1.0, label='Monte Carlo'),
            Line2D([0], [0], marker='o', color='black', linestyle='', label='Start', markersize=5),
            Line2D([0], [0], marker='X', color='black', linestyle='', label='End', markersize=6),
            Patch(color='gray', alpha=0.25, label='3σ Ellipsoid (MC)')
        ], loc='upper left', bbox_to_anchor=(0.03, 1), frameon=True)

        plt.tight_layout()
        fname = f"tk_{t_start_replan:.6f}_unc_{level}.pdf"
        plt.savefig(os.path.join("uncertainty_aware_outputs", fname), bbox_inches='tight', dpi=600, pad_inches=0.5)
        plt.close()
        print(f"[Saved] {fname}")

        summary.append({
            "t_start_replan": t_start_replan,
            "uncertainty": level,
            "volume": volume,
            "n_samples": len(mc_endpoints)
        })

    df = pd.DataFrame(summary)
    df.to_csv("uncertainty_aware_outputs/mc_ellipsoid_volumes_tmax_neighbors.csv", index=False)
    print("[Saved] mc_ellipsoid_volumes_tmax_neighbors.csv")
    print(df)

if __name__ == "__main__":
    main()