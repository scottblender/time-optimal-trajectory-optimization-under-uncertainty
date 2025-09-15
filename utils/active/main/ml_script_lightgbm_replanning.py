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
    p, f, g, h, k, L = mee
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

def run_replanning_simulation(model, t_start_replan, t_end_replan, window_type, data, diag_mins, diag_maxs):
    print("\n" + "="*20 + f" RUNNING FOR {window_type.upper()} WINDOW " + "="*20)
    print(f"[t_start = {t_start_replan:.3f}] Replanning for window {t_start_replan:.2f} -> {t_end_replan:.2f} TU")

    mu, F_nom, c, m0, g0= data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    r_nom, v_nom, mass_nom, lam_tr = data["r_tr"], data["v_tr"], data["mass_tr"], data["lam_tr"]
    del_t_nom = data["del_t_nom"]
    t_vals = np.asarray(data["backTspan"][::-1])
    
    rand_diag_1 = np.random.uniform(low=diag_mins, high=diag_maxs, size=len(diag_mins))
    rand_diag_2 = np.random.uniform(low=diag_mins, high=diag_maxs, size=len(diag_mins))
    P_models = {'random1': np.diag(rand_diag_1), 'random2': np.diag(rand_diag_2)}
    
    DU_km = 696340.0
    F_val = 0.9 * F_nom
    # --- NEW: Initialize lists to store new data ---
    summary, distance_summary, thrust_summary = [], [], []

    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val = r_nom[idx_start], v_nom[idx_start], mass_nom[idx_start]
    
    nominal_state_replan_start = np.hstack([rv2mee(r0.reshape(1, 3), v0.reshape(1, 3), mu).flatten(), m0_val, lam_tr[idx_start]])
    del_t_nom_start = del_t_nom[idx_start]

    sol_nominal = solve_ivp(lambda t, x: odefunc(t, x, mu, F_nom, c, m0, g0), [t_start_replan, t_end_replan], nominal_state_replan_start, t_eval=np.linspace(t_start_replan, t_end_replan, 100))
    replan_r_nom, _ = mee2rv(*sol_nominal.y[:6], mu)

    sol_nominal_Fval = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_replan, t_end_replan], nominal_state_replan_start, t_eval=np.linspace(t_start_replan, t_end_replan, 100))
    replan_r_Fval, _ = mee2rv(*sol_nominal_Fval.y[:6], mu)
    
    P_pos_km2, P_vel_kms2, P_mass_kg2 = np.eye(3)*0.01, np.eye(3)*1e-10, np.array([[1e-3]])
    P_cart = np.block([[P_pos_km2/(DU_km**2), np.zeros((3,4))], [np.zeros((4,3)), np.diag([0,0,0,0])]]) # Simplified
    state_k = np.hstack([r0, v0, m0_val])
    shared_samples = np.random.multivariate_normal(state_k, P_cart[:7, :7], size=1000)
    
    for level, P_model in P_models.items():
        print(f"  Level: {level} — propagating MC...")
        mc_endpoints, all_mc_r = [], []
        
        for s in shared_samples:
            try:
                mee = np.hstack([rv2mee(s[:3].reshape(1, 3), s[3:6].reshape(1, 3), mu).flatten(), s[6]])
                x_input = np.hstack([t_start_replan, mee, np.diag(P_model)])
                lam = model.predict(pd.DataFrame([x_input], columns=['t', 'p', 'f', 'g', 'h', 'k','L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']))[0]
                S = np.hstack([mee, lam])
                
                sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), [t_start_replan, t_end_replan], S, t_eval=np.linspace(t_start_replan, t_end_replan, 100))
                r, _ = mee2rv(*sol.y[:6], mu)
                all_mc_r.append(r)
                mc_endpoints.append(r[-1])
            except Exception:
                continue

        mc_endpoints = np.array(mc_endpoints)
        mu_mc_endpoints = np.mean(mc_endpoints, axis=0)
        mc_mean_trajectory_r = np.mean(all_mc_r, axis=0)
        
        # --- NEW: Calculate and store thrust vectors and distances ---
        del_t_pred = compute_thrust_direction(mu, F_val, nominal_state_replan_start[:6], model.predict(pd.DataFrame([np.hstack([t_start_replan, nominal_state_replan_start[:7], np.diag(P_model)])], columns=['t', 'p', 'f', 'g', 'h', 'k','L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']))[0])
        thrust_summary.append({ 'level': level, 'pred_x': del_t_pred[0], 'pred_y': del_t_pred[1], 'pred_z': del_t_pred[2], 'nom_x': del_t_nom_start[0], 'nom_y': del_t_nom_start[1], 'nom_z': del_t_nom_start[2], 'diff_x': del_t_pred[0]-del_t_nom_start[0], 'diff_y': del_t_pred[1]-del_t_nom_start[1], 'diff_z': del_t_pred[2]-del_t_nom_start[2] })
        distance_summary.append({ 'level': level, 'dist_mc_vs_fnom': np.linalg.norm(mu_mc_endpoints - replan_r_nom[-1]), 'dist_mc_vs_fval': np.linalg.norm(mu_mc_endpoints - replan_r_Fval[-1]), 'dist_fnom_vs_fval': np.linalg.norm(replan_r_nom[-1] - replan_r_Fval[-1]) })

        # --- Plotting ---
        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(replan_r_nom[:, 0], replan_r_nom[:, 1], replan_r_nom[:, 2], color='black', lw=2.0, label="Nominal (F_nom)")
        ax.plot(replan_r_Fval[:, 0], replan_r_Fval[:, 1], replan_r_Fval[:, 2], color='blue', linestyle='-.', lw=2.0, label="Nominal (F_val)")
        ax.plot(mc_mean_trajectory_r[:, 0], mc_mean_trajectory_r[:, 1], mc_mean_trajectory_r[:, 2], linestyle='--', color='red', lw=1.5, label='MC Mean')
        ax.scatter(*replan_r_nom[0], color='black', s=20, zorder=5)
        plot_3sigma_ellipsoid(ax, mu_mc_endpoints, np.cov(mc_endpoints.T), color='gray', alpha=0.25)
        
        # --- NEW: Enhanced Inset Plot ---
        inset_ax = fig.add_axes([0.63, 0.73, 0.34, 0.34], projection='3d')
        inset_ax.scatter(mc_endpoints[:,0], mc_endpoints[:,1], mc_endpoints[:,2], color='0.6', s=5, marker='.', alpha=0.3) # regular x
        # Plot last 3 steps of trajectories as dashed lines
        inset_ax.plot(*replan_r_nom[-4:].T, color='black', linestyle='--')
        inset_ax.plot(*replan_r_Fval[-4:].T, color='blue', linestyle='--')
        inset_ax.plot(*mc_mean_trajectory_r[-4:].T, color='red', linestyle='--')
        # Plot endpoints
        inset_ax.scatter(*replan_r_nom[-1], color='black', s=60, marker='X')
        inset_ax.scatter(*replan_r_Fval[-1], color='blue', s=60, marker='P') # P for 'Propagated'
        inset_ax.scatter(*mu_mc_endpoints, color='red', s=60, marker='*')
        # Adjust zoom
        all_points = np.vstack([replan_r_nom[-3:], replan_r_Fval[-3:], mc_mean_trajectory_r[-3:]])
        center = np.mean(all_points, axis=0)
        radius = np.max(np.linalg.norm(all_points - center, axis=1)) * 1.2
        inset_ax.set_xlim(center[0]-radius, center[0]+radius); inset_ax.set_ylim(center[1]-radius, center[1]+radius); inset_ax.set_zlim(center[2]-radius, center[2]+radius)
        inset_ax.set_xticks([]); inset_ax.set_yticks([]); inset_ax.set_zticks([])
        inset_ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel("X [DU]"); ax.set_ylabel("Y [DU]"); ax.set_zlabel("Z [DU]")
        set_axes_equal(ax)
        from matplotlib.lines import Line2D; from matplotlib.patches import Patch
        ax.legend(handles=[ Line2D([0], [0], color='black', lw=2.0, label='Nominal (F_nom)'), Line2D([0], [0], color='blue', linestyle='-.', lw=2.0, label='Nominal (F_val)'), Line2D([0], [0], linestyle='--', color='red', lw=1.5, label='MC Mean'), Patch(color='gray', alpha=0.25, label='3σ Ellipsoid (MC)') ], loc='upper left', bbox_to_anchor=(0.03, 1), frameon=True)
        plt.tight_layout()
        fname = f"tk_{t_start_replan:.3f}_unc_{level}_{window_type}.pdf"
        plt.savefig(os.path.join("uncertainty_aware_outputs", fname), bbox_inches='tight', dpi=600)
        plt.close()
        print(f"  [Saved] {fname}")

    # --- NEW: Save the collected data to files ---
    pd.DataFrame(thrust_summary).to_csv(os.path.join("uncertainty_aware_outputs", f"thrust_vectors_{window_type}.csv"), index=False)
    print(f"  [Saved] thrust_vectors_{window_type}.csv")
    pd.DataFrame(distance_summary).to_excel(os.path.join("uncertainty_aware_outputs", f"endpoint_distances_{window_type}.xlsx"), index=False)
    print(f"  [Saved] endpoint_distances_{window_type}.xlsx")

def main():
    try:
        model_max = joblib.load("trained_model_max.pkl")
        model_min = joblib.load("trained_model_min.pkl")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find model file: {e}. Please run the training script first.")
        return

    data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
    os.makedirs("uncertainty_aware_outputs", exist_ok=True)
    
    width_file_path = "stride_1440min/bundle_segment_widths.txt"
    if not os.path.exists(width_file_path):
        print(f"[ERROR] Width file not found: {width_file_path}")
        return

    with open(width_file_path) as f: lines = f.readlines()[1:]
    times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    time_vals = times_arr[:, 0]
    
    max_idx = int(np.argmax(times_arr[:, 1]))
    t_start_replan_max, t_end_replan_max = time_vals[max(0, max_idx - 2)], time_vals[max(0, max_idx + 2)]

    min_idx_raw = int(np.argmin(times_arr[:, 1]))
    min_idx = np.argsort(times_arr[:, 1])[1] if min_idx_raw == len(times_arr) - 1 else min_idx_raw
    t_start_replan_min, t_end_replan_min = time_vals[max(0, min_idx - 2)], time_vals[max(0, max_idx + 2)]

    try:
        diag_mins, diag_maxs = np.load("diag_mins.npy"), np.load("diag_maxs.npy")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find covariance files: {e}. Please run the training script first.")
        return
        
    run_replanning_simulation(model=model_max, t_start_replan=t_start_replan_max, t_end_replan=t_end_replan_max, window_type='max', data=data, diag_mins=diag_mins, diag_maxs=diag_maxs)
    run_replanning_simulation(model=model_min, t_start_replan=t_start_replan_min, t_end_replan=t_end_replan_min, window_type='min', data=data, diag_mins=diag_mins, diag_maxs=diag_maxs)

if __name__ == "__main__":
    main()