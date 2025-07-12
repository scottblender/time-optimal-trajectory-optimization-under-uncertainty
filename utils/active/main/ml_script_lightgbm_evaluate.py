import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.integrate import solve_ivp
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import mee2rv
import rv2mee
import odefunc

mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81

P_pos = np.eye(3) * 0.01
P_vel = np.eye(3) * 0.0001
P_mass = np.array([[0.0001]])
P_init = np.block([
    [P_pos, np.zeros((3, 3)), np.zeros((3, 1))],
    [np.zeros((3, 3)), P_vel, np.zeros((3, 1))],
    [np.zeros((1, 3)), np.zeros((1, 3)), P_mass]
])

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

def compute_kl_divergence(mu1, sigma1, mu2, sigma2):
    k = mu1.shape[0]
    sigma2_inv = np.linalg.inv(sigma2)
    trace_term = np.trace(sigma2_inv @ sigma1)
    diff = mu2 - mu1
    quadratic_term = diff.T @ sigma2_inv @ diff
    sign1, logdet1 = np.linalg.slogdet(sigma1)
    sign2, logdet2 = np.linalg.slogdet(sigma2)
    if sign1 <= 0 or sign2 <= 0:
        print(f"[WARN] Non-positive-definite covariance matrix detected")
        return 0.0
    log_det_term = logdet2 - logdet1
    kl_div = 0.5 * (trace_term + quadratic_term - k + log_det_term)
    return max(kl_div, 0.0)

def monte_carlo_propagation(init_state_mee, control_profile, segments, num_samples=300):
    r0_eci, v0_eci = mee2rv.mee2rv(*[np.array([val]) for val in init_state_mee[:6]], mu)
    m0_val = init_state_mee[6]
    mean_eci = np.hstack([r0_eci.flatten(), v0_eci.flatten(), m0_val])
    samples_eci = np.random.multivariate_normal(mean_eci, P_init, size=num_samples)

    r_hist = []
    for sample in tqdm(samples_eci, desc="Monte Carlo", leave=False):
        r_sample, v_sample = sample[:3], sample[3:6]
        m_sample = sample[6]
        r_sample = r_sample.reshape(1, 3)
        v_sample = v_sample.reshape(1, 3)
        state = np.hstack([rv2mee.rv2mee(r_sample, v_sample, mu), m_sample])

        r_traj = []
        for i, (t0, t1) in enumerate(segments):
            lam = control_profile[i]
            S = np.concatenate([state, lam])
            t_eval = np.linspace(t0, t1, 20)
            Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                           [t0, t1], S, t_eval=t_eval)
            r, _ = mee2rv.mee2rv(*Sf.y[:6], mu)
            r_traj.append(r)
            state = Sf.y[:7, -1]
        r_hist.append(np.vstack(r_traj))
    return np.array(r_hist)

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.2, scale=3.0):
    from scipy.linalg import eigh
    vals, vecs = eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    rx, ry, rz = scale * np.sqrt(vals)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))
    coords = np.stack((x, y, z), axis=-1)
    coords = coords @ vecs.T + mean
    ax.plot_surface(coords[..., 0], coords[..., 1], coords[..., 2],
                    rstride=2, cstride=2, color=color, alpha=alpha, edgecolor='none')

def evaluate_segment(X, y, model, Wm, Wc, label):
    y_pred = model.predict(X[:, :-2])
    time_vals = np.unique(X[:, 0])
    segments = list(zip(time_vals[:-1], time_vals[1:]))

    sigma_ids = np.unique(X[:, -1]).astype(int)
    r_pred_stack = []
    control_profile = []
    init_state = None

    for sigma_idx in tqdm(sigma_ids, desc=f"{label}: Sigma Prop", leave=False):
        r_sigma = []
        for i, (t0, t1) in enumerate(segments):
            row = X[(X[:, 0] == t0) & (X[:, -1] == sigma_idx)][0]
            lam = y_pred[(X[:, 0] == t0) & (X[:, -1] == sigma_idx)][0]
            S = np.concatenate([row[1:8], lam])
            t_eval = np.linspace(t0, t1, 20)
            Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                           [t0, t1], S, t_eval=t_eval)
            r, _ = mee2rv.mee2rv(*Sf.y[:6], mu)
            r_sigma.append(r)
            if sigma_idx == 0 and i == 0:
                init_state = row[1:8]
            if sigma_idx == 0:
                control_profile.append(lam)
        r_pred_stack.append(np.vstack(r_sigma))

    r_pred_stack = np.array(r_pred_stack)
    mean_r = np.sum(Wm[:, None, None] * r_pred_stack, axis=0)
    devs_r = r_pred_stack - mean_r[None]
    P_full_r = np.einsum("i,ijk,ijl->jkl", Wc, devs_r, devs_r)
    P_diag_r = np.array([np.diag(np.diag(P)) for P in P_full_r])
    inv_cov_r = inv(P_diag_r[-1])
    r_pred = r_pred_stack[0]

    r_mc = monte_carlo_propagation(init_state, control_profile, segments)
    mean_mc = np.mean(r_mc, axis=0)
    devs_mc = r_mc - mean_mc[None]
    W_mc = np.ones(r_mc.shape[0]) / r_mc.shape[0]
    P_mc_full = np.einsum("i,ijk,ijl->jkl", W_mc, devs_mc, devs_mc)
    P_mc_diag = np.array([np.diag(np.diag(P)) for P in P_mc_full])
    inv_cov_mc = inv(P_mc_diag[-1])

    final_time = time_vals[-1]
    row_sigma0 = X[(X[:, 0] == final_time) & (X[:, -1] == 0) & np.all(X[:, 8:15] == 0, axis=1)][0]
    r_ref, _ = mee2rv.mee2rv(*row_sigma0[1:7].reshape(6, 1), mu)
    r_ref = r_ref.flatten()

    if label.endswith("bundle_0"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for traj in r_mc[:20]:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='black', alpha=0.2)
        ax.plot(r_pred[:, 0], r_pred[:, 1], r_pred[:, 2], label="Predicted (sigma 0)", linewidth=2)
        ax.plot(mean_mc[:, 0], mean_mc[:, 1], mean_mc[:, 2], label="MC Mean", linestyle='--')
        plot_3sigma_ellipsoid(ax, mean_r[-1], P_diag_r[-1], color='blue', alpha=0.2)
        plot_3sigma_ellipsoid(ax, mean_mc[-1], P_mc_diag[-1], color='red', alpha=0.2)
        ax.scatter(*r_ref, color='green', label="Reference")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"Trajectory Comparison ({label})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"trajectory_plot_{label}.png", dpi=600)
        plt.close()

    mahalanobis_pred = mahalanobis(r_pred[-1], r_ref, inv_cov_r)
    mahalanobis_mc_mean = mahalanobis(mean_mc[-1], r_ref, inv_cov_mc)
    mahalanobis_list = [
        mahalanobis(mee2rv.mee2rv(*X[(X[:, 0] == final_time) & (X[:, -1] == idx)][0][1:7].reshape(6, 1), mu)[0].flatten(),
                    r_ref, inv_cov_r)
        for idx in sigma_ids[1:]
    ]

    kl_div = compute_kl_divergence(mean_r[-1], P_diag_r[-1], mean_mc[-1], P_mc_diag[-1])
    lam_pred_0 = control_profile[0]
    lam_true_0 = y[(X[:, 0] == time_vals[0]) & (X[:, -1] == 0)][0]
    cos_sim = cosine_similarity(lam_pred_0, lam_true_0)
    mc_vs_pred_mse = np.mean((r_pred - mean_mc) ** 2)
    control_true = [y[(X[:, 0] == t) & (X[:, -1] == 0)][0] for t in time_vals[:-1]]
    control_mse = mean_squared_error(np.array(control_profile), np.array(control_true))

    return {
        "bundle": label,
        "cosine similarity": cos_sim,
        "final position deviation": np.linalg.norm(r_pred[-1] - r_ref),
        "mahalanobis_pred": mahalanobis_pred,
        "mahalanobis_mc_mean": mahalanobis_mc_mean,
        "trace diff (cov)": np.trace(np.abs(P_mc_diag[-1] - P_diag_r[-1])),
        "KL divergence (pred || MC)": kl_div,
        "mc vs predicted trajectory mse": mc_vs_pred_mse,
        "control MSE": control_mse,
        "x mse": np.mean((r_ref[0] - r_pred[:, 0]) ** 2),
        "y mse": np.mean((r_ref[1] - r_pred[:, 1]) ** 2),
        "z mse": np.mean((r_ref[2] - r_pred[:, 2]) ** 2),
        "mahalanobis > 3 count": sum(d > 3 for d in mahalanobis_list)
    }

if __name__ == "__main__":
    model = joblib.load("trained_model.pkl")
    Wm = joblib.load("Wm.pkl")
    Wc = joblib.load("Wc.pkl")
    data_max = joblib.load("segment_max.pkl")
    data_min = joblib.load("segment_min.pkl")

    results = []

    for bundle_id in tqdm(np.unique(data_max["X"][:, -2]).astype(int), desc="Max segment bundles"):
        Xb = data_max["X"][data_max["X"][:, -2] == bundle_id]
        yb = data_max["y"][data_max["X"][:, -2] == bundle_id]
        label = f"max_bundle_{bundle_id}"
        results.append(evaluate_segment(Xb, yb, model, Wm, Wc, label))

    for bundle_id in tqdm(np.unique(data_min["X"][:, -2]).astype(int), desc="Min segment bundles"):
        Xb = data_min["X"][data_min["X"][:, -2] == bundle_id]
        yb = data_min["y"][data_min["X"][:, -2] == bundle_id]
        label = f"min_bundle_{bundle_id}"
        results.append(evaluate_segment(Xb, yb, model, Wm, Wc, label))

    pd.DataFrame(results).to_csv("ml_eval_mc_serial_metrics.csv", index=False)
    print("Saved: ml_eval_mc_serial_metrics.csv")
