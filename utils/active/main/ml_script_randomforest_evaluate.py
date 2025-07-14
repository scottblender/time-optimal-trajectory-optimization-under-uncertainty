# File: ml_script_rf_evaluate_per_bundle.py

import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from tqdm import tqdm

plt.rcParams.update({'font.size': 8})
plt.rcParams['axes.grid'] = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import mee2rv
import rv2mee
import odefunc

# Constants
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

def monte_carlo_propagation(init_state_mee, control_profile, segments, num_samples=100):
    r0_eci, v0_eci = mee2rv.mee2rv(*[np.array([val]) for val in init_state_mee[:6]], mu)
    m0_val = init_state_mee[6]
    mean_eci = np.hstack([r0_eci.flatten(), v0_eci.flatten(), m0_val])
    samples_eci = np.random.multivariate_normal(mean_eci, P_init, size=num_samples)

    r_hist = []
    for sample in samples_eci:
        r_sample, v_sample = sample[:3], sample[3:6]
        m_sample = sample[6]
        r_sample = r_sample.reshape(1, 3)
        v_sample = v_sample.reshape(1, 3)
        state_mee = np.hstack([rv2mee.rv2mee(r_sample, v_sample, mu), m_sample])
        state = state_mee.copy()

        r_traj = []
        for i, (t0, t1) in enumerate(segments):
            lam = control_profile[i]
            S = np.concatenate([state, lam])
            t_eval = np.linspace(t0, t1, 20)
            Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0), [t0, t1], S, t_eval=t_eval)
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
    radii = scale * np.sqrt(np.maximum(vals, 0))

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.cos(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    xyz = np.stack([x, y, z], axis=-1)

    for i in range(xyz.shape[0]):
        for j in range(xyz.shape[1]):
            xyz[i, j] = vecs @ xyz[i, j] + mean

    ax.plot_surface(xyz[:, :, 0], xyz[:, :, 1], xyz[:, :, 2], rstride=1, cstride=1,
                    color=color, alpha=alpha, linewidth=0)

def evaluate_bundle(X, y, model, Wm, Wc, label, bundle_idx):
    X_bundle = X[X[:, -2] == bundle_idx]
    y_bundle = y[X[:, -2] == bundle_idx]
    y_pred_bundle = model.predict(X_bundle[:, :-2])
    time_vals = np.unique(X_bundle[:, 0])
    segments = list(zip(time_vals[:-1], time_vals[1:]))

    r_stack, control_profile, init_state = [], [], None
    sigma_indices = np.unique(X_bundle[:, -1]).astype(int)

    for sigma_idx in sigma_indices:
        r_sigma = []
        for i, (t0, t1) in enumerate(segments):
            row = X_bundle[(X_bundle[:, 0] == t0) & (X_bundle[:, -1] == sigma_idx)][0]
            lam = y_pred_bundle[(X_bundle[:, 0] == t0) & (X_bundle[:, -1] == sigma_idx)][0]
            S = np.concatenate([row[1:8], lam])
            t_eval = np.linspace(t0, t1, 20)
            Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0), [t0, t1], S, t_eval=t_eval)
            r, _ = mee2rv.mee2rv(*Sf.y[:6], mu)
            r_sigma.append(r)
            if sigma_idx == 0 and i == 0:
                init_state = row[1:8]
            if sigma_idx == 0:
                control_profile.append(lam)
        r_stack.append(np.vstack(r_sigma))

    r_stack = np.array(r_stack)
    mean_r = np.sum(Wm[:, None, None] * r_stack, axis=0)
    devs_r = r_stack - mean_r[None, :, :]
    P_pred = np.einsum("i,ijk,ijl->jkl", Wc, devs_r, devs_r)
    P_diag_pred = np.array([np.diag(np.diag(P)) for P in P_pred])
    inv_cov_pred = inv(P_diag_pred[-1])

    r_mc = monte_carlo_propagation(init_state, control_profile, segments)
    mean_mc_r = np.mean(r_mc, axis=0)
    devs_mc = r_mc - mean_mc_r[None, :, :]
    W_mc = np.ones(r_mc.shape[0]) / r_mc.shape[0]
    P_mc = np.einsum("i,ijk,ijl->jkl", W_mc, devs_mc, devs_mc)
    P_diag_mc = np.array([np.diag(np.diag(P)) for P in P_mc])
    inv_cov_mc = inv(P_diag_mc[-1])

    r_mc_final = r_mc[:, -1, :]
    mean_mc_final = np.mean(r_mc_final, axis=0)
    final_time = time_vals[-1]
    sigma0_rows = X_bundle[(X_bundle[:, 0] == final_time) & (X_bundle[:, -1] == 0)]
    appended = next((row for row in sigma0_rows if np.allclose(row[8:15], 0)), None)
    r_ref, _ = mee2rv.mee2rv(*appended[1:7].reshape(6, 1), mu)
    r_ref = r_ref.flatten()

    kl_div = compute_kl_divergence(mean_r[-1], P_diag_pred[-1], mean_mc_final, P_diag_mc[-1])
    mahalanobis_pred = mahalanobis(mean_r[-1], r_ref, inv_cov_pred)
    mahalanobis_mc = mahalanobis(mean_mc_final, r_ref, inv_cov_mc)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*mean_r.T, label='Predicted', color='blue')
    ax.plot(*mean_mc_r.T, label='MC Mean', color='black')
    ax.scatter(*r_ref, label='True σ₀', color='red', s=30)
    plot_3sigma_ellipsoid(ax, mean_r[-1], P_diag_pred[-1], color='blue', alpha=0.2)
    plot_3sigma_ellipsoid(ax, mean_mc_final, P_diag_mc[-1], color='black', alpha=0.2)
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_zlabel("Z [km]")
    ax.legend()
    plt.tight_layout()
    fname = f"traj_3sigma_{label}_bundle_{bundle_idx}.png"
    plt.savefig(fname)
    plt.close()

    return {
        "label": label,
        "bundle_idx": bundle_idx,
        "kl_divergence": kl_div,
        "mahalanobis_pred": mahalanobis_pred,
        "mahalanobis_mc": mahalanobis_mc
    }

def main():
    model = joblib.load("trained_model.pkl")
    Wm = joblib.load("Wm.pkl")
    Wc = joblib.load("Wc.pkl")
    data_max = joblib.load("segment_max.pkl")
    data_min = joblib.load("segment_min.pkl")

    results = []

    for label, data in [("max", data_max), ("min", data_min)]:
        bundle_ids = np.unique(data["X"][:, -2]).astype(int)
        for bundle_idx in tqdm(bundle_ids, desc=f"Evaluating {label}"):
            result = evaluate_bundle(data["X"], data["y"], model, Wm, Wc, label, bundle_idx)
            results.append(result)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("rf_eval_all_bundles.csv", index=False)
    print("[INFO] Saved: rf_eval_all_bundles.csv")

if __name__ == "__main__":
    main()
