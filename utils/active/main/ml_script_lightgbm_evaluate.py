import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from numpy.linalg import inv
from scipy.integrate import solve_ivp
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
from odefunc import odefunc

plt.rcParams.update({'font.size': 8})
plt.rcParams['axes.grid'] = True

# === Covariance ===
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
    sigma2_inv = inv(sigma2)
    trace_term = np.trace(sigma2_inv @ sigma1)
    diff = mu2 - mu1
    quad_term = diff.T @ sigma2_inv @ diff
    sign1, logdet1 = np.linalg.slogdet(sigma1)
    sign2, logdet2 = np.linalg.slogdet(sigma2)
    if sign1 <= 0 or sign2 <= 0:
        return 0.0
    return 0.5 * (trace_term + quad_term - k + (logdet2 - logdet1))

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.2, scale=3.0):
    cov = 0.5 * (cov + cov.T) + np.eye(3) * 1e-10
    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.any(eigvals <= 0): return
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    radii = scale * np.sqrt(eigvals)
    u, v = np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 40)
    x = radii[0]*np.outer(np.cos(u), np.sin(v))
    y = radii[1]*np.outer(np.sin(u), np.sin(v))
    z = radii[2]*np.outer(np.ones_like(u), np.cos(v))
    ellipsoid = np.stack((x, y, z), axis=-1) @ eigvecs.T + mean
    ax.plot_surface(ellipsoid[...,0], ellipsoid[...,1], ellipsoid[...,2],
                    rstride=1, cstride=1, color=color, alpha=alpha, linewidth=0)
    ax.plot_wireframe(ellipsoid[...,0], ellipsoid[...,1], ellipsoid[...,2],
                      rstride=5, cstride=5, color='k', alpha=0.2, linewidth=0.3)

def set_axes_equal(ax):
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    ranges = [abs(l[1] - l[0]) for l in [xlim, ylim, zlim]]
    centers = [np.mean(l) for l in [xlim, ylim, zlim]]
    max_range = max(ranges) / 2
    ax.set_xlim([centers[0]-max_range, centers[0]+max_range])
    ax.set_ylim([centers[1]-max_range, centers[1]+max_range])
    ax.set_zlim([centers[2]-max_range, centers[2]+max_range])
    ax.set_box_aspect([1.25, 1, 0.75])

def mc_worker(args):
    sample, init_state_mee, control_profile, segments, mu, F, c, m0, g0 = args
    r_sample, v_sample = sample[:3], sample[3:6]
    m_sample = sample[6]
    r_sample = r_sample.reshape(1, 3)
    v_sample = v_sample.reshape(1, 3)
    state_mee = np.hstack([rv2mee(r_sample, v_sample, mu), m_sample])
    state = state_mee.copy()
    r_traj = []
    for i, (t0, t1) in enumerate(segments):
        lam = control_profile[i]
        S = np.concatenate([state, lam])
        sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F, c, m0, g0), [t0, t1], S,
                        t_eval=np.linspace(t0, t1, 20))
        r, _ = mee2rv(*sol.y[:6], mu)
        r_traj.append(r)
        state = sol.y[:7, -1]
    return np.vstack(r_traj)

def monte_carlo_propagation_parallel(init_state_mee, control_profile, segments, mu, F, c, m0, g0, num_samples=1000, num_workers=None):
    r0_eci, v0_eci = mee2rv(*[np.array([val]) for val in init_state_mee[:6]], mu)
    m0_val = init_state_mee[6]
    mean_eci = np.hstack([r0_eci.flatten(), v0_eci.flatten(), m0_val])
    samples = np.random.multivariate_normal(mean_eci, P_init, size=num_samples)
    args_list = [(s, init_state_mee, control_profile, segments, mu, F, c, m0, g0) for s in samples]
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        r_mc = list(tqdm(ex.map(mc_worker, args_list), total=len(args_list), desc="Monte Carlo"))
    return np.array(r_mc)

def evaluate_bundle(args):
    b, X, y, model, Wm, Wc, label = args
    mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81
    time_vals = np.unique(X[:, 0])
    segments = list(zip(time_vals[:-1], time_vals[1:]))
    sigma_indices = np.unique(X[:, -1]).astype(int)
    y_pred = model.predict(X[:, :-2])
    control_profile, r_stack, v_stack = [], [], []
    init_state = None

    for s in sigma_indices:
        r_sigma, v_sigma = [], []
        for i, (t0, t1) in enumerate(segments):
            row = X[(X[:, 0] == t0) & (X[:, -1] == s)][0]
            lam = y_pred[(X[:, 0] == t0) & (X[:, -1] == s)][0]
            S = np.concatenate([row[1:8], lam])
            sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F, c, m0, g0), [t0, t1], S,
                            t_eval=np.linspace(t0, t1, 20))
            r, v = mee2rv(*sol.y[:6], mu)
            r_sigma.append(r)
            v_sigma.append(v)
            if s == 0 and i == 0:
                init_state = row[1:8]
            if s == 0:
                control_profile.append(lam)
        r_stack.append(np.vstack(r_sigma))
        v_stack.append(np.vstack(v_sigma))

    r_stack = np.array(r_stack)
    mean_r = np.sum(Wm[:, None, None] * r_stack, axis=0)
    devs_r = r_stack - mean_r[None, :, :]
    P_diag_r = np.einsum("i,ijk,ijl->jkl", Wc, devs_r, devs_r)

    r_pred = r_stack[0]
    r_mc = monte_carlo_propagation_parallel(init_state, control_profile, segments, mu, F, c, m0, g0)

    mean_mc = np.mean(r_mc, axis=0)
    cov_mc = np.cov(r_mc[:, -1, :].T)
    kl = compute_kl_divergence(mean_r[-1], P_diag_r[-1], mean_mc[-1], cov_mc)

    final_time = time_vals[-1]
    row = X[(X[:, 0] == final_time) & (X[:, -1] == 0) & np.allclose(X[:, 8:15], 0, atol=1e-10)][0]
    r_ref, v_ref = mee2rv(*row[1:7].reshape(6, 1), mu)
    r_ref, v_ref = r_ref.flatten(), v_ref.flatten()

    out_path = f"outputs/segment_{label}_bundle_{b}/sigma_mc_comparison.pdf"
    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(r_stack)):
        r = r_stack[i]
        ax.plot(r[:, 0], r[:, 1], r[:, 2], color='black' if i==0 else 'gray',
                linestyle='-' if i==0 else '--', lw=2.2 if i==0 else 0.8, alpha=1.0)
    for i in range(0, len(r_mc), 10):
        ax.plot(r_mc[i][:,0], r_mc[i][:,1], r_mc[i][:,2], linestyle=':', color='dimgray', lw=0.8, alpha=0.4)
    plot_3sigma_ellipsoid(ax, mean_r[0], P_diag_r[0][:3,:3])
    plot_3sigma_ellipsoid(ax, mean_r[-1], P_diag_r[-1][:3,:3])
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
    set_axes_equal(ax)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()

    return {
        "segment": label, "bundle_idx": b,
        "final_pos_error": np.linalg.norm(r_pred[-1] - r_ref),
        "kl_divergence": kl,
        "trace_diff": np.trace(P_diag_r[-1]),
        "x_mse": np.mean((r_ref[0] - r_pred[:, 0])**2),
        "y_mse": np.mean((r_ref[1] - r_pred[:, 1])**2),
        "z_mse": np.mean((r_ref[2] - r_pred[:, 2])**2)
    }

def main():
    model = joblib.load("trained_model.pkl")
    Wm = joblib.load("Wm.pkl")
    Wc = joblib.load("Wc.pkl")
    segment_max = joblib.load("segment_max.pkl")
    segment_min = joblib.load("segment_min.pkl")

    results = []
    for label, data in [("max", segment_max), ("min", segment_min)]:
        args_list = []
        for b in np.unique(data["X"][:, -2]).astype(int):
            X_b = data["X"][data["X"][:, -2] == b]
            y_b = data["y"][data["X"][:, -2] == b]
            args_list.append((b, X_b, y_b, model, Wm, Wc, label))

        with ProcessPoolExecutor() as ex:
            results += list(tqdm(ex.map(evaluate_bundle, args_list), total=len(args_list), desc=f"Evaluating {label}"))

    df = pd.DataFrame(results)
    df.to_csv("ml_lightgbm_evaluation_results.csv", index=False)
    print("Saved: ml_lightgbm_evaluation_results.csv")

if __name__ == "__main__":
    main()
