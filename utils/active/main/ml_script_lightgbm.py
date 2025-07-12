
import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from lightgbm import LGBMRegressor
from scipy.integrate import solve_ivp
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    y = radii[1] * np.outer(np.sin(u), np.cos(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    ellipsoid = np.stack((x, y, z), axis=-1) @ eigvecs.T + mean
    ax.plot_surface(ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
                    rstride=1, cstride=1, color=color, alpha=alpha, linewidth=0)

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
        return 0.0
    log_det_term = logdet2 - logdet1
    kl_div = 0.5 * (trace_term + quadratic_term - k + log_det_term)
    return max(kl_div, 0.0)

def propagate_sigma(row, lam_vec, segments):
    r_sigma = []
    for i, (t0, t1) in enumerate(segments):
        S = np.concatenate([row[1:8], lam_vec[i]])
        Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                       [t0, t1], S, t_eval=np.linspace(t0, t1, 20))
        r, v = mee2rv.mee2rv(*Sf.y[:6], mu)
        r_sigma.append(r)
    return np.vstack(r_sigma)

def monte_carlo(init_state, control_profile, segments, n=300):
    r0, v0 = mee2rv.mee2rv(*[np.array([val]) for val in init_state[:6]], mu)
    mean = np.hstack([r0.flatten(), v0.flatten(), init_state[6]])
    samples = np.random.multivariate_normal(mean, P_init, size=n)

    def propagate(sample):
        r, v, m = sample[:3], sample[3:6], sample[6]
        S = np.hstack([rv2mee.rv2mee(r[None], v[None], mu), m])
        traj = []
        for lam, (t0, t1) in zip(control_profile, segments):
            Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                           [t0, t1], np.concatenate([S, lam]), t_eval=np.linspace(t0, t1, 20))
            r, _ = mee2rv.mee2rv(*Sf.y[:6], mu)
            traj.append(r)
            S = Sf.y[:7, -1]
        return np.vstack(traj)

    with ProcessPoolExecutor() as ex:
        results = list(tqdm(ex.map(propagate, samples), total=n, desc="Monte Carlo"))
    return np.array(results)

def evaluate_segment(X_seg, y_seg, model, Wm, label):
    t_vals = np.unique(X_seg[:, 0])
    segments = list(zip(t_vals[:-1], t_vals[1:]))
    y_pred = model.predict(X_seg[:, :-2])
    sigma_ids = np.unique(X_seg[:, -1]).astype(int)
    trajs = []

    for sigma in tqdm(sigma_ids, desc=f"{label}: Model Propagation"):
        rows = X_seg[X_seg[:, -1] == sigma]
        lams = y_pred[X_seg[:, -1] == sigma]
        trajs.append(propagate_sigma(rows[0], lams, segments))

    r_pred_stack = np.array(trajs)
    mean_r = np.sum(Wm[:, None, None] * r_pred_stack, axis=0)
    r_pred = r_pred_stack[0]

    final_time = t_vals[-1]
    ref_row = X_seg[(X_seg[:, 0] == final_time) & (X_seg[:, -1] == 0)][0]
    r_ref, v_ref = mee2rv.mee2rv(*ref_row[1:7].reshape(6, 1), mu)
    r_ref, v_ref = r_ref.flatten(), v_ref.flatten()

    init_state = X_seg[(X_seg[:, 0] == t_vals[0]) & (X_seg[:, -1] == 0)][0][1:8]
    control_profile = [y_pred[(X_seg[:, 0] == t) & (X_seg[:, -1] == 0)][0] for t in t_vals[:-1]]
    r_mc = monte_carlo(init_state, control_profile, segments)

    mean_mc = np.mean(r_mc, axis=0)
    dev_mc = r_mc - mean_mc[None]
    P_mc = np.einsum("ijk,ijl->jkl", dev_mc, dev_mc) / r_mc.shape[0]
    P_pred = np.einsum("i,ijk,ijl->jkl", Wm, r_pred_stack - mean_r[None], r_pred_stack - mean_r[None])

    kl_val = compute_kl_divergence(mean_r[-1], P_pred[-1], mean_mc[-1], P_mc[-1])

    mahal_pred = mahalanobis(r_pred[-1], r_ref, inv(np.diag(np.diag(P_pred[-1]))))
    mahal_mc_mean = mahalanobis(mean_mc[-1], r_ref, inv(np.diag(np.diag(P_mc[-1]))))
    mahal_sigmas = []
    for traj in r_pred_stack[1:]:
        mahal_sigmas.append(mahalanobis(traj[-1], r_ref, inv(np.diag(np.diag(P_pred[-1])))))

    metrics = {
        "segment": label,
        "final_position_error": np.linalg.norm(r_pred[-1] - r_ref),
        "cosine_similarity": cosine_similarity(control_profile[0], y_seg[(X_seg[:, 0] == t_vals[0]) & (X_seg[:, -1] == 0)][0]),
        "mc_vs_pred_mse": np.mean((r_pred - mean_mc) ** 2),
        "control_MSE": np.mean((np.array(control_profile) - y_seg[X_seg[:, -1] == 0]) ** 2),
        "mahalanobis_pred": mahal_pred,
        "mahalanobis_mc_mean": mahal_mc_mean,
        "kl_divergence": kl_val,
        "trace_diff": np.trace(np.abs(P_mc[-1] - P_pred[-1])),
        "x_mse": np.mean((r_ref[0] - r_pred[:, 0])**2),
        "y_mse": np.mean((r_ref[1] - r_pred[:, 1])**2),
        "z_mse": np.mean((r_ref[2] - r_pred[:, 2])**2),
        "mahalanobis_gt3": sum(d > 3 for d in mahal_sigmas)
    }

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r_ref[:, 0], r_ref[:, 1], r_ref[:, 2], color='black', lw=2.2, label='Actual σ₀')
    ax.plot(r_pred[:, 0], r_pred[:, 1], r_pred[:, 2], color='gray', lw=1.5, linestyle='--', label='Predicted σ₀')
    for i in range(0, r_mc.shape[0], 10):
        ax.plot(r_mc[i, :, 0], r_mc[i, :, 1], r_mc[i, :, 2], color='dimgray', alpha=0.3, lw=0.7, linestyle=':')
    plot_3sigma_ellipsoid(ax, mean_r[-1], P_pred[-1])
    plot_3sigma_ellipsoid(ax, mean_mc[-1], P_mc[-1])
    ax.set_xlabel('X [km]'); ax.set_ylabel('Y [km]'); ax.set_zlabel('Z [km]')
    set_axes_equal(ax)
    ax.legend()
    plt.savefig(f"ml_eval_segment_{label}/comparison.pdf", dpi=600, bbox_inches='tight')
    plt.close()

    return metrics

# === Main workflow ===
X_all, y_all, Wm, Wc = [], [], None, None
for file in sorted(glob.glob("baseline_stride_1/batch_*/data.pkl")):
    d = joblib.load(file)
    X_all.append(d["X"])
    y_all.append(d["y"])
    Wm, Wc = d["Wm"], d["Wc"]

X, y = np.vstack(X_all), np.vstack(y_all)

print("Training LightGBM model...")
model = LGBMRegressor(n_estimators=300, max_depth=10, learning_rate=0.03)
model.fit(X[:, :-2], y)

times = np.loadtxt("stride_4000min/bundle_segment_widths.txt", skiprows=1)
max_t, min_t = times[np.argmax(times[:, 1]), 0], times[np.argmin(times[:, 1]), 0]

X_max, y_max = X[(X[:, 0] == max_t) | (X[:, 0] == max_t + 1)], y[(X[:, 0] == max_t) | (X[:, 0] == max_t + 1)]
X_min, y_min = X[(X[:, 0] == min_t) | (X[:, 0] == min_t + 1)], y[(X[:, 0] == min_t) | (X[:, 0] == min_t + 1)]

os.makedirs("ml_eval_segment_max", exist_ok=True)
os.makedirs("ml_eval_segment_min", exist_ok=True)

results = [
    evaluate_segment(X_max, y_max, model, Wm, "max"),
    evaluate_segment(X_min, y_min, model, Wm, "min")
]

pd.DataFrame(results).to_csv("ml_eval_segment_metrics.csv", index=False)
print("Saved: ml_eval_segment_metrics.csv")
