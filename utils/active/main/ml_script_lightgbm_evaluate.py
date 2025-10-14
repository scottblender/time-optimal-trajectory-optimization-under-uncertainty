# ml_script_lightgbm_evaluate.py
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from tqdm import tqdm

# === Setup ===
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['LGBM_LOGLEVEL'] = '2'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
from odefunc import odefunc

# === Constants ===
mu, F, c, m0, g0 = 27.8996, 0.33, 4.4246, 4000, 9.81

# === Metric Utilities ===
def compute_kl_divergence(mu1, sigma1, mu2, sigma2):
    """Computes the KL divergence between two multivariate normal distributions."""
    k = mu1.shape[0]
    # Add regularization for stability
    sigma1 = sigma1 + np.eye(k) * 1e-9
    sigma2 = sigma2 + np.eye(k) * 1e-9
    try:
        sigma2_inv = inv(sigma2)
    except np.linalg.LinAlgError:
        return np.nan # Return NaN if covariance is singular
        
    trace_term = np.trace(sigma2_inv @ sigma1)
    diff = mu2 - mu1
    quad_term = diff.T @ sigma2_inv @ diff
    
    sign1, logdet1 = np.linalg.slogdet(sigma1)
    sign2, logdet2 = np.linalg.slogdet(sigma2)
    
    if sign1 <= 0 or sign2 <= 0:
        return np.nan
        
    return 0.5 * (trace_term + quad_term - k + logdet2 - logdet1)

def compute_thrust_direction(mee, lam):
    """Computes the unit vector for the thrust direction from costates."""
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
    norm_mat = np.linalg.norm(mat)
    return (mat.flatten() / norm_mat) if norm_mat > 0 else np.full(3, np.nan)

def evaluate_segment(X, y, model, label, bundle_idx):
    """
    Evaluates a single data segment by propagating trajectories and computing metrics.
    NO PLOTTING.
    """
    X_b = X[X[:, -2] == bundle_idx]
    y_b = y[X[:, -2] == bundle_idx]
    times = np.unique(X_b[:, 0])
    if len(times) < 3: # Need at least start, middle, end points for propagation
        return None
    t0, t1 = times[1], times[2]

    sigma_indices = np.unique(X_b[:, -1]).astype(int)
    r_pred, v_pred, m_pred = [], [], []
    r_actual, v_actual, m_actual = [], [], []
    
    feature_cols = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']

    def get_row(t, sigma_idx, is_y=False):
        mask = np.isclose(X_b[:, 0], t) & (X_b[:, -1] == sigma_idx)
        rows = y_b[mask] if is_y else X_b[mask]
        return rows[0] if len(rows) > 0 else None

    # Propagate trajectories for all sigma points
    for sigma_idx in sigma_indices:
        x_row = get_row(t0, sigma_idx)
        if x_row is None: continue

        # Predicted trajectory
        x_df = pd.DataFrame([x_row[:-2]], columns=feature_cols)
        lam_pred = model.predict(x_df)[0]
        S_pred = np.concatenate([x_row[1:8], lam_pred])
        sol_pred = solve_ivp(lambda t, x: odefunc(t, x, mu, F, c, m0, g0),
                             [t0, t1], S_pred, t_eval=[t1])
        r_p, v_p = mee2rv(*sol_pred.y[:6], mu)
        r_pred.append(r_p.flatten())
        v_pred.append(v_p.flatten())
        m_pred.append(sol_pred.y[6, -1])

        # Actual trajectory
        lam_actual = get_row(t0, sigma_idx, is_y=True)
        if lam_actual is None: continue
        S_actual = np.concatenate([x_row[1:8], lam_actual])
        sol_actual = solve_ivp(lambda t, x: odefunc(t, x, mu, F, c, m0, g0),
                               [t0, t1], S_actual, t_eval=[t1])
        r_a, v_a = mee2rv(*sol_actual.y[:6], mu)
        r_actual.append(r_a.flatten())
        v_actual.append(v_a.flatten())
        m_actual.append(sol_actual.y[6, -1])

        if sigma_idx == 0:
            mee0 = x_row[1:8]
            del_t_pred = compute_thrust_direction(mee0, lam_pred)
            del_t_true = compute_thrust_direction(mee0, lam_actual)

    if not r_pred: # If no points were processed, exit
        return None

    r_pred, v_pred, m_pred = np.array(r_pred), np.array(v_pred), np.array(m_pred)
    r_actual, v_actual, m_actual = np.array(r_actual), np.array(v_actual), np.array(m_actual)
    
    # Get reference state from the dataset at the end time
    ref_mask = np.isclose(X_b[:, 0], t1) & (X_b[:, -1] == 0)
    if not np.any(ref_mask):
        return None
    row_sigma0_end = X_b[ref_mask][0]
    mee_ref = row_sigma0_end[1:8]
    r_ref, v_ref = mee2rv(*mee_ref[:6], mu)
    r_ref, v_ref = r_ref.flatten(), v_ref.flatten()
    m_ref = mee_ref[6]

    # Compute distribution statistics using standard numpy
    mu_pred, cov_pred = np.mean(r_pred, axis=0), np.cov(r_pred.T)
    mu_act, cov_act = np.mean(r_actual, axis=0), np.cov(r_actual.T)
    
    # Compute metrics
    maha_pred = np.mean([mahalanobis(r, mu_pred, inv(cov_pred + 1e-9*np.eye(3))) for r in r_pred])
    maha_act = np.mean([mahalanobis(r, mu_act, inv(cov_act + 1e-9*np.eye(3))) for r in r_actual])
    dot = np.clip(np.dot(del_t_pred, del_t_true), -1, 1)
    control_angle_deg = np.degrees(np.arccos(dot)) if not (np.isnan(del_t_pred).any() or np.isnan(del_t_true).any()) else np.nan

    return {
        "label": label,
        "bundle_idx": bundle_idx,
        "pos_mse_pred": np.mean(np.sum((r_pred - r_ref) ** 2, axis=1)),
        "pos_mse_actual": np.mean(np.sum((r_actual - r_ref) ** 2, axis=1)),
        "vel_mse_pred": np.mean(np.sum((v_pred - v_ref) ** 2, axis=1)),
        "vel_mse_actual": np.mean(np.sum((v_actual - v_ref) ** 2, axis=1)),
        "mass_mse_pred": np.mean((m_pred - m_ref) ** 2),
        "mass_mse_actual": np.mean((m_actual - m_ref) ** 2),
        "kl_pred_vs_actual": compute_kl_divergence(mu_pred, cov_pred, mu_act, cov_act),
        "mahal_pred_mean": maha_pred,
        "mahal_actual_mean": maha_act,
        "control_angle_deg": control_angle_deg,
    }

# === Main ===
def main():
    try:
        model_min = joblib.load("trained_model_min_optimal.pkl")
        model_max = joblib.load("trained_model_max_optimal.pkl")
        data_max = joblib.load("segment_max.pkl")
        data_min = joblib.load("segment_min.pkl")
    except FileNotFoundError as e:
        print(f"[ERROR] A required data file is missing: {e}. Please ensure all input and model files are present.")
        return

    os.makedirs("eval_lightgbm_outputs", exist_ok=True)
    models = {"min": model_min, "max": model_max}
    results = []

    for label, data in [("min", data_min), ("max", data_max)]:
        model = models[label]
        bundle_indices = np.unique(data["X"][:, -2]).astype(int)
        for bundle_idx in tqdm(bundle_indices, desc=f"Evaluating '{label}' bundles"):
            res = evaluate_segment(data["X"], data["y"], model, label, bundle_idx)
            if res:
                results.append(res)
    
    if not results:
        print("\n[WARN] No results were generated. Check input data and script logic.")
        return

    df = pd.DataFrame(results)
    output_path = "eval_lightgbm_outputs/metrics_summary_lgbm.csv"
    df.to_csv(output_path, index=False)
    
    print("\n--- Evaluation Summary ---")
    print(df.to_string())
    print(f"\n[SUCCESS] Evaluation complete. Summary saved to {output_path}")

if __name__ == "__main__":
    main()