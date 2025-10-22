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

# === Feature Definitions (Consistent with other scripts) ===
features_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
features_corrective = features_optimal + [
    'delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz',
    'delta_m', 't_go'
]

# === Metric Utilities ===
def compute_kl_divergence(mu1, sigma1, mu2, sigma2):
    """Computes the KL divergence between two multivariate normal distributions."""
    k = mu1.shape[0]
    sigma1 = sigma1 + np.eye(k) * 1e-9
    sigma2 = sigma2 + np.eye(k) * 1e-9
    try:
        sigma2_inv = inv(sigma2)
    except np.linalg.LinAlgError:
        return np.nan
    trace_term = np.trace(sigma2_inv @ sigma1)
    diff = mu2 - mu1
    quad_term = diff.T @ sigma2_inv @ diff
    sign1, logdet1 = np.linalg.slogdet(sigma1)
    sign2, logdet2 = np.linalg.slogdet(sigma2)
    if sign1 <= 0 or sign2 <= 0:
        return np.nan
    return 0.5 * (trace_term + quad_term - k + logdet2 - logdet1)

def compute_thrust_direction(mee, lam, mu):
    """Computes the unit vector for the thrust direction from costates."""
    # Check if input arrays have expected length
    if len(mee) < 7 or len(lam) < 7:
        # print(f"[Debug] Invalid input shape. MEE: {mee.shape}, Lam: {lam.shape}")
        return np.full(3, np.nan)
        
    p, f, g, h, k, L = mee[:6] # Only use first 6 MEEs
    lam_p, lam_f, lam_g, lam_h, lam_k, lam_L = lam[:6] # Only use first 6 costates

    lam_matrix = np.array([[lam_p, lam_f, lam_g, lam_h, lam_k, lam_L]]).T
    SinL, CosL = np.sin(L), np.cos(L)
    w = 1 + f * CosL + g * SinL

    if np.isclose(w, 0, atol=1e-10):
        return np.full(3, np.nan)

    s = 1 + h**2 + k**2
    # Ensure p is non-negative before sqrt
    if p < 0:
        # print(f"[Debug] Negative p value encountered: {p}")
        p = 0 # Or handle appropriately, e.g., return NaN
        
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

def evaluate_segment(df_b, y_opt_b, y_corr_b, model, label, bundle_idx, model_type, mu, F, c, m0, g0):
    """
    Evaluates a single data segment by propagating trajectories and computing metrics.
    df_b, y_opt_b, y_corr_b are now guaranteed to have matching lengths and indices.
    """
    times = np.unique(df_b['t'])
    if len(times) < 3:
        # print(f"[Debug] Bundle {bundle_idx} skipped: Found only {len(times)} unique times.")
        return None
    # Use first and second unique times for propagation segment
    t0, t1 = times[0], times[1]

    sigma_indices = np.unique(df_b['sigma_idx']).astype(int)
    r_pred, v_pred, m_pred = [], [], []
    r_actual, v_actual, m_actual = [], [], []

    del_t_pred, del_t_true = np.full(3, np.nan), np.full(3, np.nan)

    # Convert y-data to DataFrames WITH RESET INDEX for easier lookup by integer location
    # This aligns the indices with the boolean mask applied earlier
    df_y_opt = pd.DataFrame(y_opt_b).reset_index(drop=True)
    df_y_corr = pd.DataFrame(y_corr_b).reset_index(drop=True)
    df_b = df_b.reset_index(drop=True) # Reset index of the filtered DataFrame too

    def get_rows_at_time(t, sigma_idx):
        # Use boolean indexing on the reset-index DataFrame
        mask = np.isclose(df_b['t'], t) & (df_b['sigma_idx'] == sigma_idx)
        if not mask.any():
            return None, None, None, None

        # Get the integer location (iloc) of the first match
        iloc_index = df_b.index[mask][0]
        x_row_series = df_b.iloc[iloc_index]
        y_opt_row = df_y_opt.iloc[iloc_index].values
        y_corr_row = df_y_corr.iloc[iloc_index].values
        
        # Ensure y_opt_row and y_corr_row have the correct shape for subtraction
        if y_opt_row.shape != y_corr_row.shape:
             print(f"[Debug] Shape mismatch! y_opt: {y_opt_row.shape}, y_corr: {y_corr_row.shape} at index {iloc_index}")
             return None, None, None, None

        lam_nom_row = y_opt_row - y_corr_row # Reconstruct nominal lambda

        return x_row_series, y_opt_row, lam_nom_row, iloc_index # Return iloc index

    # Propagate trajectories for all sigma points
    for sigma_idx in sigma_indices:
        x_row, lam_actual, lam_nom, row_iloc = get_rows_at_time(t0, sigma_idx)
        if x_row is None:
            # print(f"[Debug] Bundle {bundle_idx}, Sigma {sigma_idx}: No data found at t={t0}")
            continue

        # --- Predict Lambda based on model type ---
        try:
            if model_type == 'optimal':
                features_for_pred = x_row[features_optimal].values.reshape(1, -1)
                x_df = pd.DataFrame(features_for_pred, columns=features_optimal)
                lam_pred = model.predict(x_df)[0]
            else: # corrective
                features_for_pred = x_row[features_corrective].values.reshape(1, -1)
                x_df = pd.DataFrame(features_for_pred, columns=features_corrective)
                lam_correction = model.predict(x_df)[0]
                lam_pred = lam_nom + lam_correction # Apply correction
        except Exception as e:
            print(f"[ERROR] Prediction failed for Bundle {bundle_idx}, Sigma {sigma_idx}, Time {t0}: {e}")
            print(f"Features shape: {features_for_pred.shape}")
            continue # Skip this sigma point if prediction fails

        # Predicted trajectory
        S_pred = np.concatenate([x_row[1:8].values, lam_pred])
        try:
            sol_pred = solve_ivp(lambda t, x: odefunc(t, x, mu, F, c, m0, g0),
                                 [t0, t1], S_pred, t_eval=[t1], rtol=1e-6, atol=1e-9)
            if not sol_pred.success: raise ValueError("Pred integration failed")
            r_p, v_p = mee2rv(*sol_pred.y[:6, -1], mu) # Use state at t1
        except Exception as e:
             print(f"[WARN] Predicted propagation failed for Bundle {bundle_idx}, Sigma {sigma_idx}: {e}")
             continue # Skip if propagation fails
             
        r_pred.append(r_p.flatten())
        v_pred.append(v_p.flatten())
        m_pred.append(sol_pred.y[6, -1])

        # Actual trajectory
        S_actual = np.concatenate([x_row[1:8].values, lam_actual])
        try:
            sol_actual = solve_ivp(lambda t, x: odefunc(t, x, mu, F, c, m0, g0),
                                   [t0, t1], S_actual, t_eval=[t1], rtol=1e-6, atol=1e-9)
            if not sol_actual.success: raise ValueError("Actual integration failed")
            r_a, v_a = mee2rv(*sol_actual.y[:6, -1], mu) # Use state at t1
        except Exception as e:
             print(f"[WARN] Actual propagation failed for Bundle {bundle_idx}, Sigma {sigma_idx}: {e}")
             # Don't append if actual fails, to keep lists aligned
             r_pred.pop()
             v_pred.pop()
             m_pred.pop()
             continue

        r_actual.append(r_a.flatten())
        v_actual.append(v_a.flatten())
        m_actual.append(sol_actual.y[6, -1])

        if sigma_idx == 0:
            mee0 = x_row[1:8].values
            del_t_pred = compute_thrust_direction(mee0, lam_pred, mu)
            del_t_true = compute_thrust_direction(mee0, lam_actual, mu)

    # Check if we successfully processed any sigma points
    if not r_pred or not r_actual or len(r_pred) != len(r_actual):
         print(f"[WARN] Bundle {bundle_idx}: Not enough valid sigma points propagated. Pred: {len(r_pred)}, Act: {len(r_actual)}. Skipping.")
         return None

    r_pred, v_pred, m_pred = np.array(r_pred), np.array(v_pred), np.array(m_pred)
    r_actual, v_actual, m_actual = np.array(r_actual), np.array(v_actual), np.array(m_actual)

    # Get reference state from the dataset at the end time t1
    ref_row, _, _, _ = get_rows_at_time(t1, 0)
    if ref_row is None:
        print(f"[WARN] Bundle {bundle_idx}: Reference state (sigma 0) not found at t={t1}. Skipping.")
        return None

    mee_ref = ref_row[1:8].values
    r_ref, v_ref = mee2rv(*mee_ref[:6], mu)
    r_ref, v_ref = r_ref.flatten(), v_ref.flatten()
    m_ref = mee_ref[6]

    # Compute distribution statistics
    mu_pred, cov_pred = np.mean(r_pred, axis=0), np.cov(r_pred.T)
    mu_act, cov_act = np.mean(r_actual, axis=0), np.cov(r_actual.T)

    # Compute metrics
    # Add checks for covariance matrix invertibility before mahalanobis
    try:
        inv_cov_pred = inv(cov_pred + 1e-9*np.eye(3))
        maha_pred = np.mean([mahalanobis(r, mu_pred, inv_cov_pred) for r in r_pred])
    except np.linalg.LinAlgError:
        maha_pred = np.nan
        
    try:
        inv_cov_act = inv(cov_act + 1e-9*np.eye(3))
        maha_act = np.mean([mahalanobis(r, mu_act, inv_cov_act) for r in r_actual])
    except np.linalg.LinAlgError:
        maha_act = np.nan
        
    # Check for NaN in thrust vectors before dot product
    if np.isnan(del_t_pred).any() or np.isnan(del_t_true).any():
        control_angle_deg = np.nan
    else:
        dot = np.clip(np.dot(del_t_pred, del_t_true), -1, 1)
        control_angle_deg = np.degrees(np.arccos(dot))

    kl_div = compute_kl_divergence(mu_pred, cov_pred, mu_act, cov_act)

    return {
        "label": label,
        "bundle_idx": bundle_idx,
        "model_type": model_type,
        "pos_mse_pred": np.mean(np.sum((r_pred - r_ref) ** 2, axis=1)),
        "pos_mse_actual": np.mean(np.sum((r_actual - r_ref) ** 2, axis=1)),
        "vel_mse_pred": np.mean(np.sum((v_pred - v_ref) ** 2, axis=1)),
        "vel_mse_actual": np.mean(np.sum((v_actual - v_ref) ** 2, axis=1)),
        "mass_mse_pred": np.mean((m_pred - m_ref) ** 2),
        "mass_mse_actual": np.mean((m_actual - m_ref) ** 2),
        "kl_pred_vs_actual": kl_div,
        "mahal_pred_mean": maha_pred,
        "mahal_actual_mean": maha_act,
        "del_t_pred_x": del_t_pred[0] if not np.isnan(del_t_pred).any() else np.nan,
        "del_t_pred_y": del_t_pred[1] if not np.isnan(del_t_pred).any() else np.nan,
        "del_t_pred_z": del_t_pred[2] if not np.isnan(del_t_pred).any() else np.nan,
        "del_t_true_x": del_t_true[0] if not np.isnan(del_t_true).any() else np.nan,
        "del_t_true_y": del_t_true[1] if not np.isnan(del_t_true).any() else np.nan,
        "del_t_true_z": del_t_true[2] if not np.isnan(del_t_true).any() else np.nan,
        "control_angle_deg": control_angle_deg,
    }

# === Main ===
def main():
    try:
        # Load all models
        models = {
            "min_optimal": joblib.load("trained_model_min_optimal.pkl"),
            "max_optimal": joblib.load("trained_model_max_optimal.pkl"),
            "min_corrective": joblib.load("trained_model_min_corrective.pkl"),
            "max_corrective": joblib.load("trained_model_max_corrective.pkl")
        }
        # Load the single processed data file
        processed_data = joblib.load("processed_training_data.pkl")
        # Load nominal data to get constants
        nominal_data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
        mu, F, c, m0, g0 = nominal_data["mu"], nominal_data["F"], nominal_data["c"], nominal_data["m0"], nominal_data["g0"]

    except FileNotFoundError as e:
        print(f"[ERROR] A required data file is missing: {e}. Please ensure all input and model files are present.")
        return

    os.makedirs("eval_lightgbm_outputs", exist_ok=True)
    results = []

    # Define all evaluation runs
    eval_runs = [
        {"label": "min", "model_type": "optimal", "model": models["min_optimal"]},
        {"label": "min", "model_type": "corrective", "model": models["min_corrective"]},
        {"label": "max", "model_type": "optimal", "model": models["max_optimal"]},
        {"label": "max", "model_type": "corrective", "model": models["max_corrective"]}
    ]

    for run in eval_runs:
        label = run["label"]
        model_type = run["model_type"]
        model = run["model"]

        # Get the correct data slice
        data_slice = processed_data[f"{label}_data"]
        df = data_slice["df"]
        y_opt = data_slice["y_opt"]
        y_corr = data_slice["y_corr"]

        bundle_indices = np.unique(df['bundle_idx']).astype(int)

        for bundle_idx in tqdm(bundle_indices, desc=f"Evaluating '{label}' bundles ({model_type})"):
            # --- CORRECTED: Use boolean mask for consistent filtering ---
            bundle_mask = (df['bundle_idx'] == bundle_idx)
            df_b = df[bundle_mask]
            
            # Ensure y_opt and y_corr are numpy arrays before filtering
            y_opt_np = np.asarray(y_opt)
            y_corr_np = np.asarray(y_corr)
            
            # Filter y arrays using the same boolean mask based on the original df index
            # This requires y arrays to have the same length as the original df before filtering
            
            # Check if lengths match before attempting to filter y
            if len(y_opt_np) != len(df) or len(y_corr_np) != len(df):
                 print(f"[ERROR] Length mismatch! df: {len(df)}, y_opt: {len(y_opt_np)}, y_corr: {len(y_corr_np)}. Cannot filter y arrays correctly.")
                 continue # Skip this bundle or handle error

            # We need the indices from the original df that correspond to the mask
            original_indices = df.index[bundle_mask]

            # Use these original indices (if they are integer positions) or convert to integer positions
            # Assuming y_opt_np and y_corr_np align with the original df's implicit 0-based index
            # If df.index are original large numbers, we need a way to map them back
            
            # SAFER APPROACH: Pass integer indices corresponding to the mask
            iloc_indices = np.where(bundle_mask)[0]

            y_opt_b = y_opt_np[iloc_indices]
            y_corr_b = y_corr_np[iloc_indices]
            # --- END CORRECTION ---

            res = evaluate_segment(
                df_b, y_opt_b, y_corr_b, model,
                label, bundle_idx, model_type,
                mu, F, c, m0, g0
            )
            if res:
                results.append(res)

    if not results:
        print("\n[WARN] No results were generated. Check input data and script logic.")
        return

    df_results = pd.DataFrame(results) # Changed variable name to df_results
    output_path = "eval_lightgbm_outputs/metrics_summary_lgbm.csv"
    df_results.to_csv(output_path, index=False)

    print("\n--- Evaluation Summary (Mean Metrics) ---")
    # Group by both label and model_type for a clearer summary
    print(df_results.groupby(['label', 'model_type']).mean(numeric_only=True).to_markdown(floatfmt=".4g"))
    print(f"\n[SUCCESS] Evaluation complete. Full summary saved to {output_path}")

if __name__ == "__main__":
    main()

