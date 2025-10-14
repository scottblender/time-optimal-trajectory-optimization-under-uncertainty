# ml_script_lightgbm_evaluate_model_summary.py

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# ==============================================================================
# === HELPER FUNCTION ==========================================================
# ==============================================================================
def compute_thrust_direction_vectorized(mu, mee_array, lam_array):
    """
    Computes optimal thrust direction vectors from arrays of MEE states and costates.
    """
    p, f, g, h, k, L = [mee_array[:, i] for i in range(6)]
    lam_p, lam_f, lam_g, lam_h, lam_k, lam_L = [lam_array[:, i] for i in range(6)]

    lam_matrix = np.stack([lam_p, lam_f, lam_g, lam_h, lam_k, lam_L], axis=1)
    
    SinL, CosL = np.sin(L), np.cos(L)
    w = 1 + f * CosL + g * SinL
    
    is_degenerate = np.isclose(w, 0, atol=1e-10)
    
    s = 1 + h**2 + k**2
    C1 = np.sqrt(p / mu)
    C2 = 1 / w
    C3 = h * SinL - k * CosL

    A = np.zeros((mee_array.shape[0], 6, 3))
    A[:, 0, 1] = 2 * p * C2 * C1
    A[:, 1, 0] = C1 * SinL
    A[:, 1, 1] = C1 * C2 * ((w + 1) * CosL + f)
    A[:, 1, 2] = -C1 * (g / w) * C3
    A[:, 2, 0] = -C1 * CosL
    A[:, 2, 1] = C1 * C2 * ((w + 1) * SinL + g)
    A[:, 2, 2] = C1 * (f / w) * C3
    A[:, 3, 2] = C1 * s * CosL * C2 / 2
    A[:, 4, 2] = C1 * s * SinL * C2 / 2
    A[:, 5, 2] = C1 * C2 * C3
    
    mat = np.einsum('nij,nj->ni', A.transpose(0, 2, 1), lam_matrix)
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    
    u_vecs = np.zeros_like(mat)
    non_zero_norm_mask = norm.flatten() > 1e-9
    u_vecs[non_zero_norm_mask] = mat[non_zero_norm_mask] / norm[non_zero_norm_mask]
    u_vecs[is_degenerate] = np.nan
    
    return u_vecs

# ==============================================================================
# === MAIN EVALUATION FUNCTION =================================================
# ==============================================================================
def evaluate_model(model_path, data_path, model_name):
    print("\n" + "#"*50)
    print(f"# STARTING EVALUATION FOR: {model_name}")
    print("#"*50)

    try:
        model = joblib.load(model_path)
        data = joblib.load(data_path)
        mu = 27.8996 
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load file for {model_name}: {e}.")
        return None

    X_eval_raw, y_true_raw = data["X"], data["y"]
    
    # Define columns based on the training script
    x_cols_raw = ['t', 'p', 'f', 'g', 'h', 'k','L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'bundle_idx', 'sigma_idx']
    features_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
    mee_cols = ['p', 'f', 'g', 'h', 'k', 'L', 'mass']
    y_cols = [f'y_{i}' for i in range(y_true_raw.shape[1])]
    
    df_x = pd.DataFrame(X_eval_raw, columns=x_cols_raw)
    df_y = pd.DataFrame(y_true_raw, columns=y_cols)
    df_combined = pd.concat([df_x, df_y], axis=1).dropna()
    
    if len(df_combined) == 0:
        print(f"[ERROR] No consistent data found for {model_name}.")
        return None

    group_cols = ['t', 'sigma_idx', 'bundle_idx']
    df_dedup = df_combined.groupby(group_cols, sort=False).tail(1).reset_index(drop=True)
    
    X_eval, y_true = df_dedup[features_optimal], df_dedup[y_cols].to_numpy()
    print(f"[INFO] Evaluating {model_name} on {len(y_true)} de-duplicated samples.")

    y_pred = model.predict(X_eval)

    # --- PERFORMANCE METRICS ---
    results = {"model_name": model_name}
    costate_labels = ['p', 'f', 'g', 'h', 'k', 'L', 'mass']
    
    r2_per_output = r2_score(y_true, y_pred, multioutput='raw_values')
    rmse_per_output = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    
    results['Avg Costate R²'] = np.mean(r2_per_output)
    results['Avg Costate RMSE'] = np.mean(rmse_per_output)
    for i, label in enumerate(costate_labels):
        results[f'R² (λ_{label})'] = r2_per_output[i]
        results[f'RMSE (λ_{label})'] = rmse_per_output[i]

    # --- THRUST VECTOR EVALUATION ---
    mee_and_mass_array = df_dedup[mee_cols].to_numpy()
    u_true = compute_thrust_direction_vectorized(mu, mee_and_mass_array, y_true)
    u_pred = compute_thrust_direction_vectorized(mu, mee_and_mass_array, y_pred)
    
    valid_mask = ~np.isnan(u_true).any(axis=1) & ~np.isnan(u_pred).any(axis=1)
    if valid_mask.sum() > 0:
        thrust_r2 = r2_score(u_true[valid_mask], u_pred[valid_mask], multioutput='raw_values')
        thrust_rmse = np.sqrt(mean_squared_error(u_true[valid_mask], u_pred[valid_mask], multioutput='raw_values'))
        results['Avg Thrust R²'] = np.mean(thrust_r2)
        results['Avg Thrust RMSE'] = np.mean(thrust_rmse)
        for i, comp in enumerate(['x', 'y', 'z']):
            results[f'R² (u_{comp})'] = thrust_r2[i]
            results[f'RMSE (u_{comp})'] = thrust_rmse[i]
    else:
        print(f"[WARNING] No valid thrust vectors for {model_name}.")
        results.update({'Avg Thrust R²': np.nan, 'Avg Thrust RMSE': np.nan})

    print(f"[INFO] Evaluation complete for {model_name}.")
    
    # --- PLOTTING (Optional but good practice) ---
    # The plotting code from your original script can be placed here if desired.
    # It has been omitted for brevity as per the request to focus on the summary table.

    return results

# ==============================================================================
# === SCRIPT EXECUTION =========================================================
# ==============================================================================
if __name__ == '__main__':
    models_to_evaluate = [
        {
            "model_path": "trained_model_min_optimal.pkl",
            "data_path": "segment_min.pkl",
            "model_name": "Min Width"
        },
        {
            "model_path": "trained_model_max_optimal.pkl",
            "data_path": "segment_max.pkl",
            "model_name": "Max Width"
        }
    ]

    all_results = []
    for config in models_to_evaluate:
        result = evaluate_model(**config)
        if result:
            all_results.append(result)
            
    if not all_results:
        print("\nNo models were evaluated successfully. Exiting.")
    else:
        df_results = pd.DataFrame(all_results)
        df_results.set_index('model_name', inplace=True)

        # --- Print Summary Table ---
        summary_cols = ['Avg Costate R²', 'Avg Costate RMSE', 'Avg Thrust R²', 'Avg Thrust RMSE']
        print("\n\n" + "="*60)
        print("          MODEL PERFORMANCE SUMMARY")
        print("="*60)
        print(df_results[summary_cols].to_markdown(floatfmt=".4f"))

        # --- Print Detailed Costate Table ---
        costate_cols_r2 = [col for col in df_results.columns if 'R² (λ' in col]
        costate_cols_rmse = [col for col in df_results.columns if 'RMSE (λ' in col]
        print("\n\n" + "="*60)
        print("          DETAILED COSTATE METRICS (R²)")
        print("="*60)
        print(df_results[costate_cols_r2].to_markdown(floatfmt=".4f"))
        
        print("\n\n" + "="*60)
        print("          DETAILED COSTATE METRICS (RMSE)")
        print("="*60)
        print(df_results[costate_cols_rmse].to_markdown(floatfmt=".4f"))

        # --- Print Detailed Thrust Table ---
        thrust_cols_r2 = [col for col in df_results.columns if 'R² (u' in col]
        thrust_cols_rmse = [col for col in df_results.columns if 'RMSE (u' in col]
        if thrust_cols_r2: # Only print if thrust data was available
            print("\n\n" + "="*60)
            print("          DETAILED THRUST METRICS (R² & RMSE)")
            print("="*60)
            print(df_results[thrust_cols_r2 + thrust_cols_rmse].to_markdown(floatfmt=".4f"))