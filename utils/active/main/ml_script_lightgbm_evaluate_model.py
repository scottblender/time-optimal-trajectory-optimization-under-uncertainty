import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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
    s = 1 + h**2 + k**2
    C1 = np.sqrt(p / mu)
    
    # Handle potential division by zero if w is close to 0
    w[np.isclose(w, 0)] = 1e-12
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
    
    return u_vecs

# ==============================================================================
# === MAIN EVALUATION FUNCTION =================================================
# ==============================================================================
def evaluate_model(model_path, df_x, y_true, mu, model_name, model_type):
    print("\n" + "#"*50)
    print(f"# EVALUATING: {model_name} ({model_type} model)")
    print("#"*50)

    try:
        model = joblib.load(model_path)
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load model file: {e}.")
        return None

    # Define feature sets based on the model type being evaluated
    features_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
    
    # --- CORRECTED: Removed magnitude features from the list ---
    features_corrective = features_optimal + [
        'delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz',
        'delta_m', 't_go'
    ]
    
    if model_type == 'optimal':
        X_eval = df_x[features_optimal]
    elif model_type == 'corrective':
        X_eval = df_x[features_corrective]
    else:
        print(f"[ERROR] Unknown model type: {model_type}")
        return None

    print(f"[INFO] Evaluating model on {len(y_true)} samples.")
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
    mee_and_mass_array = df_x[['p', 'f', 'g', 'h', 'k', 'L']].to_numpy() # Note: mass is not needed
    u_true = compute_thrust_direction_vectorized(mu, mee_and_mass_array, y_true[:, :6])
    u_pred = compute_thrust_direction_vectorized(mu, mee_and_mass_array, y_pred[:, :6])
    
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
        results.update({'Avg Thrust R²': np.nan, 'Avg Thrust RMSE': np.nan})

    print(f"[INFO] Evaluation complete for {model_name}.")
    return results

# ==============================================================================
# === SCRIPT EXECUTION =========================================================
# ==============================================================================
if __name__ == '__main__':
    # Load the single processed data file once
    try:
        processed_data = joblib.load("processed_training_data.pkl")
        nominal_data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
        mu = nominal_data["mu"]
    except FileNotFoundError as e:
        print(f"[FATAL ERROR] A required data file is missing: {e}")
        print("Please ensure 'processed_training_data.pkl' and 'bundle_data_1440min.pkl' are present.")
        exit()

    # Define all four models to be evaluated
    models_to_evaluate = [
        {"model_name": "Max Width Optimal", "model_path": "trained_model_max_optimal.pkl", "data_key": "max_data", "model_type": "optimal"},
        {"model_name": "Max Width Corrective", "model_path": "trained_model_max_corrective.pkl", "data_key": "max_data", "model_type": "corrective"},
        {"model_name": "Min Width Optimal", "model_path": "trained_model_min_optimal.pkl", "data_key": "min_data", "model_type": "optimal"},
        {"model_name": "Min Width Corrective", "model_path": "trained_model_min_corrective.pkl", "data_key": "min_data", "model_type": "corrective"},
    ]

    all_results = []
    for config in models_to_evaluate:
        # Select the correct data slice from the loaded file
        data_slice = processed_data[config["data_key"]]
        df_x = data_slice["df"]
        
        # Select the correct target variable based on model type
        if config["model_type"] == "optimal":
            y_true = data_slice["y_opt"]
        else: # corrective
            y_true = data_slice["y_corr"]
            
        result = evaluate_model(
            model_path=config["model_path"],
            df_x=df_x,
            y_true=y_true,
            mu=mu,
            model_name=config["model_name"],
            model_type=config["model_type"]
        )
        if result:
            all_results.append(result)
            
    if not all_results:
        print("\nNo models were evaluated successfully. Exiting.")
    else:
        df_results = pd.DataFrame(all_results).set_index('model_name')

        # --- Print Summary Tables ---
        summary_cols = ['Avg Costate R²', 'Avg Costate RMSE', 'Avg Thrust R²', 'Avg Thrust RMSE']
        print("\n\n" + "="*65)
        print(" " * 20 + "MODEL PERFORMANCE SUMMARY")
        print("="*65)
        print(df_results[summary_cols].to_markdown(floatfmt=".4f"))

        costate_cols_r2 = [col for col in df_results.columns if 'R² (λ' in col]
        print("\n\n" + "="*65)
        print(" " * 18 + "DETAILED COSTATE METRICS (R²)")
        print("="*65)
        print(df_results[costate_cols_r2].to_markdown(floatfmt=".4f"))
        
        thrust_cols_r2 = [col for col in df_results.columns if 'R² (u' in col]
        if thrust_cols_r2:
            print("\n\n" + "="*65)
            print(" " * 18 + "DETAILED THRUST METRICS (R²)")
            print("="*65)
            print(df_results[thrust_cols_r2].to_markdown(floatfmt=".4f"))

