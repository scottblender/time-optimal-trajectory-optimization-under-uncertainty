# ml_script_lightgbm_evaluate_model.py

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))

# ==============================================================================
# === DIAGNOSTIC FUNCTION ======================================================
# ==============================================================================
def inspect_file(filepath):
    """Loads a .pkl file and prints the shape of the 'X' and 'y' arrays."""
    print("\n" + "="*40)
    print(f"INSPECTING FILE: {filepath}")
    print("="*40)
    
    try:
        data = joblib.load(filepath)
        print(f"File loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Could not load file: {e}")
        return

    for key in ['X', 'y']:
        if key in data:
            content = data[key]
            if hasattr(content, 'shape'):
                print(f"  -> Content of '{key}': Shape = {content.shape}")
            else:
                print(f"  -> Content of '{key}': Type = {type(content)}, Length = {len(content)}")
        else:
            print(f"  -> Key '{key}' NOT FOUND in file.")
    print("="*40)

# ==============================================================================
# === HELPER FUNCTIONS =========================================================
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
    print("\n" + "#"*40)
    print(f"STARTING EVALUATION FOR: {model_name}")
    print("#"*40)

    try:
        model = joblib.load(model_path)
        data = joblib.load(data_path)
        # Gravitational parameter mu should be consistent across project
        mu = 27.8996 
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load file: {e}. Ensure training has been run and data files are present.")
        return

    X_eval_raw = data["X"]
    y_true_raw = data["y"]
    
    # Define columns based on the training script
    x_cols_raw = ['t', 'p', 'f', 'g', 'h', 'k','L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'bundle_idx', 'sigma_idx']
    features_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
    mee_cols = ['p', 'f', 'g', 'h', 'k', 'L', 'mass']

    y_cols = [f'y_{i}' for i in range(y_true_raw.shape[1])]
    
    df_x = pd.DataFrame(X_eval_raw, columns=x_cols_raw)
    df_y = pd.DataFrame(y_true_raw, columns=y_cols)

    df_combined = pd.concat([df_x, df_y], axis=1).dropna()
    
    if len(df_combined) == 0:
        print("[ERROR] No consistent data found after combining X and y. The source file may be corrupt.")
        return

    print("[INFO] De-duplicating combined evaluation data...")
    group_cols = ['t', 'sigma_idx', 'bundle_idx']
    df_dedup = df_combined.groupby(group_cols, sort=False).tail(1).reset_index(drop=True)
    
    X_eval = df_dedup[features_optimal]
    y_true = df_dedup[y_cols].to_numpy()
    print(f"[INFO] Data de-duplicated. Final evaluation samples: {len(y_true)}.")

    print("[INFO] Making predictions on the evaluation set...")
    y_pred = model.predict(X_eval)
    print("[INFO] Predictions complete.")

    # --- COSTATE PERFORMANCE METRICS ---
    r2_per_output = r2_score(y_true, y_pred, multioutput='raw_values')
    r2_avg = np.mean(r2_per_output)
    rmse_per_output = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    rmse_avg = np.mean(rmse_per_output)

    print("\n--- Costate Performance Metrics ---")
    print(f"Overall Average R² Score: {r2_avg:.4f}")
    print(f"Overall Average RMSE:     {rmse_avg:.4f}")
    
    print("\n--- Per-Costate Metrics ---")
    costate_labels = ['p', 'f', 'g', 'h', 'k', 'L', 'mass']
    for i, label in enumerate(costate_labels):
        print(f"  Costate λ_{label}: R² = {r2_per_output[i]:.4f}, RMSE = {rmse_per_output[i]:.4f}")

    # === THRUST VECTOR EVALUATION ===
    print("\n[INFO] Calculating thrust vectors from true and predicted costates...")
    mee_and_mass_array = df_dedup[mee_cols].to_numpy()

    u_true = compute_thrust_direction_vectorized(mu, mee_and_mass_array, y_true)
    u_pred = compute_thrust_direction_vectorized(mu, mee_and_mass_array, y_pred)
    
    valid_mask = ~np.isnan(u_true).any(axis=1) & ~np.isnan(u_pred).any(axis=1)
    u_true_valid = u_true[valid_mask]
    u_pred_valid = u_pred[valid_mask]
    
    if len(u_true_valid) > 0:
        print(f"[INFO] Evaluating thrust on {len(u_true_valid)} non-degenerate samples.")
        
        thrust_r2 = r2_score(u_true_valid, u_pred_valid, multioutput='raw_values')
        thrust_rmse = np.sqrt(mean_squared_error(u_true_valid, u_pred_valid, multioutput='raw_values'))

        print("\n--- Thrust Vector Performance Metrics ---")
        print(f"Overall Average R² Score: {np.mean(thrust_r2):.4f}")
        print(f"Overall Average RMSE:     {np.mean(thrust_rmse):.4f}")
        
        print("\n--- Per-Component Metrics ---")
        for i, component in enumerate(['u_x', 'u_y', 'u_z']):
            print(f"  Component {component}: R² = {thrust_r2[i]:.4f}, RMSE = {thrust_rmse[i]:.4f}")
    else:
        print("[WARNING] No valid (non-degenerate) thrust vectors could be computed.")

    # --- PLOTTING: COSTATES ---
    print("\n[INFO] Generating costate plot...")
    fig_costate, axes_costate = plt.subplots(3, 3, figsize=(15, 15))
    fig_costate.suptitle(f'Costate Prediction vs. Actual for {model_name}', fontsize=16)
    axes_costate = axes_costate.flatten()

    for i in range(y_true.shape[1]):
        ax = axes_costate[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.2, s=10)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
        ax.set_title(f'Costate λ_{costate_labels[i]} (R² = {r2_per_output[i]:.3f})')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.grid(True, linestyle=':')
    
    for i in range(y_true.shape[1], len(axes_costate)):
        fig_costate.delaxes(axes_costate[i])
    
    fig_costate.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename_costate = f"costate_evaluation_{model_name}.png"
    plt.savefig(plot_filename_costate)
    print(f"[INFO] Costate plot saved to {plot_filename_costate}")
    plt.close(fig_costate)

    # --- PLOTTING: THRUST VECTORS ---
    if len(u_true_valid) > 0:
        print("[INFO] Generating thrust vector plot...")
        fig_thrust, axes_thrust = plt.subplots(1, 3, figsize=(18, 5))
        fig_thrust.suptitle(f'Thrust Vector Prediction vs. Actual for {model_name}', fontsize=16)
        
        for i, component in enumerate(['u_x', 'u_y', 'u_z']):
            ax = axes_thrust[i]
            ax.scatter(u_true_valid[:, i], u_pred_valid[:, i], alpha=0.2, s=10)
            lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
            ax.set_title(f'Thrust Component {component} (R² = {thrust_r2[i]:.3f})')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.grid(True, linestyle=':')
            ax.set_aspect('equal', 'box')
            
        fig_thrust.tight_layout(rect=[0, 0.03, 1, 0.93])
        plot_filename_thrust = f"thrust_evaluation_{model_name}.png"
        plt.savefig(plot_filename_thrust)
        print(f"[INFO] Thrust plot saved to {plot_filename_thrust}")
        plt.close(fig_thrust)

if __name__ == '__main__':
    print("--- RUNNING DIAGNOSTIC CHECK ON SAVED DATA FILES ---")
    # NOTE: This script assumes 'segment_max.pkl' and 'segment_min.pkl' exist as evaluation sets.
    inspect_file("segment_max.pkl")
    inspect_file("segment_min.pkl")
    print("\n--- DIAGNOSTIC CHECK COMPLETE ---\n")
    
    evaluate_model("trained_model_max_optimal.pkl", "segment_max.pkl", "Max_Uncertainty_Model")
    evaluate_model("trained_model_min_optimal.pkl", "segment_min.pkl", "Min_Uncertainty_Model")