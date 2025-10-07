# evaluate_model.py

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score, mean_squared_error

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
import odefunc

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
def get_initial_state(t_start, forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu):
    start_idx = np.argmin(np.abs(forwardTspan - t_start))
    r0, v0 = r_tr[start_idx], v_tr[start_idx]
    mee0 = rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten()
    m0_prop, lam0_prop = mass_tr[start_idx], lam_tr[start_idx]
    return np.concatenate([mee0, [m0_prop], lam0_prop])

def propagate_and_get_nominal_state(t_eval_points, x0, mu, F, c, m0, g0):
    t_span = (t_eval_points[0], t_eval_points[-1])
    sol = solve_ivp(
        odefunc.odefunc, t_span, x0, args=(mu, F, c, m0, g0),
        t_eval=sorted(t_eval_points), dense_output=True, method='RK45', rtol=1e-6, atol=1e-9
    )
    propagated_mees = sol.y[:6, :].T
    r_propagated, v_propagated = mee2rv(*propagated_mees.T, mu)
    return {t: (r, v) for t, r, v in zip(sol.t, r_propagated, v_propagated)}

def add_engineered_features(df, nominal_state_lookup, mu):
    new_cols = ['pos_error_score', 'vel_error_score', 'energy_error_score']
    for col in new_cols:
        df[col] = 0.0
    
    epsilon = 1e-9 

    unique_times = df['t'].unique()
    grouped = df.groupby('t')
    
    for t in tqdm(unique_times, desc="[INFO] Engineering features for eval set"):
        time_key = min(nominal_state_lookup.keys(), key=lambda k: abs(k-t))
        group_df = grouped.get_group(t)
        group_indices = group_df.index
        
        r_nom, v_nom = nominal_state_lookup[time_key]
        
        mees = group_df[['p', 'f', 'g', 'h', 'k', 'L']].values
        r_samples, v_samples = mee2rv(*mees.T, mu)
        
        norm_r_samples = np.linalg.norm(r_samples, axis=1)
        norm_r_nom = np.linalg.norm(r_nom)
        df.loc[group_indices, 'pos_error_score'] = (norm_r_samples - norm_r_nom) / (norm_r_samples + norm_r_nom + epsilon)

        norm_v_samples = np.linalg.norm(v_samples, axis=1)
        norm_v_nom = np.linalg.norm(v_nom)
        df.loc[group_indices, 'vel_error_score'] = (norm_v_samples - norm_v_nom) / (norm_v_samples + norm_v_nom + epsilon)

        E_nom = 0.5 * np.dot(v_nom, v_nom) - mu / norm_r_nom
        E_samples = 0.5 * np.sum(v_samples**2, axis=1) - mu / norm_r_samples
        df.loc[group_indices, 'energy_error_score'] = (E_samples - E_nom) / (E_samples + E_nom + epsilon)
        
    return df

# ==============================================================================
# === MAIN EVALUATION FUNCTION =================================================
# ==============================================================================
def evaluate_model(model_path, data_path, nominal_data_bundle, model_name):
    print("\n" + "#"*40)
    print(f"STARTING EVALUATION FOR: {model_name}")
    print("#"*40)

    try:
        model = joblib.load(model_path)
        data = joblib.load(data_path)
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load file: {e}. Ensure training has been run first.")
        return

    X_eval_raw = data["X"]
    y_true_raw = data["y"]
    
    # --- ROBUST DE-DUPLICATION PIPELINE ---
    x_cols = ['t', 'p', 'f', 'g', 'h', 'k','L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'bundle_idx', 'sigma_idx']
    y_cols = [f'y_{i}' for i in range(y_true_raw.shape[1])]
    
    df_x = pd.DataFrame(X_eval_raw, columns=x_cols)
    df_y = pd.DataFrame(y_true_raw, columns=y_cols)

    df_combined = pd.concat([df_x, df_y], axis=1).dropna()
    
    if len(df_combined) == 0:
        print("[ERROR] No consistent data found after combining X and y. The source file may be corrupt.")
        return

    print("[INFO] De-duplicating combined evaluation data...")
    group_cols = ['t', 'sigma_idx', 'bundle_idx']
    df_dedup_combined = df_combined.groupby(group_cols, sort=False).tail(1)
    
    df_eval_dedup = df_dedup_combined[x_cols]
    y_true = df_dedup_combined[y_cols].to_numpy()
    print(f"[INFO] Data de-duplicated. Samples: {len(y_true)}.")

    # --- FEATURE ENGINEERING ---
    mu = nominal_data_bundle["mu"]
    r_tr, v_tr, mass_tr, lam_tr = nominal_data_bundle["r_tr"], nominal_data_bundle["v_tr"], nominal_data_bundle["mass_tr"], nominal_data_bundle["lam_tr"]
    forwardTspan = nominal_data_bundle["backTspan"][::-1]
    F_nom, c_nom, m0_nom, g0_nom = nominal_data_bundle["F"], nominal_data_bundle["c"], nominal_data_bundle["m0"], nominal_data_bundle["g0"]

    t_eval_unique = np.unique(np.round(df_eval_dedup['t'], 6))
    x0 = get_initial_state(t_eval_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
    nominal_state_lookup = propagate_and_get_nominal_state(t_eval_unique, x0, mu, F_nom, c_nom, m0_nom, g0_nom)
    
    df_eval_featured = add_engineered_features(df_eval_dedup.copy(), nominal_state_lookup, mu)
    
    feature_cols = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass',
                    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                    'pos_error_score', 'vel_error_score', 'energy_error_score']

    # MODIFIED: Keep the features as a DataFrame to preserve names
    X_features_df = df_eval_featured[feature_cols]

    # --- PREDICTION AND METRICS ---
    print("[INFO] Making predictions on the evaluation set...")
    # MODIFIED: Pass the DataFrame to the predict method
    y_pred = model.predict(X_features_df)
    print("[INFO] Predictions complete.")

    r2_per_output = r2_score(y_true, y_pred, multioutput='raw_values')
    r2_avg = np.mean(r2_per_output)
    rmse_per_output = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    rmse_avg = np.mean(rmse_per_output)

    print("\n--- Performance Metrics ---")
    print(f"Overall Average R² Score: {r2_avg:.4f}")
    print(f"Overall Average RMSE:     {rmse_avg:.4f}")
    
    print("\n--- Per-Costate Metrics ---")
    for i in range(len(r2_per_output)):
        print(f"  Costate λ_{i+1}: R² = {r2_per_output[i]:.4f}, RMSE = {rmse_per_output[i]:.4f}")

    # --- PLOTTING ---
    print("\n[INFO] Generating plot...")
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Prediction vs. Actual Values for {model_name}', fontsize=16)
    axes = axes.flatten()

    for i in range(y_true.shape[1]):
        ax = axes[i]
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.2, s=10)
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Prediction')
        ax.set_title(f'Costate λ_{i+1} (R² = {r2_per_output[i]:.3f})')
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predicted Values')
        ax.grid(True, linestyle=':')

    for i in range(y_true.shape[1], len(axes)):
        fig.delaxes(axes[i])

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = f"evaluation_plot_{model_name}.png"
    plt.savefig(plot_filename)
    print(f"[INFO] Plot saved to {plot_filename}")

if __name__ == '__main__':
    # === DIAGNOSTIC CHECK RUNS FIRST ===
    print("--- RUNNING DIAGNOSTIC CHECK ON SAVED DATA FILES ---")
    inspect_file("segment_max.pkl")
    inspect_file("segment_min.pkl")
    print("\n--- DIAGNOSTIC CHECK COMPLETE ---\n")
    
    try:
        nominal_data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load nominal data bundle: {e}.")
        exit()

    evaluate_model("trained_model_max.pkl", "segment_max.pkl", nominal_data, "Max_Uncertainty_Model")
    evaluate_model("trained_model_min.pkl", "segment_min.pkl", nominal_data, "Min_Uncertainty_Model")