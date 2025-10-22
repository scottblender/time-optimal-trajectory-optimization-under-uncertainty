import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pyplot as plt

# === Configuration ===
PROCESSED_DATA_FILENAME = "processed_training_data.pkl"
MODEL_TO_PLOT = "min_corrective" # Options: "max_optimal", "max_corrective", "min_optimal", "min_corrective"
OUTPUT_PLOT_FILENAME = "feature_importance_plot.png"

# === Load Pre-processed Data ===
print(f"[INFO] Loading processed data from {PROCESSED_DATA_FILENAME}...")
try:
    processed_data = joblib.load(PROCESSED_DATA_FILENAME)
    df_dedup_max, y_max_dedup, y_max_corr = processed_data["max_data"]["df"], processed_data["max_data"]["y_opt"], processed_data["max_data"]["y_corr"]
    df_dedup_min, y_min_dedup, y_min_corr = processed_data["min_data"]["df"], processed_data["min_data"]["y_opt"], processed_data["min_data"]["y_corr"]
except FileNotFoundError:
    print(f"[ERROR] Processed data file not found. Please run '1_generate_datasets.py' first.")
    exit()

# === Model and Feature Configuration ===
base_model_params = {
    'n_estimators': 1500, 'learning_rate': 0.005, 'max_depth': 15,
    'min_child_samples': 10, 'force_row_wise': True,
    'random_state': 42, 'verbose': -1
}
features_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']

# --- CORRECTED: Removed magnitude features from the list ---
features_corrective = features_optimal + [
    'delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz',
    'delta_m', 't_go'
]

# === Training Function ===
def train_and_save_models(df_dedup, y_dedup, y_corr, model_name_prefix):
    mask = ~( (df_dedup['sigma_idx'] == 0) & np.all(np.isclose(df_dedup[['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']].values, 0.0), axis=1) )

    # Train Optimal Model
    X_opt = df_dedup.loc[mask, features_optimal].values
    y_opt = y_dedup[mask]
    print(f"[INFO] Training Optimal {model_name_prefix.upper()} model with shape: {X_opt.shape}")
    model_opt = MultiOutputRegressor(LGBMRegressor(**base_model_params))
    model_opt.fit(X_opt, y_opt)
    joblib.dump(model_opt, f"trained_model_{model_name_prefix}_optimal.pkl")

    # Train Corrective Model
    X_corr = df_dedup.loc[mask, features_corrective].values
    y_corr_masked = y_corr[mask]
    print(f"[INFO] Training Corrective {model_name_prefix.upper()} model with shape: {X_corr.shape}")
    model_corr = MultiOutputRegressor(LGBMRegressor(**base_model_params))
    model_corr.fit(X_corr, y_corr_masked)
    joblib.dump(model_corr, f"trained_model_{model_name_prefix}_corrective.pkl")

# --- Train All Models ---
print("\n" + "="*20 + " TRAINING MAX MODELS " + "="*20)
train_and_save_models(df_dedup_max, y_max_dedup, y_max_corr, "max")
print("\n" + "="*20 + " TRAINING MIN MODELS " + "="*20)
train_and_save_models(df_dedup_min, y_min_dedup, y_min_corr, "min")
print("\n[SUCCESS] All models have been trained and saved.")
