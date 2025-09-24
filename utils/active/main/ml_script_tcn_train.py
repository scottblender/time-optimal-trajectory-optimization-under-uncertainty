# ml_script_tcn_train.py

import os
import glob
import joblib
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from scipy.integrate import solve_ivp

# Local helpers
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
from odefunc import odefunc

# === Constants ===
device = torch.device("cpu")
seq_length = 100
batch_size = 1024
hidden_size = 128
num_layers = 4
learning_rate = 0.002
epochs = 50
patience = 3

# === TCN Model Definition ===
class TCN_MANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, kernel_size=3):
        super().__init__()
        layers = []
        dilation = 1
        for _ in range(num_layers):
            layers.append(nn.Conv1d(input_size, hidden_size, kernel_size, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU())
            input_size = hidden_size
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.controller = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, output_size, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.controller(x)
        return x.transpose(1, 2)

# === Dataset for Bundled Sequences ===
class BundleSequenceDataset(Dataset):
    def __init__(self, X_data, y_data, seq_length):
        self.seq_length = seq_length
        self.sequences = []

        print("[DATASET] Creating sequences from bundled data...")
        
        # --- FIX START ---
        # The input X_data has 19 features and 2 bookkeeping columns at the end.
        # We create a DataFrame and group by the last two columns without adding new ones.
        df = pd.DataFrame(X_data)
        
        # Group by the last two columns, which are bundle_idx and sigma_idx
        grouped = df.groupby([df.columns[-2], df.columns[-1]])
        
        for _, group in tqdm(grouped, desc="  -> Processing groups"):
            if len(group) <= seq_length:
                continue
            
            # Select only the first 19 feature columns for the sequence, excluding the last two.
            x_group = group.iloc[:, :-2].values
            # --- FIX END ---

            # Find original indices to get corresponding y values
            original_indices = group.index.to_numpy()
            y_group = y_data[original_indices]

            for i in range(len(group) - seq_length):
                x_seq = x_group[i:i + seq_length]
                y_target = y_group[i + seq_length]
                self.sequences.append((x_seq, y_target))
        
        print(f"[DATASET] Created {len(self.sequences):,} sequences of length {self.seq_length}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y = self.sequences[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# === Feature Engineering Functions (from LightGBM script) ===
def get_initial_state(t_start, forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu):
    start_idx = np.argmin(np.abs(forwardTspan - t_start))
    r0, v0 = r_tr[start_idx], v_tr[start_idx]
    mee0 = rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten()
    m0_prop = mass_tr[start_idx]
    lam0_prop = lam_tr[start_idx]
    x0 = np.concatenate([mee0, [m0_prop], lam0_prop])
    return x0

def propagate_and_get_positions(t_eval_points, x0, mu, F, c, m0, g0):
    t_span = (t_eval_points[0], t_eval_points[-1])
    sol = solve_ivp(
        odefunc, t_span, x0, args=(mu, F, c, m0, g0),
        t_eval=sorted(t_eval_points), dense_output=True, method='RK45', rtol=1e-6, atol=1e-9
    )
    propagated_mees = sol.y[:6, :].T
    r_propagated, _ = mee2rv(*propagated_mees.T, mu)
    return {t: r for t, r in zip(sol.t, r_propagated)}

def add_deviation_features(df, nominal_pos_lookup, mu):
    df['score'] = 0.0
    df['delta_r_x'] = 0.0
    df['delta_r_y'] = 0.0
    df['delta_r_z'] = 0.0
    unique_times = df['t'].unique()
    grouped = df.groupby('t')
    for t in tqdm(unique_times, desc="[INFO] Calculating deviation features"):
        time_key = min(nominal_pos_lookup.keys(), key=lambda k: abs(k-t))
        group_df = grouped.get_group(t)
        r_nom = nominal_pos_lookup[time_key]
        mees = group_df[['p', 'f', 'g', 'h', 'k', 'L']].values
        r_samples, _ = mee2rv(*mees.T, mu)
        delta_r = r_samples - r_nom
        distances = np.linalg.norm(delta_r, axis=1)
        max_dist = np.max(distances)
        scores = distances / max_dist if max_dist > 1e-9 else np.zeros_like(distances)
        df.loc[group_df.index, 'score'] = scores
        df.loc[group_df.index, 'delta_r_x'] = delta_r[:, 0]
        df.loc[group_df.index, 'delta_r_y'] = delta_r[:, 1]
        df.loc[group_df.index, 'delta_r_z'] = delta_r[:, 2]
    return df

# === Training Function ===
def train_model_for_segment(segment_name, X_train_data, y_train_data, scaler_X, scaler_y):
    print(f"\n{'='*25}\n[INFO] Training model for: {segment_name.upper()}\n{'='*25}")
    
    # Scale data
    X_train_scaled = scaler_X.transform(X_train_data[:, :-2]) # Exclude bookkeeping columns
    y_train_scaled = scaler_y.transform(y_train_data)
    
    # Re-attach bookkeeping columns for dataset creation
    X_train_processed = np.hstack([X_train_scaled, X_train_data[:, -2:]])

    dataset = BundleSequenceDataset(X_train_processed, y_train_scaled, seq_length)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Train sequences: {len(train_data):,} | Val sequences: {len(val_data):,}")

    model = TCN_MANN(input_size=X_train_scaled.shape[1],
                     hidden_size=hidden_size,
                     output_size=y_train_scaled.shape[1],
                     num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    best_model_state = None
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[EPOCH {epoch+1}] Training")
        for Xb, yb in pbar:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)[:, -1, :]
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                pred = model(Xb.to(device))[:, -1, :]
                loss = criterion(pred, yb.to(device))
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"[EPOCH {epoch+1:03d}] Train Loss: {train_loss:.6f} | Val Loss (scaled): {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            print("  ✅ New best model found!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[EARLY STOP] No improvement for {patience} epochs.")
                break
    
    model_save_path = f"trained_model_tcn_{segment_name}.pt"
    torch.save(best_model_state, model_save_path)
    print(f"[DONE] Model saved to {model_save_path}")

# === Main Execution ===
if __name__ == "__main__":
    # --- Step 1: Load and Segment Data (replicates LightGBM workflow) ---
    with open("stride_1440min/bundle_segment_widths.txt") as f:
        lines = f.readlines()[1:]
        times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
        time_vals = times_arr[:, 0]
        max_idx = int(np.argmax(times_arr[:, 1]))
        min_idx_raw = int(np.argmin(times_arr[:, 1]))
        min_idx = np.argsort(times_arr[:, 1])[1] if min_idx_raw == len(times_arr) - 1 else min_idx_raw
    n = 50
    t_train_neighbors_max = time_vals[max(0, max_idx - n): max_idx + n + 1]
    t_train_neighbors_min = time_vals[max(0, min_idx - n): min_idx + n + 1]
    
    X_train_max, y_train_max = [], []
    X_train_min, y_train_min = [], []

    batch_files = sorted(glob.glob("baseline_stride_1/batch_*/data.pkl"))
    for file in tqdm(batch_files, desc="[INFO] Loading raw batch data"):
        d = joblib.load(file)
        Xb, yb = d["X"], d["y"]
        for t in t_train_neighbors_max:
            idx = np.isclose(Xb[:, 0], t)
            if np.any(idx): X_train_max.append(Xb[idx]); y_train_max.append(yb[idx])
        for t in t_train_neighbors_min:
            idx = np.isclose(Xb[:, 0], t)
            if np.any(idx): X_train_min.append(Xb[idx]); y_train_min.append(yb[idx])

    X_full_max, y_full_max = np.vstack(X_train_max), np.vstack(y_train_max)
    X_full_min, y_full_min = np.vstack(X_train_min), np.vstack(y_train_min)

    # --- Step 2: Deduplication (replicates LightGBM workflow) ---
    feature_cols = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass',
                    'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
    bookkeeping_cols = ['bundle_idx', 'sigma_idx']
    
    ## ---  Deduplication for MAX data ---
    print("\n[INFO] Deduplicating MAX data...")
    df_X_max = pd.DataFrame(X_full_max, columns=feature_cols + bookkeeping_cols)
    df_y_max = pd.DataFrame(y_full_max)
    df_X_max["orig_index"] = np.arange(len(df_X_max))
    group_cols = ['t', 'sigma_idx', 'bundle_idx']
    df_dedup_max = df_X_max.groupby(group_cols, sort=False).tail(1).sort_values("orig_index")
    y_full_max_dedup = df_y_max.iloc[df_dedup_max.index].to_numpy()
    print(f"[INFO] MAX data shape before dedup: {X_full_max.shape} -> after: {df_dedup_max.shape}")
    ## --- END ADDITION ---
    
    ## --- ADDED: Deduplication for MIN data ---
    print("[INFO] Deduplicating MIN data...")
    df_X_min = pd.DataFrame(X_full_min, columns=feature_cols + bookkeeping_cols)
    df_y_min = pd.DataFrame(y_full_min)
    df_X_min["orig_index"] = np.arange(len(df_X_min))
    df_dedup_min = df_X_min.groupby(group_cols, sort=False).tail(1).sort_values("orig_index")
    y_full_min_dedup = df_y_min.iloc[df_dedup_min.index].to_numpy()
    print(f"[INFO] MIN data shape before dedup: {X_full_min.shape} -> after: {df_dedup_min.shape}")
    ## --- END ADDITION ---

    # --- Step 3: Feature Engineering (replicates LightGBM workflow) ---
    nominal_data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
    mu, F_nom, c_nom, m0_nom, g0_nom = nominal_data["mu"], nominal_data["F"], nominal_data["c"], nominal_data["m0"], nominal_data["g0"]
    r_tr, v_tr, mass_tr, lam_tr = nominal_data["r_tr"], nominal_data["v_tr"], nominal_data["mass_tr"], nominal_data["lam_tr"]
    forwardTspan = nominal_data["backTspan"][::-1]

    # Process MAX data
    t_train_max_unique = np.unique(np.round(df_dedup_max['t'], 6))
    x0_max = get_initial_state(t_train_max_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
    nominal_pos_max = propagate_and_get_positions(t_train_max_unique, x0_max, mu, F_nom, c_nom, m0_nom, g0_nom)
    df_X_max_featured = add_deviation_features(df_dedup_max, nominal_pos_max, mu)
    
    # Process MIN data
    t_train_min_unique = np.unique(np.round(df_dedup_min['t'], 6))
    x0_min = get_initial_state(t_train_min_unique[0], forwardTspan, r_tr, v_tr, mass_tr, lam_tr, mu)
    nominal_pos_min = propagate_and_get_positions(t_train_min_unique, x0_min, mu, F_nom, c_nom, m0_nom, g0_nom)
    df_X_min_featured = add_deviation_features(df_dedup_min, nominal_pos_min, mu)
    
    final_feature_cols = feature_cols + ['score', 'delta_r_x', 'delta_r_y', 'delta_r_z']
    
    X_max_with_features = np.hstack([df_X_max_featured[final_feature_cols].values, df_X_max_featured[bookkeeping_cols].values])
    X_min_with_features = np.hstack([df_X_min_featured[final_feature_cols].values, df_X_min_featured[bookkeeping_cols].values])

    # --- Step 4: Removal of Appended sigma0 Rows (replicates LightGBM workflow) ---
    ## --- sigma0 removal for MAX data ---
    print("\n[INFO] Removing appended sigma0 rows from MAX data...")
    is_sigma0_max = X_max_with_features[:, -2] == 0
    is_zero_cov_max = np.all(np.isclose(X_max_with_features[:, 8:15], 0.0, atol=1e-12), axis=1)
    is_appended_sigma0_max = is_sigma0_max & is_zero_cov_max
    X_max_final = X_max_with_features[~is_appended_sigma0_max]
    y_max_final = y_full_max_dedup[~is_appended_sigma0_max]
    print(f"[INFO] Removed {np.sum(is_appended_sigma0_max)} sigma0 rows from MAX data.")
    ## --- END  ---
    
    ## --- ADDED: sigma0 removal for MIN data ---
    print("[INFO] Removing appended sigma0 rows from MIN data...")
    is_sigma0_min = X_min_with_features[:, -2] == 0
    is_zero_cov_min = np.all(np.isclose(X_min_with_features[:, 8:15], 0.0, atol=1e-12), axis=1)
    is_appended_sigma0_min = is_sigma0_min & is_zero_cov_min
    X_min_final = X_min_with_features[~is_appended_sigma0_min]
    y_min_final = y_full_min_dedup[~is_appended_sigma0_min]
    print(f"[INFO] Removed {np.sum(is_appended_sigma0_min)} sigma0 rows from MIN data.")
    ## --- END ---

    # --- Step 5: Create and Save Scalers ---
    print("\n[INFO] Fitting and saving data scalers...")
    combined_X_for_scaling = np.vstack([X_max_final, X_min_final])
    combined_y_for_scaling = np.vstack([y_max_final, y_min_final])
    
    scaler_X = StandardScaler().fit(combined_X_for_scaling[:, :-2]) # Don't scale bookkeeping indices
    scaler_y = StandardScaler().fit(combined_y_for_scaling)
    
    joblib.dump(scaler_X, "scaler_tcn_X.pkl")
    joblib.dump(scaler_y, "scaler_tcn_y.pkl")
    print("[INFO] Scalers saved to scaler_tcn_X.pkl and scaler_tcn_y.pkl")
    
    # --- Step 6: Train Models for Each Segment ---
    train_model_for_segment("min", X_min_final, y_min_final, scaler_X, scaler_y)
    train_model_for_segment("max", X_max_final, y_max_final, scaler_X, scaler_y)