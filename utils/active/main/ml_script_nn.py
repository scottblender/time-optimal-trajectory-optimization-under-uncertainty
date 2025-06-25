import os
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import mahalanobis
from scipy.integrate import solve_ivp
from numpy.linalg import inv
import mlflow
import mlflow.pytorch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import mee2rv
import rv2mee
import odefunc

# === Constants ===
mu, F, c, m0, g0 = 27.899633640439433, 0.33, 4.4246246663455135, 4000, 9.81
num_eval_per_step = 20
bundle_idx = 32

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

class MANN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(MANN_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out)

# === Load Sigma Weights ===
W_data = joblib.load("sweep_stride_1_config_baseline_data.pkl")
Wm, Wc = W_data["Wm"], W_data["Wc"]

# === Hyperparameters ===
hidden_size = 128
num_layers = 5
epochs = 50
learning_rate = 0.001
loss_function_name = "MSELoss"

results = []
stride_files = sorted(glob.glob("sweep_stride_*_config_*_data.pkl"))

for file in stride_files:
    parts = file.split("_")
    stride = int(parts[2])
    distribution = parts[4]

    with mlflow.start_run(run_name=f"{file}"):
        # Log run configuration
        mlflow.log_param("stride", stride)
        mlflow.log_param("distribution", distribution)
        mlflow.log_param("hidden_size", hidden_size)
        mlflow.log_param("num_layers", num_layers)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("loss_function", loss_function_name)

        data = joblib.load(file)
        X_full, y_full = data["X"], data["y"]
        X_nn, y_nn = X_full[:-1], y_full[:-1]

        num_sigma = len(np.unique(X_nn[:, -1]))
        time_steps = len(np.unique(X_nn[:, 0]))
        input_size = X_nn.shape[1] - 3
        output_size = y_full.shape[1]

        X_seq = X_nn[:, 1:-2].reshape(num_sigma, time_steps, input_size)
        y_seq = y_nn.reshape(num_sigma, time_steps, output_size)

        X_seq = torch.tensor(X_seq, dtype=torch.float32)
        y_seq = torch.tensor(y_seq, dtype=torch.float32)

        model = MANN_LSTM(input_size, hidden_size, output_size, num_layers)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_seq)
            loss = criterion(outputs, y_seq)
            loss.backward()
            optimizer.step()
            print(f"[{file}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # === Predict Control ===
        model.eval()
        with torch.no_grad():
            predictions = model(X_seq).cpu().numpy().reshape(-1, output_size)

        # === Propagate All Sigma Trajectories ===
        X_bundle = X_full[X_full[:, -2] == bundle_idx]
        time_vals = np.unique(X_bundle[:, 0])
        sigma_indices = np.unique(X_bundle[:, -1]).astype(int)
        r_stack = []

        for sigma_idx in sigma_indices:
            r_sigma = []
            sigma_rows = X_bundle[(X_bundle[:, -1] == sigma_idx) & (X_bundle[:, 0] < time_vals[-1])]
            for i, t0 in enumerate(time_vals[:-1]):
                row = sigma_rows[sigma_rows[:, 0] == t0][0]
                idx = sigma_idx * (time_steps - 1) + i
                lam_pred = predictions[idx]
                S = np.concatenate([row[1:8], lam_pred])
                t_eval = np.linspace(t0, time_vals[i+1], num_eval_per_step)
                Sf = solve_ivp(lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0),
                               [t0, time_vals[i+1]], S, t_eval=t_eval)
                r_step, _ = mee2rv.mee2rv(*Sf.y[:6], mu)
                r_sigma.append(r_step)
            r_stack.append(np.stack(r_sigma))  # (time_steps-1, num_eval, 3)

        r_stack = np.array(r_stack)  # shape: (num_sigma, time_steps-1, num_eval, 3)

        # === Compute Full-Time Covariance History ===
        mean_r = np.sum(Wm[:, None, None, None] * r_stack, axis=0)
        devs_r = r_stack - mean_r[None, :, :, :]
        P_full_r = np.einsum("i,ijkl,ijkm->jklm", Wc, devs_r, devs_r)
        P_diag_r = np.array([[np.diag(np.diag(P_full_r[t, k])) for k in range(num_eval_per_step)]
                             for t in range(time_steps - 1)])

        # === Reference r_ref from appended sigma 0 ===
        sigma0_rows = X_bundle[(X_bundle[:, 0] == time_vals[-1]) & (X_bundle[:, -1] == 0)]
        appended = next((row for row in sigma0_rows if np.allclose(row[8:15], 0)), None)
        if appended is None:
            raise ValueError(f"Appended sigma 0 not found at t = {time_vals[-1]:.6f}")
        r_ref, v_ref = mee2rv.mee2rv(*appended[1:7].reshape(6, 1), mu)
        r_ref, v_ref = r_ref.flatten(), v_ref.flatten()

        # === Final Metrics ===
        final_cov = P_diag_r[-1, -1]
        inv_cov_r = inv(final_cov)
        r_pred_all = r_stack[0]
        r_final_pred = r_pred_all[-1, -1, :]
        mahalanobis_pred = mahalanobis(r_final_pred, r_ref, inv_cov_r)

        lam_pred_0 = predictions[0]
        lam_true_0 = y_full[(X_full[:, 0] == time_vals[0]) & (X_full[:, -1] == 0)][0]
        cos_sim = cosine_similarity(lam_pred_0, lam_true_0)

        metrics = {
            "model mse": mean_squared_error(y_seq.view(-1, output_size), torch.tensor(predictions)),
            "cosine similarity": cos_sim,
            "final position deviation": np.linalg.norm(r_final_pred - r_ref),
            "mahalanobis_pred": mahalanobis_pred,
            "x mse": np.mean((r_ref[0] - r_pred_all[:, -1, 0]) ** 2),
            "y mse": np.mean((r_ref[1] - r_pred_all[:, -1, 1]) ** 2),
            "z mse": np.mean((r_ref[2] - r_pred_all[:, -1, 2]) ** 2),
        }

        for j in range(3):
            metrics[f"propagated_cov_diag_{j}"] = final_cov[j, j]

        for key, val in metrics.items():
            if isinstance(val, (int, float, np.floating)):
                mlflow.log_metric(key.replace(" ", "_"), float(val))

        mlflow.pytorch.log_model(model, "model")
        results.append({
            "time stride": stride,
            "distribution": distribution,
            **metrics
        })

# === Save Final Results to CSV ===
df = pd.DataFrame(results)
df.to_csv("mann_lstm_stride_distribution_metrics.csv", index=False)
print("Saved: mann_lstm_stride_distribution_metrics.csv")
