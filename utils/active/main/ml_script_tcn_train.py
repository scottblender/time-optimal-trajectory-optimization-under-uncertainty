import os
import glob
import joblib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, random_split

# === Constants ===
device = torch.device("cpu")
seq_length = 100
batch_size = 512
hidden_size = 128
num_layers = 4
learning_rate = 0.001
epochs = 50
patience = 4

# === Model ===
class TCN_MANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, kernel_size=3):
        super().__init__()
        tcn_layers = []
        dilation = 1
        for _ in range(num_layers):
            tcn_layers.append(nn.Conv1d(input_size, hidden_size, kernel_size,
                                        padding=dilation, dilation=dilation))
            tcn_layers.append(nn.ReLU())
            input_size = hidden_size
            dilation *= 2
        self.tcn = nn.Sequential(*tcn_layers)
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

# === Dataset ===
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, time_to_rows_X, time_to_rows_y, time_vals, seq_length):
        self.X_map = time_to_rows_X
        self.y_map = time_to_rows_y
        self.times = time_vals
        self.seq_length = seq_length
        self.stride = seq_length // 2
        print("[DATASET] Building sequence index...")
        self.indices = self._build_indices()
        print(f"[DATASET] Total usable sequences: {len(self.indices)}")

    def _build_indices(self):
        indices = []
        for i in range(0, len(self.times) - self.seq_length, self.stride):
            t_seq = self.times[i:i + self.seq_length]
            t_target = self.times[i + self.seq_length]
            if all(t in self.X_map for t in t_seq) and t_target in self.y_map:
                indices.append((t_seq, t_target))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_seq, t_target = self.indices[idx]
        X_seq = np.vstack([self.X_map[t] for t in t_seq])
        y_vals = np.vstack(self.y_map[t_target])
        y_avg = y_vals.mean(axis=0)
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y_avg, dtype=torch.float32)

# === Load preprocessed data
print("[STEP 1] Loading preprocessed batches and building time index...")
time_to_rows_X = defaultdict(list)
time_to_rows_y = defaultdict(list)
batch_files = sorted(glob.glob("baseline_stride_1_cleaned/batch_*/data.pkl"))
print(f"[INFO] Found {len(batch_files)} cleaned batch files.")

for file in tqdm(batch_files, desc="[INDEX] Loading batches"):
    d = joblib.load(file, mmap_mode='r')
    Xb, yb = d["X"], d["y"]
    for i in range(len(Xb)):
        t = Xb[i, 0]
        time_to_rows_X[t].append(Xb[i])
        time_to_rows_y[t].append(yb[i])
time_vals = sorted(time_to_rows_X.keys())
print(f"[INFO] Total unique timestamps: {len(time_vals)}")

# === Dataset and loaders
print("[STEP 2] Constructing Dataset and DataLoaders...")
dataset = TimeSeriesDataset(time_to_rows_X, time_to_rows_y, time_vals, seq_length)
val_size = min(int(0.01 * len(dataset)), 1_000_000)
train_size = len(dataset) - val_size
train_data, val_data = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
print(f"[INFO] Train samples: {len(train_data)} | Val samples: {len(val_data)}")

# === Model setup
print("[STEP 3] Initializing model...")
sample_X, _ = dataset[0]
model = TCN_MANN(input_size=sample_X.shape[1], hidden_size=hidden_size,
                 output_size=sample_X.shape[1], num_layers=num_layers).to(device)
model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
print(f"[MODEL] Model initialized with input size {sample_X.shape[1]} and sequence length {seq_length}")

# === Training
print("[STEP 4] Starting training...")
best_model = None
best_val_loss = float("inf")
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    print(f"\n[EPOCH {epoch+1}] Training...")
    for batch_idx, (Xb, yb) in enumerate(train_loader):
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)[:, -1, :]
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * Xb.size(0)
        if batch_idx % 10 == 0:
            print(f"  [Batch {batch_idx}] Loss: {loss.item():.6f}")

    train_loss /= train_size

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            pred = model(Xb.to(device))[:, -1, :]
            loss = criterion(pred, yb.to(device))
            val_loss += loss.item() * Xb.size(0)
    val_loss /= val_size

    print(f"[EPOCH {epoch+1:03d}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        print(f"  ✅ New best model (val loss improved from {best_val_loss:.6f} → {val_loss:.6f})")
        best_val_loss = val_loss
        best_model = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"  ⚠️  No improvement. ({epochs_no_improve}/{patience})")
        if epochs_no_improve >= patience:
            print(f"[EARLY STOP] No improvement for {patience} epochs. Stopping training.")
            break

# === Save model
print("[STEP 5] Saving best model...")
torch.save(best_model, "trained_model_tcn.pt")
print("[DONE] Model saved to trained_model_tcn.pt")
