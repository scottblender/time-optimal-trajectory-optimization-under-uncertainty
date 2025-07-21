import os
import glob
import joblib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

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

# === Dataset ===
class SequenceByBundleDataset(Dataset):
    def __init__(self, sequence_path, seq_length):
        print("[DATASET] Loading preprocessed bundle-sigma sequences...")
        self.sequences = joblib.load(sequence_path, mmap_mode='r')
        self.seq_length = seq_length
        self.samples = []

        print("[DATASET] Indexing sequences...")
        for key, data in tqdm(self.sequences.items()):
            X_seq, y_seq = data["X"], data["y"]
            for i in range(len(X_seq) - seq_length):
                self.samples.append((key, i))  # store pointer to original array

        print(f"[DATASET] Total sequence samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (bundle, sigma), i = self.samples[idx]
        X_seq = self.sequences[(bundle, sigma)]["X"][i:i+self.seq_length]
        y = self.sequences[(bundle, sigma)]["y"][i+self.seq_length]
        return torch.tensor(X_seq, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# === Main ===
if __name__ == "__main__":
    print("[STEP 1] Loading time index from file...")
    index_data = joblib.load("time_index_tcn.pkl")
    time_to_rows_X = defaultdict(list, index_data["X_map"])
    time_to_rows_y = defaultdict(list, index_data["y_map"])
    time_vals = index_data["time_vals"]

    total_rows = sum(len(v) for v in time_to_rows_X.values())
    print(f"[INFO] Total unique time steps: {len(time_vals)}")
    print(f"[INFO] Total input rows loaded: {total_rows:,}")

    print("[STEP 2] Loading dataset...")
    dataset = SequenceByBundleDataset("bundle_sigma_sequences.pkl", seq_length)
    val_size = min(int(0.01 * len(dataset)), 100_000)
    train_size = len(dataset) - val_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[INFO] Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    print("[STEP 3] Initializing model...")
    sample_X, sample_y = dataset[0]
    model = TCN_MANN(input_size=sample_X.shape[1],
                     hidden_size=hidden_size,
                     output_size=sample_y.shape[0],
                     num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

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
            best_val_loss = val_loss
            best_model = model.state_dict()
            epochs_no_improve = 0
            print(f"  ✅ New best model (val loss improved)")
        else:
            epochs_no_improve += 1
            print(f"  ⚠️  No improvement. ({epochs_no_improve}/{patience})")
            if epochs_no_improve >= patience:
                print(f"[EARLY STOP] No improvement for {patience} epochs.")
                break

    print("[STEP 5] Saving best model...")
    torch.save(best_model, "trained_model_tcn.pt")
    print("[DONE] Model saved to trained_model_tcn.pt")
