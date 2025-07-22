import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler  # ADDED

# === Constants ===
device = torch.device("cpu")
seq_length = 100
batch_size = 1024
hidden_size = 128
num_layers = 4
learning_rate = 0.002
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
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.tcn(x)
        x = self.controller(x)
        return x.transpose(1, 2)  # (batch, seq, output)

# === Dataset ===
class MonolithicSequenceDataset(Dataset):
    def __init__(self, path, seq_length):
        print(f"[DATASET] Loading: {path}")
        d = joblib.load(path, mmap_mode='r')
        self.X = d["X"]
        self.y = d["y"]
        self.seq_length = seq_length
        self.length = len(self.X) - seq_length
        print(f"[DATASET] X shape: {self.X.shape} | y shape: {self.y.shape}")
        print(f"[DATASET] Using {self.length:,} sequences of length {seq_length}")

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x_seq = self.X[i:i+self.seq_length]
        y_next = self.y[i+self.seq_length]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_next, dtype=torch.float32)

# === Main ===
if __name__ == "__main__":
    dataset = MonolithicSequenceDataset("TCN_monolithic_sorted.pkl", seq_length)
    val_size = min(int(0.01 * len(dataset)), 100_000)
    train_size = len(dataset) - val_size
    TRAIN_LIMIT = 1_000_000
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    if TRAIN_LIMIT < train_size:
        print(f"[INFO] Capping training data to {TRAIN_LIMIT:,} samples...")
    train_data = torch.utils.data.Subset(train_data, range(TRAIN_LIMIT))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[INFO] Train samples: {len(train_data):,} | Val samples: {len(val_data):,}")

    # Load y-scaler for inverse-transforming validation
    scaler_y = joblib.load("scaler_tcn_y.pkl")

    # Initialize model
    sample_X, sample_y = dataset[0]
    model = TCN_MANN(input_size=sample_X.shape[1],
                     hidden_size=hidden_size,
                     output_size=sample_y.shape[0],
                     num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

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
                pred = model(Xb.to(device))[:, -1, :].cpu().numpy()
                y_true = yb.cpu().numpy()

                # Inverse transform to physical units
                pred_real = scaler_y.inverse_transform(pred)
                y_true_real = scaler_y.inverse_transform(y_true)

                loss = np.mean((pred_real - y_true_real)**2)
                val_loss += loss * Xb.size(0)

        val_loss /= val_size
        print(f"[EPOCH {epoch+1:03d}] Train Loss: {train_loss:.6f} | Val Loss (real units): {val_loss:.6f}")

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

    print("[STEP] Saving best model...")
    torch.save(best_model, "trained_model_tcn.pt")
    print("[DONE] Model saved to trained_model_tcn.pt")
