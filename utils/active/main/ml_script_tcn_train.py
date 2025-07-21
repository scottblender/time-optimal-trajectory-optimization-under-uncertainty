import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# === Constants ===
device = torch.device("cpu")
seq_length = 100
batch_size = 512
hidden_size = 128
num_layers = 4
learning_rate = 0.001
epochs = 50
patience = 2
num_workers = 4
TRAIN_LIMIT = 10_000_000  # cap training size

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

# === Main Training Function ===
def main():
    dataset = MonolithicSequenceDataset("TCN_monolithic_sorted.pkl", seq_length)
    val_size = min(int(0.01 * len(dataset)), 100_000)
    train_size = len(dataset) - val_size
    # Split once
    train_full, val_data = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    # If needed, cap the train set directly
    if TRAIN_LIMIT < train_size:
        print(f"[INFO] Limiting training set to {TRAIN_LIMIT:,} samples...")
        train_data = torch.utils.data.Subset(train_full, list(range(TRAIN_LIMIT)))
    else:
        train_data = train_full
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    print(f"[INFO] Train samples: {len(train_data):,} | Val samples: {len(val_data):,}")

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
        for Xb, yb in tqdm(train_loader, desc="Train Batches"):
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)[:, -1, :]
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * Xb.size(0)

        train_loss /= len(train_data)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in tqdm(val_loader, desc="Val Batches"):
                pred = model(Xb.to(device))[:, -1, :]
                loss = criterion(pred, yb.to(device))
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_data)

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

    print("[STEP] Saving best model...")
    torch.save(best_model, "trained_model_tcn.pt")
    print("[DONE] Model saved to trained_model_tcn.pt")

# === Safe Entry Point ===
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
