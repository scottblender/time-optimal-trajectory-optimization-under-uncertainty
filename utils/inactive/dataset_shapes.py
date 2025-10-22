import glob
import joblib
import os

stride_files = sorted(glob.glob("sweep_stride_*_config_*_data.pkl"))

print("=== Dataset Shapes ===")
for file in stride_files:
    data = joblib.load(file)
    X, y = data["X"], data["y"]
    print(f"{os.path.basename(file)}: X = {X.shape}, y = {y.shape}")
