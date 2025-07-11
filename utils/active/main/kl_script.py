import os
import re
import numpy as np
import csv
from glob import glob

def extract_bundle_idx(path):
    match = re.search(r"bundle_(\d+)", path)
    return int(match.group(1)) if match else -1

def read_runtime(stride_minutes):
    runtime_path = f"stride_{stride_minutes}min/runtime.txt"
    if os.path.exists(runtime_path):
        try:
            return float(open(runtime_path).read().strip())
        except:
            return None
    return None

def extract_max_position_trace(cov):
    # Cov shape: (7,7) → we take top-left 3x3 position covariance
    try:
        pos_cov = cov[:3, :3]
        eigvals = np.linalg.eigvalsh(pos_cov)
        return float(np.max(eigvals))
    except:
        return np.nan

def main():
    stride_dirs = sorted(glob("stride_*min"))
    summary_rows = []

    for stride_dir in stride_dirs:
        stride_match = re.search(r"stride_(\d+)min", stride_dir)
        if not stride_match:
            continue
        stride_minutes = int(stride_match.group(1))
        runtime = read_runtime(stride_minutes)

        for segment_dir in glob(f"{stride_dir}/segment_*"):
            kl_path = os.path.join(segment_dir, "kl_divergence.txt")
            sig_path = os.path.join(segment_dir, "cov_sigma_final.txt")
            mc_path = os.path.join(segment_dir, "cov_mc_final.txt")

            if not os.path.exists(kl_path):
                continue
            try:
                kl_vals = np.loadtxt(kl_path)
            except Exception as e:
                print(f"[WARN] Failed to load {kl_path}: {e}")
                continue

            label = "max" if "max" in segment_dir else "min"
            bundle_idx = extract_bundle_idx(segment_dir)

            cov_sigma = np.loadtxt(sig_path) if os.path.exists(sig_path) else np.full((7, 7), np.nan)
            cov_mc = np.loadtxt(mc_path) if os.path.exists(mc_path) else np.full((7, 7), np.nan)

            max_tr_sigma = extract_max_position_trace(cov_sigma)
            max_tr_mc = extract_max_position_trace(cov_mc)
            cov_ratio = max_tr_sigma / max_tr_mc if max_tr_mc > 1e-12 else np.nan

            summary_rows.append({
                "stride_minutes": stride_minutes,
                "segment": label,
                "bundle_idx": bundle_idx,
                "max_KL": np.max(kl_vals),
                "mean_KL": np.mean(kl_vals),
                "final_KL": kl_vals[-1],
                "max_tr_sigma": max_tr_sigma,
                "max_tr_mc": max_tr_mc,
                "cov_ratio": cov_ratio,
                "runtime_sec": runtime
            })

    # Sort and write
    summary_rows.sort(key=lambda x: (x["stride_minutes"], x["segment"]))
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []

    with open("kl_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"Summary written to kl_summary.csv with {len(summary_rows)} entries.")

if __name__ == "__main__":
    main()
