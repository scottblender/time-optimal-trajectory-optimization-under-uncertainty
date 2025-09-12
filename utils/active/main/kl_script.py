import os
import re
import numpy as np
import csv
import joblib
from glob import glob

def extract_bundle_idx(path):
    match = re.search(r"bundle_(\d+)", path)
    return int(match.group(1)) if match else -1

def read_runtime(stride_minutes):
    path = f"stride_{stride_minutes}min/runtime.txt"
    if os.path.exists(path):
        try:
            return float(open(path).read().strip())
        except:
            return None
    return None

def extract_max_position_trace(cov):
    try:
        pos_cov = cov[:3, :3]
        eigvals = np.linalg.eigvalsh(pos_cov)
        return float(np.max(eigvals))
    except:
        return np.nan

def read_mahalanobis(segment_dir):
    path = os.path.join(segment_dir, "mahalanobis_distances.txt")
    if os.path.exists(path):
        try:
            vals = np.loadtxt(path)
            if vals.size == 0:
                return np.nan, np.nan
            return np.max(vals), np.mean(vals)
        except:
            return np.nan, np.nan
    return np.nan, np.nan

def estimate_data_size(bundle_file):
    try:
        data = joblib.load(bundle_file)
        backTspan = data["backTspan"]
        r_b = data["r_bundles"][::-1]
        v_b = data["v_bundles"][::-1]
        m_b = data["mass_bundles"][::-1]
        num_time_steps = len(backTspan)
        num_bundles = r_b.shape[2]
        num_sigmas = 15
        num_columns = 24
        rows_per_segment = 10 * 20  # substeps * evals_per_substep
        total_rows = num_time_steps * num_bundles * num_sigmas * rows_per_segment
        return total_rows * num_columns
    except Exception as e:
        print(f"[WARN] Failed to estimate data size for {bundle_file}: {e}")
        return np.nan

def main():
    stride_dirs = sorted(glob("stride_*min"))
    summary_rows = []
    grouped_by_stride = {}

    for stride_dir in stride_dirs:
        match = re.search(r"stride_(\d+)min", stride_dir)
        if not match:
            continue
        stride_minutes = int(match.group(1))
        runtime = read_runtime(stride_minutes)
        bundle_file = os.path.join(stride_dir, f"bundle_data_{stride_minutes}min.pkl")
        est_data_size = estimate_data_size(bundle_file)

        segment_dirs = glob(os.path.join(stride_dir, "segment_*_bundle_*"))
        for segment_dir in segment_dirs:
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

            name_match = re.search(r"segment_(max|min)_(farthest|closest)_bundle", segment_dir)
            if not name_match:
                print(f"[WARN] Could not parse segment label in: {segment_dir}")
                continue

            seg_type, mode = name_match.groups()
            segment_id = f"{seg_type}_{mode}"
            bundle_idx = extract_bundle_idx(segment_dir)

            cov_sigma = np.loadtxt(sig_path) if os.path.exists(sig_path) else np.full((7, 7), np.nan)
            cov_mc = np.loadtxt(mc_path) if os.path.exists(mc_path) else np.full((7, 7), np.nan)
            max_tr_sigma = extract_max_position_trace(cov_sigma)
            max_tr_mc = extract_max_position_trace(cov_mc)
            cov_ratio = max_tr_sigma / max_tr_mc 
            max_mahal, mean_mahal = read_mahalanobis(segment_dir)

            row = {
                "stride_minutes": stride_minutes,
                "segment": segment_id,
                "bundle_idx": bundle_idx,
                "max_KL": np.max(kl_vals),
                "mean_KL": np.mean(kl_vals),
                "final_KL": kl_vals[-1],
                "max_tr_sigma": max_tr_sigma,
                "max_tr_mc": max_tr_mc,
                "cov_ratio": cov_ratio,
                "max_mahalanobis": max_mahal,
                "mean_mahalanobis": mean_mahal,
                "runtime_sec": runtime,
                "estimated_data_size": est_data_size
            }

            summary_rows.append(row)
            if stride_minutes not in grouped_by_stride:
                grouped_by_stride[stride_minutes] = {}
            grouped_by_stride[stride_minutes][segment_id] = row

    best_stride = None
    best_score = float("inf")
    scored_rows = []

    required_segments = ["max_farthest", "max_closest", "min_farthest", "min_closest"]

    for stride, segs in grouped_by_stride.items():
        if not all(seg in segs for seg in required_segments):
            continue

        # Pull all four
        rows = [segs[seg] for seg in required_segments]

        # Filter out any disqualifying KL or covariance trace
        if any(row["max_KL"] > 0.2 or row["max_tr_sigma"] > 2.0 for row in rows):
            continue

        max_kl = max(row["max_KL"] for row in rows)
        mean_kl = max(row["mean_KL"] for row in rows)
        max_cov = max(row["max_tr_sigma"] for row in rows)
        cov_mismatch = max(abs(np.log(row["cov_ratio"])) for row in rows if row["cov_ratio"] > 0)
        max_mahal = max(row["max_mahalanobis"] for row in rows)
        mean_mahal = max(row["mean_mahalanobis"] for row in rows)
        data_size = max(row["estimated_data_size"] for row in rows)

        score = (
            0.5 * max_kl +
            1.0 * mean_kl +
            8.0 * max_cov +
            1.0 * cov_mismatch +
            1.5 * max_mahal +
            0.5 * mean_mahal +
            1e-9 * data_size
        )

        scored_rows.append({
            "stride_minutes": stride,
            "mean_KL": mean_kl,
            "max_KL": max_kl,
            "max_cov": max_cov,
            "cov_mismatch": cov_mismatch,
            "max_mahalanobis": max_mahal,
            "mean_mahalanobis": mean_mahal,
            "estimated_data_size": data_size,
            "score": score
        })

        if score < best_score:
            best_score = score
            best_stride = stride

    # === Write full segment-wise summary ===
    summary_rows.sort(key=lambda x: (x["stride_minutes"], x["segment"]))
    fieldnames = list(summary_rows[0].keys()) if summary_rows else []

    with open("kl_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # === Write stride score summary ===
    with open("kl_score_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "stride_minutes", "mean_KL", "max_KL", "max_cov", "cov_mismatch",
            "max_mahalanobis", "mean_mahalanobis", "estimated_data_size", "score"
        ])
        writer.writeheader()
        for row in scored_rows:
            writer.writerow(row)

    if best_stride is not None:
        print(f"[BEST] stride_minutes = {best_stride} (score = {best_score:.4f})")
    else:
        print("[INFO] No stride passed threshold. Adjust KL or covariance bounds.")

if __name__ == "__main__":
    main()
