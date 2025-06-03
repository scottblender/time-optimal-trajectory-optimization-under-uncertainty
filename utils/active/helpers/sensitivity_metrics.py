import numpy as np
import pandas as pd

def compute_aggregate_metrics(sensitivity_df):
    results = []
    pos_columns = ["x", "y", "z"]

    for label in ["mean", "plus3", "minus3"]:
        df_label = sensitivity_df[sensitivity_df["lam_type"] == label]

        # Reference trajectory: sigma 0
        df_ref = df_label[df_label["sigma_idx"] == 0].sort_values("time")[["time"] + pos_columns]
        df_ref = df_ref.rename(columns={col: f"ref_{col}" for col in pos_columns})

        mse_list = []
        final_dev_list = []

        for sigma_idx in df_label["sigma_idx"].unique():
            if sigma_idx == 0:
                continue  # skip self-comparison

            df_sigma = df_label[df_label["sigma_idx"] == sigma_idx].sort_values("time")[["time"] + pos_columns]
            merged = pd.merge(df_sigma, df_ref, on="time", how="inner")

            if merged.shape[0] == 0:
                print(f"Skipping sigma {sigma_idx} ({label}) due to empty merge.")
                continue

            pos_sigma = merged[pos_columns].values
            pos_ref = merged[[f"ref_{col}" for col in pos_columns]].values

            squared_dists = np.linalg.norm(pos_sigma - pos_ref, axis=1) ** 2
            final_dev = np.linalg.norm(pos_sigma[-1] - pos_ref[-1])
            mse = np.mean(squared_dists)

            results.append({
                "lam_type": label,
                "sigma_idx": sigma_idx,
                "mse": mse,
                "final_pos_deviation_km": final_dev
            })

            mse_list.append(mse)
            final_dev_list.append(final_dev)

        # Add average over all nonzero sigmas
        if mse_list:
            results.append({
                "lam_type": label,
                "sigma_idx": "avg",
                "mse": np.mean(mse_list),
                "final_pos_deviation_km": np.mean(final_dev_list)
            })

    return pd.DataFrame(results)
