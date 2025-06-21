import numpy as np
import pandas as pd

def compute_aggregate_metrics(sensitivity_df, output_csv="sensitivity_metrics.csv"):
    results = []
    pos_columns = ["x", "y", "z"]

    for label in ["mean", "plus3", "minus3"]:
        df_label = sensitivity_df[sensitivity_df["lam_type"] == label]

        # Reference: sigma 0
        df_ref = df_label[df_label["sigma_idx"] == 0].sort_values("time")[["time"] + pos_columns]
        df_ref = df_ref.rename(columns={col: f"ref_{col}" for col in pos_columns})

        mse_x_list = []
        mse_y_list = []
        mse_z_list = []
        final_dev_list = []

        for sigma_idx in df_label["sigma_idx"].unique():
            if sigma_idx == 0:
                continue

            df_sigma = df_label[df_label["sigma_idx"] == sigma_idx].sort_values("time")[["time"] + pos_columns]
            merged = pd.merge(df_sigma, df_ref, on="time", how="inner")

            if merged.shape[0] == 0:
                print(f"Skipping sigma {sigma_idx} ({label}) due to empty merge.")
                continue

            pos_sigma = merged[pos_columns].values
            pos_ref = merged[[f"ref_{col}" for col in pos_columns]].values

            mse_x = np.mean((pos_sigma[:, 0] - pos_ref[:, 0]) ** 2)
            mse_y = np.mean((pos_sigma[:, 1] - pos_ref[:, 1]) ** 2)
            mse_z = np.mean((pos_sigma[:, 2] - pos_ref[:, 2]) ** 2)
            final_dev = np.linalg.norm(pos_sigma[-1] - pos_ref[-1])

            results.append({
                "lam_type": label,
                "sigma_idx": sigma_idx,
                "x mse": mse_x,
                "y mse": mse_y,
                "z mse": mse_z,
                "final position deviation": final_dev
            })

            mse_x_list.append(mse_x)
            mse_y_list.append(mse_y)
            mse_z_list.append(mse_z)
            final_dev_list.append(final_dev)

        # Add mean row
        if mse_x_list:
            results.append({
                "lam_type": label,
                "sigma_idx": "ALL",
                "x mse": np.mean(mse_x_list),
                "y mse": np.mean(mse_y_list),
                "z mse": np.mean(mse_z_list),
                "final position deviation": np.mean(final_dev_list)
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    return df_results
