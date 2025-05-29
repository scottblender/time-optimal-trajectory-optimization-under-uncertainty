import numpy as np
import pandas as pd

def compute_sensitivity_metrics_all_sigmas(sensitivity_df):
    """
    Computes total trajectory MSE and final position deviation for each sigma_idx,
    comparing plus3 and minus3 against the mean trajectory.
    
    Returns:
        DataFrame with columns:
        ['sigma_idx', 'variant', 'trajectory_mse', 'final_pos_deviation_km']
    """
    all_metrics = []

    unique_sigmas = sensitivity_df["sigma_idx"].unique()
    for sigma_idx in sorted(unique_sigmas):
        df_mean = sensitivity_df[
            (sensitivity_df["sigma_idx"] == sigma_idx) &
            (sensitivity_df["lam_type"] == "mean")
        ].sort_values("time")

        for variant in ["plus3", "minus3"]:
            df_var = sensitivity_df[
                (sensitivity_df["sigma_idx"] == sigma_idx) &
                (sensitivity_df["lam_type"] == variant)
            ].sort_values("time")

            if df_mean.shape[0] != df_var.shape[0]:
                print(f"Skipping sigma_idx {sigma_idx}, variant {variant} due to mismatched length.")
                continue

            pos_mean = df_mean[["x", "y", "z"]].values
            pos_var = df_var[["x", "y", "z"]].values

            # Compute total trajectory MSE across all timesteps
            trajectory_mse = np.mean(np.linalg.norm(pos_mean - pos_var, axis=1)**2)

            # Final position deviation in km
            final_dev = np.linalg.norm(pos_mean[-1] - pos_var[-1])

            all_metrics.append({
                "sigma_idx": sigma_idx,
                "variant": variant,
                "trajectory_mse": trajectory_mse,
                "final_pos_deviation_km": final_dev
            })

    return pd.DataFrame(all_metrics)
