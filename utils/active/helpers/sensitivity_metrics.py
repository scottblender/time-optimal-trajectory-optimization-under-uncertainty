import numpy as np

def compute_sensitivity_metrics(sensitivity_df, sigma_idx=0):
    """
    Computes max MSE over time and final position deviation for plus3 and minus3 vs mean.
    """
    metrics = {}
    for variant in ["plus3", "minus3"]:
        df_mean = sensitivity_df[
            (sensitivity_df["sigma_idx"] == sigma_idx) &
            (sensitivity_df["lam_type"] == "mean")
        ].sort_values("time")

        df_var = sensitivity_df[
            (sensitivity_df["sigma_idx"] == sigma_idx) &
            (sensitivity_df["lam_type"] == variant)
        ].sort_values("time")

        # Align time and ensure same length
        if df_mean.shape[0] != df_var.shape[0]:
            raise ValueError(f"Mismatch in length for mean vs {variant} trajectories")

        # Compute MSE over position
        pos_mean = df_mean[["x", "y", "z"]].values
        pos_var = df_var[["x", "y", "z"]].values
        mse_per_step = np.mean((pos_mean - pos_var)**2, axis=1)
        max_mse = np.max(mse_per_step)

        # Compute final position deviation
        final_dev = np.linalg.norm(pos_mean[-1] - pos_var[-1])

        metrics[variant] = {
            "max_mse": max_mse,
            "final_pos_deviation_km": final_dev
        }

    return metrics
