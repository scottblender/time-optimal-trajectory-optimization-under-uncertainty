import joblib
import os
import sys
import numpy as np
import csv
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import compute_nominal_trajectory_params
import compute_bundle_trajectory_params

def stream_initial_csv_for_multiple_bundles(
    backTspan, r_bundles, v_bundles, mass_bundles, new_lam_bundles,
    bundle_indices, output_filename
):
    backTspan_rev = backTspan[::-1]
    r_bundles_rev = r_bundles[::-1]
    v_bundles_rev = v_bundles[::-1]
    m_bundles_rev = mass_bundles[::-1]
    lam_bundles_rev = new_lam_bundles[::-1]

    header = [
        "time", "x", "y", "z", "vx", "vy", "vz", "mass",
        "lam0", "lam1", "lam2", "lam3", "lam4", "lam5", "lam6",
        "bundle_index"
    ]

    total_rows = len(backTspan_rev) * len(bundle_indices)

    with open(output_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        progress = tqdm(total=total_rows, desc=f"Writing {output_filename}")
        for bundle_index in bundle_indices:
            r_b = r_bundles_rev[:, :, bundle_index]
            v_b = v_bundles_rev[:, :, bundle_index]
            m_b = m_bundles_rev[:, bundle_index]
            lam_b = lam_bundles_rev[:, :, bundle_index]

            for t_idx in range(len(backTspan_rev)):
                row = [
                    backTspan_rev[t_idx],
                    *r_b[t_idx], *v_b[t_idx],
                    m_b[t_idx],
                    *lam_b[t_idx],
                    bundle_index
                ]
                writer.writerow(row)
                progress.update(1)
        progress.close()

    print(f"Saved initial bundle states to {output_filename}")


def main():
    mu_s = 132712 * 10**6 * 1e9
    p_sol, tfound, s0, mu, F, c, m0, g0, R_V_0, V_V_0, DU, TU = compute_nominal_trajectory_params.compute_nominal_trajectory_params()

    num_bundles = 50
    time_resolution_list = np.arange(1000, 6001, 500)  # in minutes

    for time_resolution_minutes in time_resolution_list:
        print(f"\n[INFO] Generating bundle data at {time_resolution_minutes} min resolution...")

        r_tr, v_tr, mass_tr, S_bundles, r_bundles, v_bundles, new_lam_bundles, mass_bundles, backTspan = \
            compute_bundle_trajectory_params.compute_bundle_trajectory_params(
                p_sol, s0, tfound, mu, F, c, m0, g0, R_V_0, V_V_0, DU, TU,
                num_bundles, time_resolution_minutes
            )

        out_dir = f"stride_{time_resolution_minutes}min"
        os.makedirs(out_dir, exist_ok=True)

        # Save bundle data
        out_pkl = os.path.join(out_dir, f"bundle_data_{time_resolution_minutes}min.pkl")
        joblib.dump({
            "r_tr": r_tr,
            "v_tr": v_tr,
            "mass_tr": mass_tr,
            "S_bundles": S_bundles,
            "r_bundles": r_bundles,
            "v_bundles": v_bundles,
            "new_lam_bundles": new_lam_bundles,
            "mass_bundles": mass_bundles,
            "backTspan": backTspan,
            "mu": mu, "F": F, "c": c, "m0": m0, "g0": g0
        }, out_pkl)
        print(f"Saved: {out_pkl}")

        # Save initial CSVs for bundle indices [0, 24]
        out_csv = os.path.join(out_dir, f"initial_bundles_{time_resolution_minutes}min.csv")
        stream_initial_csv_for_multiple_bundles(
            backTspan, r_bundles, v_bundles, mass_bundles, new_lam_bundles,
            bundle_indices=list(range(num_bundles - 25)),
            output_filename=out_csv
        )


if __name__ == "__main__":
    main()
