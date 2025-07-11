import os
import sys
import joblib
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import generate_sigma_points
from solve_trajectories import solve_trajectories_with_covariance_parallel_with_progress


def main():
    print("Loading bundle data...")
    data = joblib.load("bundle_data.pkl")
    r_bundles = data["r_bundles"][::-1]
    v_bundles = data["v_bundles"][::-1]
    mass_bundles = data["mass_bundles"][::-1]
    new_lam_bundles = data["new_lam_bundles"][::-1]
    backTspan = data["backTspan"]
    mu, F, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    print("Bundle data loaded.")

    num_bundles = r_bundles.shape[2]
    nsd = 7
    beta, kappa = 2, 3 - nsd
    alpha = np.sqrt(9 / (nsd + kappa))
    bundle_indices_full = np.arange(num_bundles)

    uncertainty_configs = [
        {"name": "baseline", "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.0001]])},
        {"name": "high_pos", "P_pos": np.eye(3) * 0.1,  "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.0001]])},
        {"name": "high_vel", "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.001,  "P_mass": np.array([[0.0001]])},
        {"name": "high_mass", "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.01]])},
        {"name": "high_pos_vel", "P_pos": np.eye(3) * 0.1, "P_vel": np.eye(3) * 0.001, "P_mass": np.array([[0.0001]])},
        {"name": "high_pos_mass", "P_pos": np.eye(3) * 0.1, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.01]])},
        {"name": "high_vel_mass", "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.001, "P_mass": np.array([[0.01]])},
        {"name": "high_all", "P_pos": np.eye(3) * 0.1, "P_vel": np.eye(3) * 0.001, "P_mass": np.array([[0.01]])},
    ]

    time_strides = np.arange(1, 5)

    # === Run full baseline propagation in batches ===
    batch_size = 5
    full_time_steps = np.arange(len(backTspan))
    num_time_steps_full = len(full_time_steps)
    baseline_config = uncertainty_configs[0]

    for batch_start in range(0, num_bundles, batch_size):
        batch_end = min(batch_start + batch_size, num_bundles)
        print(f"\n=== Running baseline (stride=1) | Bundles {batch_start} to {batch_end-1} ===")

        # Slice bundles
        r_batch = r_bundles[:, :, batch_start:batch_end]
        v_batch = v_bundles[:, :, batch_start:batch_end]
        m_batch = mass_bundles[:, batch_start:batch_end]
        lam_batch = new_lam_bundles[:, :, batch_start:batch_end]

        sigmas_combined, _, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
            nsd=nsd, alpha=alpha, beta=beta, kappa=kappa,
            P_pos=baseline_config["P_pos"],
            P_vel=baseline_config["P_vel"],
            P_mass=baseline_config["P_mass"],
            time_steps=full_time_steps,
            r_bundles=r_batch,
            v_bundles=v_batch,
            mass_bundles=m_batch,
            num_workers=os.cpu_count()
        )

        trajectories, P_hist, means_hist, X, y = solve_trajectories_with_covariance_parallel_with_progress(
            backTspan, full_time_steps, num_time_steps_full, batch_end - batch_start,
            sigmas_combined, lam_batch, m_batch,
            mu, F, c, m0, g0, Wm, Wc,
            num_workers=os.cpu_count()
        )

        out_dir = f"baseline_stride_1/batch_{batch_start}_{batch_end-1}"
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump({"X": X, "y": y, "Wm": Wm, "Wc": Wc}, os.path.join(out_dir, "data.pkl"))
        print(f"Saved: {out_dir}/data.pkl")

    # === Run all time strides and configs for bundle 0 only ===
    print("\n=== Running time stride/config sweep for bundle 0 only ===")
    r0 = r_bundles[:, :, 0][:, :, np.newaxis]
    v0 = v_bundles[:, :, 0][:, :, np.newaxis]
    m0s = mass_bundles[:, 0][:, np.newaxis]
    lam0 = new_lam_bundles[:, :, 0][:, :, np.newaxis]

    for stride in time_strides:
        time_steps = np.arange(0, len(backTspan), step=stride)
        num_time_steps = len(time_steps)

        for config in uncertainty_configs:
            name = config["name"]
            print(f"\n=== Running: stride={stride}, config={name} (bundle 0) ===")
            out_dir = f"sweep_bundle0_stride_{stride}_config_{name}"
            os.makedirs(out_dir, exist_ok=True)

            sigmas_combined, _, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
                nsd=nsd, alpha=alpha, beta=beta, kappa=kappa,
                P_pos=config["P_pos"], P_vel=config["P_vel"], P_mass=config["P_mass"],
                time_steps=time_steps,
                r_bundles=r0, v_bundles=v0, mass_bundles=m0s,
                num_workers=os.cpu_count()
            )

            trajectories, P_hist, means_hist, X, y = solve_trajectories_with_covariance_parallel_with_progress(
                backTspan, time_steps, num_time_steps, 1,
                sigmas_combined, lam0, m0s,
                mu, F, c, m0, g0, Wm, Wc,
                num_workers=os.cpu_count()
            )

            joblib.dump({"X": X, "y": y, "Wm": Wm, "Wc": Wc}, os.path.join(out_dir, "data.pkl"))
            print(f"Saved: {out_dir}/data.pkl")

    print("\n All sweep and baseline batches completed.")


if __name__ == "__main__":
    main()
