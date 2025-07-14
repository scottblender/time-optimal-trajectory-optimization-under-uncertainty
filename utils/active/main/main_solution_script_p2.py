import os
import sys
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
import generate_sigma_points
from solve_trajectories import solve_trajectories_with_covariance_parallel_with_progress

def main():
    print("Loading bundle data...")
    stride_minutes = 4000
    bundle_path = f"stride_{stride_minutes}min/bundle_data_{stride_minutes}min.pkl"
    data = joblib.load(bundle_path)
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
    batch_size = 5
    full_time_steps = np.arange(len(backTspan))
    num_time_steps_full = len(full_time_steps)

    baseline_config = {
        "P_pos": np.eye(3) * 0.01,
        "P_vel": np.eye(3) * 0.0001,
        "P_mass": np.array([[0.0001]])
    }

    for batch_start in range(0, num_bundles, batch_size):
        batch_end = min(batch_start + batch_size, num_bundles)
        global_bundle_indices = list(range(batch_start, batch_end))

        print(f"\n=== Running baseline | Bundles {batch_start} to {batch_end - 1} ===")

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
            backTspan, full_time_steps, num_time_steps_full,
            sigmas_combined, lam_batch, m_batch,
            mu, F, c, m0, g0, Wm, Wc,
            global_bundle_indices=global_bundle_indices,
            num_workers=os.cpu_count()
        )

        out_dir = f"baseline_stride_1/batch_{batch_start}_{batch_end - 1}"
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump({"X": X, "y": y, "Wm": Wm, "Wc": Wc}, os.path.join(out_dir, "data.pkl"))
        print(f"Saved: {out_dir}/data.pkl")

        # === Check output integrity ===
        sigma0_rows = X[(X[:, -1] == 0) & np.all(np.abs(X[:, 8:15]) < 1e-12, axis=1)]
        expected = (num_time_steps_full - 1) * len(global_bundle_indices)
        print(f"[CHECK] Found {len(sigma0_rows)} appended sigma₀ rows (expected {expected})")
        if len(sigma0_rows) != expected:
            print("[WARN] Some sigma₀ rows may be missing.")

        # === Check bundle indices in X ===
        bundle_ids_in_X = set(np.unique(X[:, -2]).astype(int))
        expected_ids = set(global_bundle_indices)
        missing = expected_ids - bundle_ids_in_X
        extra = bundle_ids_in_X - expected_ids

        if missing:
            print(f"[ERROR] Missing bundle indices in X: {sorted(missing)}")
        if extra:
            print(f"[WARN] Unexpected bundle indices in X: {sorted(extra)}")
        if not missing and not extra:
            print("[CHECK] Bundle indices in X match expected global indices.")

    print("\nAll baseline batches completed.")

if __name__ == "__main__":
    main()
