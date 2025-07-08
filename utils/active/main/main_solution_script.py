import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
import generate_sigma_points
from solve_trajectories import solve_trajectories_with_covariance_parallel_with_progress as solve_trajectories_with_covariance_parallel
from generate_monte_carlo_trajectories import generate_monte_carlo_trajectories_parallel as generate_monte_carlo_trajectories

def main():
    print("Loading bundle data...")
    data = joblib.load("bundle_data.pkl")
    r_bundles = data["r_bundles"]
    v_bundles = data["v_bundles"]
    mass_bundles = data["mass_bundles"]
    new_lam_bundles = data["new_lam_bundles"]
    backTspan = data["backTspan"]
    mu, F, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    print("Bundle data loaded.")

    num_bundles = r_bundles.shape[2]
    r_b = r_bundles[::-1]
    v_b = v_bundles[::-1]
    m_b = mass_bundles[::-1]
    lam_b = new_lam_bundles[::-1]

    nsd = 7
    beta, kappa = 2, float(3 - nsd)
    alpha = np.sqrt(9 / (nsd + kappa))

    uncertainty_configs = [
        {"name": "baseline", "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.0001]])},
        {"name": "high_pos", "P_pos": np.eye(3) * 0.1, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.0001]])},
        {"name": "high_vel", "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.001, "P_mass": np.array([[0.0001]])},
        {"name": "high_mass", "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.01]])},
        {"name": "high_pos_vel", "P_pos": np.eye(3) * 0.1, "P_vel": np.eye(3) * 0.001, "P_mass": np.array([[0.0001]])},
        {"name": "high_pos_mass", "P_pos": np.eye(3) * 0.1, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.01]])},
        {"name": "high_vel_mass", "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.001, "P_mass": np.array([[0.01]])},
        {"name": "high_all", "P_pos": np.eye(3) * 0.1, "P_vel": np.eye(3) * 0.001, "P_mass": np.array([[0.01]])},
    ]

    time_strides = np.arange(1, 9)

    bundle_widths = np.linalg.norm(np.max(r_b, axis=2) - np.min(r_b, axis=2), axis=1)
    max_idx = np.argmax(bundle_widths)
    forwardTspan = np.flip(backTspan)
    max_t_idx = max_idx if max_idx < len(forwardTspan) - 1 else len(forwardTspan) - 2
    max_time_steps = np.array([max_t_idx, max_t_idx + 1])

    bundle_idx_for_diag = random.randint(0, num_bundles - 1)

    for stride in time_strides:
        full_time_steps = np.arange(0, len(backTspan), step=stride)
        desired_length = len(backTspan)

        for config in uncertainty_configs:
            name = config["name"]
            print(f"\n=== Running stride {stride}, distribution {name} ===")

            if stride == 1 and name == "baseline":
                time_steps = full_time_steps[:desired_length]
                num_time_steps = len(time_steps)
            else:
                time_steps = max_time_steps
                num_time_steps = 2

            sigmas_combined, P_combined, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
                nsd=nsd, alpha=alpha, beta=beta, kappa=kappa,
                P_pos=config["P_pos"], P_vel=config["P_vel"], P_mass=config["P_mass"],
                num_time_steps=num_time_steps, backTspan=backTspan,
                r_bundles=r_b, v_bundles=v_b, mass_bundles=m_b,
                num_workers=os.cpu_count()
            )

            output_dir = f"results_stride_{stride}_config_{name}"

            trajectories, P_hist, means_hist, X, y, time_vector = solve_trajectories_with_covariance_parallel(
                backTspan, time_steps, num_time_steps, num_bundles,
                sigmas_combined, lam_b, m_b, mu, F, c, m0, g0, Wm, Wc,
                num_workers=os.cpu_count(), output_dir=output_dir
            )

            out_prefix = f"sweep_stride_{stride}_config_{name}"
            joblib.dump({
                "X": X, "y": y, "trajectories": trajectories,
                "P_combined_history": P_hist, "means_history": means_hist,
                "Wm": Wm, "Wc": Wc
            }, f"{out_prefix}_data.pkl")
            print(f"Saved propagated results for stride={stride}, config={name}")

            if stride == 1 and name == "baseline":
                t_k = forwardTspan[max_t_idx + 1]
                X_tk1 = X[(np.isclose(X[:, 0], t_k)) & (X[:, -2] == bundle_idx_for_diag)]
                sigma0_row = next((row for row in X_tk1 if row[-1] == 0 and np.allclose(row[8:15], 0.0)), None)

                if sigma0_row is not None:
                    p, f, g, h, k, L = sigma0_row[1:7]
                    r_ref, _ = mee2rv(np.array([p]), np.array([f]), np.array([g]),
                                      np.array([h]), np.array([k]), np.array([L]), mu)
                    r_ref = r_ref[0]

                    bundle_mask = (X[:, -2] == bundle_idx_for_diag)
                    time_diff = np.abs(time_vector[bundle_mask].flatten() - t_k)
                    idx_in_bundle = np.argmin(time_diff)
                    num_sigmas = sigmas_combined.shape[1]
                    true_time_idx = idx_in_bundle // num_sigmas

                    P_final = P_hist[bundle_idx_for_diag][true_time_idx][:3, :3]
                    inv_cov = inv(P_final)

                    print(f"\n=== Mahalanobis Diagnostic at t = {t_k:.6f} (bundle={bundle_idx_for_diag}) ===")
                    for sigma_idx in range(num_sigmas):
                        row = next((r for r in X_tk1 if r[-1] == sigma_idx and not np.allclose(r[8:15], 0.0)), None)
                        if row is None:
                            print(f"Sigma {sigma_idx:2d}: [not found]")
                            continue
                        p, f, g, h, k, L = row[1:7]
                        r_eci, _ = mee2rv(np.array([p]), np.array([f]), np.array([g]),
                                          np.array([h]), np.array([k]), np.array([L]), mu)
                        d = mahalanobis(r_eci[0], r_ref, inv_cov)
                        print(f"Sigma {sigma_idx:2d}: Mahalanobis distance = {d:.6f}")

                    print(f"\nSigma Point Covariance Matrix at t = {t_k:.6f} (position only):")
                    print(P_final)

                print("Running Monte Carlo simulation...")
                mc_traj, mc_P_hist, mc_means_hist, mc_X, mc_y = generate_monte_carlo_trajectories(
                    backTspan=backTspan, time_steps=max_time_steps, num_time_steps=2,
                    num_bundles=1, sigmas_combined=sigmas_combined,
                    new_lam_bundles=lam_b, mu=mu, F=F, c=c, m0=m0, g0=g0,
                    num_samples=10000, num_workers=os.cpu_count()
                )

                X_bundle_single = X[(X[:, -2] == bundle_idx_for_diag) &
                                    ((X[:, 0] == forwardTspan[max_t_idx]) | (X[:, 0] == forwardTspan[max_t_idx + 1]))]

                joblib.dump({
                    "bundle_index": bundle_idx_for_diag,
                    "forwardTspan_tk": forwardTspan[max_t_idx],
                    "forwardTspan_tkp1": forwardTspan[max_t_idx + 1],
                    "mc_X": mc_X, "mc_y": mc_y, "mc_traj": mc_traj,
                    "X_bundle_single": X_bundle_single
                }, f"mc_and_bundle_{bundle_idx_for_diag}_at_t{max_t_idx}.pkl")
                print(f"Saved Monte Carlo and bundle trajectory for bundle={bundle_idx_for_diag} at t={forwardTspan[max_t_idx]:.6f}")

                mc_P_final = mc_P_hist[0, 0, -1, :3, :3]
                print(f"\nMonte Carlo Covariance Matrix at t = {t_k:.6f} (position only):")
                print(mc_P_final)

if __name__ == "__main__":
    main()
