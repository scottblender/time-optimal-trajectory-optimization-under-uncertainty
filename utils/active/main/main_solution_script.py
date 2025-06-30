import os
import sys
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
import generate_sigma_points
from solve_trajectories import solve_trajectories_with_covariance_parallel_with_progress as solve_trajectories_with_covariance_parallel
import generate_monte_carlo_trajectories

def main():
    # === Load bundle data ===
    print("Loading bundle data...")
    data = joblib.load("bundle_data.pkl")
    r_tr = data["r_tr"]
    v_tr = data["v_tr"]
    mass_tr = data["mass_tr"]
    S_bundles = data["S_bundles"]
    r_bundles = data["r_bundles"]
    v_bundles = data["v_bundles"]
    new_lam_bundles = data["new_lam_bundles"]
    mass_bundles = data["mass_bundles"]
    backTspan = data["backTspan"]
    mu, F, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    print("Bundle data loaded.")

    # === Parameters ===
    num_bundles = r_bundles.shape[2]
    r_b = r_bundles[::-1]
    v_b = v_bundles[::-1]
    m_b = mass_bundles[::-1]
    lam_b = new_lam_bundles[::-1]

    # === Sigma Point Settings ===
    nsd = 7
    beta, kappa = 2, float(3 - nsd)
    alpha = np.sqrt(9 / (nsd + kappa))

    # === Sigma Point Uncertainty Sweep Configurations ===
    uncertainty_configs = [
        {"name": "baseline",         "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.0001]])},
        {"name": "high_pos",         "P_pos": np.eye(3) * 0.1,  "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.0001]])},
        {"name": "high_vel",         "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.001,  "P_mass": np.array([[0.0001]])},
        {"name": "high_mass",        "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.01]])},
        {"name": "high_pos_vel",     "P_pos": np.eye(3) * 0.1,  "P_vel": np.eye(3) * 0.001,  "P_mass": np.array([[0.0001]])},
        {"name": "high_pos_mass",    "P_pos": np.eye(3) * 0.1,  "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.01]])},
        {"name": "high_vel_mass",    "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.001,  "P_mass": np.array([[0.01]])},
        {"name": "high_all",         "P_pos": np.eye(3) * 0.1,  "P_vel": np.eye(3) * 0.001,  "P_mass": np.array([[0.01]])},
    ]

    # === Time Stride Sweep ===
    time_strides = np.arange(1, 9)
    desired_length = len(backTspan)

    for stride in time_strides:
        time_steps = np.arange(0, len(backTspan), step=stride)[:desired_length]
        num_time_steps = len(time_steps)

        for config in uncertainty_configs:
            name = config["name"]
            print(f"\n=== Running stride {stride}, distribution {name} ===")

            sigmas_combined, P_combined, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
                nsd=nsd, alpha=alpha, beta=beta, kappa=kappa,
                P_pos=config["P_pos"], P_vel=config["P_vel"], P_mass=config["P_mass"],
                num_time_steps=num_time_steps, backTspan=backTspan,
                r_bundles=r_b, v_bundles=v_b, mass_bundles=m_b,
                num_workers=os.cpu_count()
            )

            print("Launching trajectory propagation tasks...")
            print(f"Total bundles: {num_bundles}, Time steps: {num_time_steps}, Workers: {os.cpu_count()}")
            trajectories, P_hist, means_hist, X, y = solve_trajectories_with_covariance_parallel(
                backTspan, time_steps, num_time_steps, num_bundles,
                sigmas_combined, lam_b, m_b, mu, F, c, m0, g0, Wm, Wc,
                num_workers=os.cpu_count()
            )

            out_prefix = f"sweep_stride_{stride}_config_{name}"
            joblib.dump({
                "X": X, "y": y, "trajectories": trajectories,
                "P_combined_history": P_hist, "means_history": means_hist,
                "Wm": Wm, "Wc": Wc
            }, f"{out_prefix}_data.pkl")
            print(f"Saved propagated results for stride={stride}, config={name}")

            # === Mahalanobis Diagnostic ===
            unique_times = np.unique(np.round(X[:, 0], decimals=10))
            if len(unique_times) >= 2:
                t_k1 = random.choice(unique_times[1:])
                bundle_indices = np.unique(X[:, -2]).astype(int)
                bundle_index = random.choice(bundle_indices)

                X_tk1 = X[(np.isclose(X[:, 0], t_k1)) & (X[:, -2] == bundle_index)]

                sigma0_row = next((row for row in X_tk1 if row[-1] == 0 and np.allclose(row[8:15], 0.0)), None)
                if sigma0_row is not None:
                    p, f, g, h, k, L = sigma0_row[1:7]
                    r_ref, _ = mee2rv(np.array([p]), np.array([f]), np.array([g]), np.array([h]), np.array([k]), np.array([L]), mu)
                    r_ref = r_ref[0]

                    time_idx = np.where(np.isclose(np.array(backTspan)[::-1], t_k1))[0][0]
                    P_final = P_hist[bundle_index][0][time_idx][:3, :3]
                    inv_cov = inv(P_final)

                    print(f"\n=== Mahalanobis Diagnostic at t = {t_k1:.6f} (stride={stride}, config={name}, bundle={bundle_index}) ===")
                    for sigma_idx in range(15):
                        row = next((r for r in X_tk1 if r[-1] == sigma_idx and not np.allclose(r[8:15], 0.0)), None)
                        if row is None:
                            print(f"Sigma {sigma_idx:2d}: [not found]")
                            continue
                        p, f, g, h, k, L = row[1:7]
                        r_eci, _ = mee2rv(np.array([p]), np.array([f]), np.array([g]),
                                          np.array([h]), np.array([k]), np.array([L]), mu)
                        d = mahalanobis(r_eci[0], r_ref, inv_cov)
                        print(f"Sigma {sigma_idx:2d}: Mahalanobis distance = {d:.6f}")

            if stride == 1 and name == "baseline":
                print("Running Monte Carlo simulation...")
                mc_traj, mc_P_hist, mc_means_hist, mc_X, mc_y = generate_monte_carlo_trajectories.generate_monte_carlo_trajectories(
                    backTspan=backTspan, time_steps=time_steps, num_time_steps=num_time_steps,
                    num_bundles=1, sigmas_combined=sigmas_combined,
                    new_lam_bundles=lam_b, mu=mu, F=F, c=c, m0=m0, g0=g0,
                    num_samples=1000
                )
                joblib.dump({
                    "X": mc_X, "y": mc_y, "trajectories": mc_traj,
                    "P_combined_history": mc_P_hist, "means_history": mc_means_hist
                }, f"{out_prefix}_monte_carlo.pkl")
                print("Monte Carlo results saved.")

if __name__ == "__main__":
    main()