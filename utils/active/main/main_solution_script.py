import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv, eigh
from tqdm import tqdm
import zarr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
import generate_sigma_points
from solve_trajectories import solve_trajectories_with_covariance_parallel_with_progress as solve_trajectories_with_covariance_parallel

def plot_3sigma_ellipsoid(ax, mean, cov, color, label):
    vals, vecs = eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    rx, ry, rz = 3 * np.sqrt(vals)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = rx * np.outer(np.cos(u), np.sin(v))
    y = ry * np.outer(np.sin(u), np.sin(v))
    z = rz * np.outer(np.ones_like(u), np.cos(v))

    ellipsoid = np.array([x.flatten(), y.flatten(), z.flatten()])
    ellipsoid = vecs @ ellipsoid
    x_e, y_e, z_e = ellipsoid + mean.reshape(3, 1)

    ax.plot_wireframe(x_e.reshape(x.shape), y_e.reshape(y.shape), z_e.reshape(z.shape),
                      color=color, alpha=0.25, label=label)

def main():
    import joblib
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
    forwardTspan = np.flip(backTspan)

    bundle_widths = np.linalg.norm(np.max(r_b, axis=2) - np.min(r_b, axis=2), axis=1)
    max_idx = np.argmax(bundle_widths)

    sorted_indices = np.argsort(bundle_widths)
    for idx in sorted_indices:
        if idx + 1 < len(forwardTspan):
            min_t_idx = idx
            break
    else:
        raise ValueError("No valid min_t_idx found with a next time step.")

    max_t_idx = min(max_idx, len(forwardTspan) - 2)

    time_strides = np.arange(1, 9)
    uncertainty_configs = [
        {"name": "baseline", "P_pos": np.eye(3) * 0.01, "P_vel": np.eye(3) * 0.0001, "P_mass": np.array([[0.0001]])},
    ]

    for stride in time_strides:
        time_steps = np.arange(0, len(backTspan), step=stride)
        num_time_steps = len(time_steps)

        for config in uncertainty_configs:
            name = config["name"]
            print(f"\n=== Running stride {stride}, distribution {name} ===")

            sigmas_combined, _, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
                nsd=7, alpha=np.sqrt(9 / (7 + (3 - 7))), beta=2, kappa=(3 - 7),
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

            print(f"Saved propagated results to {output_dir}")

            if stride == 1 and name == "baseline":
                for label, t_idx_global in [("max-width", max_t_idx), ("min-width", min_t_idx)]:
                    bundle_idx = 1
                    t0 = forwardTspan[t_idx_global]
                    t1 = forwardTspan[t_idx_global + 1]
                    t_min, t_max = min(t0, t1), max(t0, t1)

                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')

                    mask_bundle = X[:, -2] == bundle_idx
                    mask_time = (X[:, 0] >= t_min - 1e-5) & (X[:, 0] <= t_max + 1e-5)
                    X_segment = X[mask_bundle & mask_time]

                    print(f"[DEBUG] X_segment shape: {X_segment.shape}")
                    if X_segment.size == 0:
                        print(f"[WARN] No data for bundle {bundle_idx} in time segment {t_min}-{t_max}")
                        continue

                    num_sigmas = int(np.max(X[:, -1]) + 1)
                    for sigma_idx in range(num_sigmas):
                        X_sigma = X_segment[X_segment[:, -1] == sigma_idx]
                        if len(X_sigma) == 0:
                            continue
                        X_sigma = X_sigma[np.argsort(X_sigma[:, 0])]
                        p, f, g, h, k, L = [X_sigma[:, i] for i in range(1, 7)]
                        try:
                            r, _ = mee2rv(p, f, g, h, k, L, mu)
                            if r is None or np.any(np.isnan(r)):
                                print(f"[WARN] Skipping sigma {sigma_idx} — mee2rv returned invalid data")
                                continue
                            ax.plot(r[:, 0], r[:, 1], r[:, 2], color='blue', alpha=0.5)
                        except Exception as e:
                            print(f"[ERROR] mee2rv failed on sigma {sigma_idx}: {e}")
                            continue

                    ax.set_title(f"Sigma Point Trajectories (Bundle {bundle_idx}, {label})")
                    ax.set_xlabel("X [km]")
                    ax.set_ylabel("Y [km]")
                    ax.set_zlabel("Z [km]")
                    ax.legend()
                    ax.set_box_aspect([1.25, 1, 0.75])
                    plt.tight_layout()
                    plt.show()

if __name__ == "__main__":
    main()
