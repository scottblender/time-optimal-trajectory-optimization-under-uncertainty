import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh, inv
from scipy.spatial.distance import mahalanobis
from tqdm import tqdm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
import generate_sigma_points
from solve_trajectories import solve_trajectories_with_covariance_parallel_with_progress
from generate_monte_carlo_trajectories import generate_monte_carlo_trajectories_parallel

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.4, scale=3.0):
    eigvals, eigvecs = eigh(cov)
    radii = scale * np.sqrt(np.maximum(eigvals, 0))
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    coords = np.stack((x, y, z), axis=-1) @ eigvecs.T + mean
    ax.plot_surface(coords[:, :, 0], coords[:, :, 1], coords[:, :, 2],
                    rstride=1, cstride=1, alpha=alpha, color=color, linewidth=0)

def check_X_matches_trajectories(X, trajectories, backTspan, time_steps, num_substeps=10, evals_per_substep=20, atol=1e-6):
    print("\n[INFO] Verifying X matches trajectories...")
    forwardTspan = backTspan[::-1]
    segment_times = [forwardTspan[t] for t in time_steps]

    t_flat = []
    for seg_idx in range(len(segment_times) - 1):
        tstart = segment_times[seg_idx]
        tend = segment_times[seg_idx + 1]
        sub_times = np.linspace(tstart, tend, num_substeps + 1)
        for k in range(num_substeps):
            t_eval = np.linspace(sub_times[k], sub_times[k + 1], evals_per_substep)
            t_flat.extend(t_eval)
    t_flat = np.array(t_flat)

    num_matched = 0
    num_total = X.shape[0]

    for row in X:
        t = row[0]
        bundle_idx = int(row[-2])
        sigma_idx = int(row[-1])
        found = False

        for seg_idx, segment in enumerate(trajectories[bundle_idx]):
            if sigma_idx >= len(segment):
                continue
            traj = segment[sigma_idx]
            for i, state in enumerate(traj):
                if np.isclose(t, t_flat[i], atol=atol):
                    found = True
                    break
            if found:
                break

        if found:
            num_matched += 1
        else:
            print(f"[WARN] X time {t:.6f} not found in trajectory for bundle {bundle_idx}, sigma {sigma_idx}")

    print(f"[DONE] {num_matched}/{num_total} X rows matched to trajectories ({100.0 * num_matched / num_total:.2f}%)\n")

def main():
    data = joblib.load("bundle_data.pkl")
    r_b = data["r_bundles"][::-1]
    v_b = data["v_bundles"][::-1]
    m_b = data["mass_bundles"][::-1]
    lam_b = data["new_lam_bundles"][::-1]
    backTspan = data["backTspan"]
    mu, F, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    forwardTspan = np.flip(backTspan)
    num_bundles = r_b.shape[2]

    bundle_widths = np.linalg.norm(np.max(r_b, axis=2) - np.min(r_b, axis=2), axis=1)
    max_t_idx = int(np.argmax(bundle_widths))
    min_t_idx = next(i for i in np.argsort(bundle_widths) if i + 1 < len(forwardTspan))

    uncertainty_config = {
        "P_pos": np.eye(3) * 0.01,
        "P_vel": np.eye(3) * 0.0001,
        "P_mass": np.array([[0.0001]])
    }

    for label, idx in [("Max", max_t_idx), ("Min", min_t_idx)]:
        time_steps = np.array([idx, idx + 1])
        width_val = bundle_widths[idx]

        print(f"\n=== Plotting {label} Bundle Width Interval ===")

        sigmas_combined, _, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
            nsd=7, alpha=np.sqrt(9 / (7 + (3 - 7))), beta=2, kappa=(3 - 7),
            P_pos=uncertainty_config["P_pos"],
            P_vel=uncertainty_config["P_vel"],
            P_mass=uncertainty_config["P_mass"],
            time_steps=time_steps,
            r_bundles=r_b, v_bundles=v_b, mass_bundles=m_b,
            num_workers=os.cpu_count()
        )

        trajectories, P_hist, means_hist, X_sigma, _ = solve_trajectories_with_covariance_parallel_with_progress(
            backTspan, time_steps, len(time_steps), num_bundles,
            sigmas_combined, lam_b, m_b, mu, F, c, m0, g0, Wm, Wc,
            num_workers=os.cpu_count()
        )

        mc_traj, P_mc, _, _, _ = generate_monte_carlo_trajectories_parallel(
            backTspan, time_steps, len(time_steps), num_bundles,
            sigmas_combined, lam_b, mu, F, c, m0, g0,
            num_samples=500, num_workers=os.cpu_count()
        )

        bundle_idx = 0
        sigma_traj = trajectories[bundle_idx]
        mc_bundle_traj = mc_traj[bundle_idx]
        P_sigma = P_hist[bundle_idx][0]
        P_mc_final = P_mc[bundle_idx][0]  # 3D (segment, 7, 7)

        # Extract final appended sigma0 from X and convert to ECI
        sigma0_rows = X_sigma[(X_sigma[:, -2] == bundle_idx) & (X_sigma[:, -1] == 0)]
        mee_final = sigma0_rows[-1, 1:8]
        p, f, g, h, k, L, m = mee_final
        r, v = mee2rv(p, f, g, h, k, L, mu)
        appended_sigma0_final = np.hstack((r, v, [m]))

        # Flatten sigma point trajectories
        sigma_point_trajectories = [
            np.concatenate([seg[sigma] for seg in sigma_traj], axis=0)
            for sigma in range(len(sigma_traj[0]))
        ]

        # Mahalanobis distances to appended sigma0
        P_inv = inv(P_sigma[-1][:7, :7] + 1e-10 * np.eye(7))
        mahal_dists = []
        for traj in sigma_point_trajectories[1:]:  # skip sigma 0
            x = traj[-1, :7]
            d = mahalanobis(x, appended_sigma0_final, P_inv)
            mahal_dists.append(d)
        mahal_dists = np.array(mahal_dists)

        # Save covariances and Mahalanobis distances
        np.savetxt(f"cov_{label.lower()}_sigma.txt", np.vstack([P_sigma[0], P_sigma[-1]]),
                   fmt="%.6f", header=f"Initial and Final Covariance (Sigma) | Bundle Width = {width_val:.6f}")
        np.savetxt(f"cov_{label.lower()}_mc.txt", np.vstack([P_mc_final[0], P_mc_final[-1]]),
                   fmt="%.6f", header=f"Initial and Final Covariance (Monte Carlo) | Bundle Width = {width_val:.6f}")
        with open(f"cov_{label.lower()}_sigma.txt", "a") as f:
            f.write("\nMahalanobis distances to appended sigma 0 at final step:\n")
            np.savetxt(f, mahal_dists[None], fmt="%.6f")

        check_X_matches_trajectories(X_sigma, trajectories, backTspan, time_steps)

        # === Plotting ===
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(False)

        for sigma_idx in range(len(sigma_traj[0])):
            full_traj = np.concatenate([segment[sigma_idx] for segment in sigma_traj], axis=0)
            r = full_traj[:, :3]
            color = 'black' if sigma_idx == 0 else 'gray'
            lw = 1.8 if sigma_idx == 0 else 1.2
            alpha = 1.0 if sigma_idx == 0 else 0.8
            ax.plot(r[:, 0], r[:, 1], r[:, 2], color=color, linewidth=lw, alpha=alpha)
            ax.scatter(r[0, 0], r[0, 1], r[0, 2], color=color, s=30, marker='o', alpha=alpha)
            ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color=color, s=40, marker='X', alpha=alpha)

        # 3-sigma ellipsoids
        plot_3sigma_ellipsoid(ax, r[0], P_sigma[0][:3, :3], color='0.5', alpha=0.25)
        plot_3sigma_ellipsoid(ax, r[-1], P_sigma[-1][:3, :3], color='0.5', alpha=0.25)

        # Monte Carlo (subsampled for clarity)
        for sample_idx in range(0, len(mc_bundle_traj[0]), 10):  # every 10th sample
            full_mc = np.concatenate([segment[sample_idx] for segment in mc_bundle_traj], axis=0)
            r = full_mc[:, :3]
            ax.plot(r[:, 0], r[:, 1], r[:, 2], color='0.6', lw=0.5, alpha=0.08)

        plot_3sigma_ellipsoid(ax, r[0], P_mc_final[0][:3, :3], color='0.6', alpha=0.25)
        plot_3sigma_ellipsoid(ax, r[-1], P_mc_final[-1][:3, :3], color='0.6', alpha=0.25)

        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')
        ax.set_title(f"Sigma Point vs. Monte Carlo Trajectories\n({label} Bundle Width Interval)")
        ax.view_init(elev=25, azim=135)
        ax.set_box_aspect([1.25, 1, 0.75])

        legend_elements = [
            Line2D([0], [0], color='black', lw=1.8, label='Nominal'),
            Line2D([0], [0], color='gray', lw=1.2, linestyle='-', label='Sigma Points'),
            Line2D([0], [0], color='0.6', lw=0.5, linestyle=':', label='Monte Carlo'),
            Patch(facecolor='0.5', edgecolor='0.5', alpha=0.25, label='3-Sigma Ellipsoid')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9, frameon=True, facecolor='white', edgecolor='black')

        plt.tight_layout()
        plt.savefig(f"fig_{label.lower()}_comparison.pdf", format='pdf', dpi=600)
        plt.close()

if __name__ == "__main__":
    main()
