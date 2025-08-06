import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.linalg import inv
from tqdm import tqdm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import joblib

plt.rcParams.update({'font.size': 10})

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
import generate_sigma_points
from solve_trajectories import solve_trajectories_with_covariance_parallel_with_progress
from generate_monte_carlo_trajectories import generate_monte_carlo_trajectories_parallel

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.2, scale=3.0):
    cov = 0.5 * (cov + cov.T) + np.eye(3) * 1e-10
    eigvals, eigvecs = np.linalg.eigh(cov)
    if np.any(eigvals <= 0): return
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    radii = scale * np.sqrt(eigvals)
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    ellipsoid = np.stack((x, y, z), axis=-1) @ eigvecs.T + mean
    ax.plot_surface(ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
                    rstride=1, cstride=1, color=color, alpha=alpha, linewidth=0)
    ax.plot_wireframe(ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
                      rstride=5, cstride=5, color='k', alpha=0.2, linewidth=0.3)

def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    ranges = [abs(lim[1] - lim[0]) for lim in [x_limits, y_limits, z_limits]]
    centers = [np.mean(lim) for lim in [x_limits, y_limits, z_limits]]
    max_range = max(ranges) / 2
    ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
    ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
    ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])
    ax.set_box_aspect([1.25, 1, 0.75])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

def set_max_ticks(fig, n=5):
    from matplotlib.ticker import MaxNLocator
    for ax in fig.get_axes():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=n))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=n))
        if hasattr(ax, 'zaxis'):
            ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

def compute_kl_divergence(mu1, sigma1, mu2, sigma2, epsilon=1e-10):
    """
    Computes KL divergence D_KL(P || Q) between two Gaussians:
        P ~ N(mu1, sigma1), Q ~ N(mu2, sigma2)

    Applies regularization to ensure invertibility and uses pseudo-inverse for stability.
    """
    k = mu1.shape[0]

    # --- Regularize both covariances ---
    sigma1_reg = sigma1 + np.eye(k) * epsilon
    sigma2_reg = sigma2 + np.eye(k) * epsilon

    # --- Attempt pseudo-inverse for numerical safety ---
    try:
        sigma2_inv = np.linalg.pinv(sigma2_reg)
    except np.linalg.LinAlgError:
        print("[ERROR] σ₂ inverse failed even with pseudo-inverse fallback.")
        return np.inf

    trace_term = np.trace(sigma2_inv @ sigma1_reg)
    diff = mu2 - mu1
    quadratic_term = diff.T @ sigma2_inv @ diff

    # --- Log determinant using regularized matrices ---
    sign1, logdet1 = np.linalg.slogdet(sigma1_reg)
    sign2, logdet2 = np.linalg.slogdet(sigma2_reg)

    if sign1 <= 0 or sign2 <= 0:
        print(f"[WARN] Non-positive-definite covariance matrix detected after regularization")
        return np.inf

    log_det_term = logdet2 - logdet1
    kl_div = 0.5 * (trace_term + quadratic_term - k + log_det_term)

    # --- Debug Info ---
    print(f"[KL DEBUG] trace = {trace_term:.4f}, quad = {quadratic_term:.4f}, log_det = {log_det_term:.4f}, total = {kl_div:.4f}")
    evals_sp = np.linalg.eigvalsh(sigma1_reg)
    evals_mc = np.linalg.eigvalsh(sigma2_reg)
    print(f"  SP eigenvalues: {np.round(evals_sp, 3)}")
    print(f"  MC eigenvalues: {np.round(evals_mc, 3)}")
    print(f"  SP min/max: {evals_sp.min():.2e} / {evals_sp.max():.2e}")
    print(f"  MC min/max: {evals_mc.min():.2e} / {evals_mc.max():.2e}")

    return max(kl_div, 0.0)

def main():
    stride_minutes_list = np.arange(100, 1001, 100)

    for stride_minutes in stride_minutes_list:
        start_time = time.time()
        out_root = f"stride_{stride_minutes}min"
        bundle_file = os.path.join(out_root, f"bundle_data_{stride_minutes}min.pkl")
        if not os.path.exists(bundle_file):
            print(f"[SKIP] Missing file: {bundle_file}")
            continue

        print(f"\n=== Processing {bundle_file} ===")
        data = joblib.load(bundle_file)
        r_tr = data["r_tr"]
        r_b = data["r_bundles"][::-1]
        v_b = data["v_bundles"][::-1]
        m_b = data["mass_bundles"][::-1]
        lam_b = data["new_lam_bundles"][::-1]
        backTspan = data["backTspan"]
        forwardTspan = backTspan[::-1]
        mu, F, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
        num_bundles = r_b.shape[2]

        # === Nominal vs bundle plots ===
        for name, include_bundles in [("bundle_vs_nominal_trajectories", True), ("nominal_only_trajectory", False)]:
            fig = plt.figure(figsize=(6.5, 5.5))
            ax = fig.add_subplot(111, projection='3d')
            if include_bundles:
                for i in range(num_bundles):
                    ax.plot(r_b[:, 0, i], r_b[:, 1, i], r_b[:, 2, i], color='0.6', alpha=0.25)
            ax.plot(r_tr[:, 0], r_tr[:, 1], r_tr[:, 2], color='black', linewidth=2, label='Nominal')
            ax.scatter(r_tr[0, 0], r_tr[0, 1], r_tr[0, 2], color='black', marker='o', s=20, label='Start')
            ax.scatter(r_tr[-1, 0], r_tr[-1, 1], r_tr[-1, 2], color='black', marker='X', s=25, label='End')
            ax.set_xlabel("X [DU]"); ax.set_ylabel("Y [DU]"); ax.set_zlabel("Z [DU]")
            set_axes_equal(ax)
            set_max_ticks(fig)
            ax.legend(loc='upper left', bbox_to_anchor=(0.03, 0.95))
            plt.savefig(f"{out_root}/{name}.pdf", dpi=600, bbox_inches='tight', pad_inches=0.5)
            plt.close()

        # === Segment width computation ===
        widths = []
        for t in range(r_b.shape[0]):
            points = r_b[t].T
            dists = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
            max_dist = np.max(dists)
            widths.append((forwardTspan[t], max_dist))

        widths_array = np.array(widths)
        np.savetxt(f"{out_root}/bundle_segment_widths.txt", widths_array, fmt="%.6f", header="time_sec width_km")

        max_t_idx = int(np.argmax(widths_array[:, 1]))
        min_t_idx = int(np.argmin(widths_array[:, 1]))
        if min_t_idx == len(widths_array) - 1:
            sorted_indices = np.argsort(widths_array[:, 1])
            min_t_idx = sorted_indices[1]

        print(f"[INFO] Max width at t = {widths_array[max_t_idx, 0]:.2f} TU → {widths_array[max_t_idx, 1]:.6f} km")
        print(f"[INFO] Min width at t = {widths_array[min_t_idx, 0]:.2f} TU → {widths_array[min_t_idx, 1]:.6f} km")

        for label, idx in [("min", min_t_idx),("max", max_t_idx)]:
            dists = np.linalg.norm(r_b[idx] - r_tr[idx][:, np.newaxis], axis=0)
            bundle_farthest = int(np.argmax(dists))
            bundle_closest = int(np.argmin(dists))

            for mode, bundle_idx in [("farthest", bundle_farthest), ("closest", bundle_closest)]:
                print(f"[INFO] {label.upper()} segment — {mode} bundle idx = {bundle_idx}")
                r0 = r_b[:, :, bundle_idx][:, :, np.newaxis]
                v0 = v_b[:, :, bundle_idx][:, :, np.newaxis]
                m0s = m_b[:, bundle_idx][:, np.newaxis]
                lam0 = lam_b[:, :, bundle_idx][:, :, np.newaxis]
                time_steps = np.array([idx, idx + 1])
                global_bundle_indices = [bundle_idx]
                out_dir = f"{out_root}/segment_{label}_{mode}_bundle_{bundle_idx}"
                os.makedirs(out_dir, exist_ok=True)

                DU_km = 696340.0  # Sun radius in km
                g0_s = 9.81
                TU = np.sqrt(DU_km / g0_s)
                VU_kms = DU_km / TU # Convert m/s to km/s using g0

                # Set desired physical covariances (e.g., 0.1 km², 1e-4 (km/s)², 1 kg²)
                P_pos_km2 = np.eye(3) * 0.1        # km²
                P_vel_kms2 = np.eye(3) * 1e-6     # (km/s)²
                P_mass_kg2 = np.array([[1e-2]])     # kg²

                # Convert to non-dimensional units for input
                P_pos = P_pos_km2 / (DU_km**2)
                P_vel = P_vel_kms2 / (VU_kms**2)
                P_mass = P_mass_kg2 / (4000**2)

                sigmas_combined, _, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
                    nsd=7, alpha=np.sqrt(9 / (7 + (3 - 7))), beta=2, kappa=(3 - 7),
                    P_pos=P_pos, P_vel=P_vel, P_mass=P_mass,
                    time_steps=time_steps, r_bundles=r0, v_bundles=v0, mass_bundles=m0s,
                    num_workers=os.cpu_count()
                )

                traj, P_sigma_list, mu_sigma_list, X_sigma, _ = solve_trajectories_with_covariance_parallel_with_progress(
                    backTspan, time_steps, 2,
                    sigmas_combined, lam0, m0s, mu, F, c, m0, g0, Wm, Wc,
                    global_bundle_indices=global_bundle_indices,
                    num_workers=os.cpu_count()
                )

                mc_traj, P_mc, mu_mc, _, _ = generate_monte_carlo_trajectories_parallel(
                    backTspan, time_steps, 2,
                    sigmas_combined, lam0, mu, F, c, m0, g0,
                    global_bundle_indices=global_bundle_indices,
                    num_samples=1000, num_workers=os.cpu_count()
                )

                P_sigma = P_sigma_list[0]
                # === Debug print to compare sigma covariance at segment start ===
                P_init_diag = np.diag(P_pos)
                P_sigma_start_diag = np.diag(P_sigma[0, 0, :3, :3])

                print(f"[DEBUG] {label.upper()} / {mode} / Bundle {bundle_idx} — Segment Start Covariance Check:")
                print(f"    Empirical Sigma Cov Diag at t0: {P_sigma_start_diag}")
                print(f"    Expected P_init Diag:           {P_init_diag}")
                print(f"    Ratio:                          {P_sigma_start_diag / P_init_diag}")
                mu_sigma = mu_sigma_list[0]
                kl_vals = [compute_kl_divergence(mu_sigma[0, t], P_sigma[0, t], mu_mc[0, 0, t], P_mc[0, 0, t])
                           for t in range(P_sigma.shape[1])]
                # === Optional: Mahalanobis distance squared per timestep ===
                for t in range(P_sigma.shape[1]):
                    diff = mu_sigma[0, t] - mu_mc[0, 0, t]
                    cov_inv = inv(P_mc[0, 0, t] + 1e-12 * np.eye(7))  # add epsilon for safety
                    d2 = diff.T @ cov_inv @ diff
                    print(f"[MAHAL@t={t:02d}] Mahalanobis² = {d2:.3f} — |μ_SP − μ_MC| = {np.linalg.norm(diff):.3e}")
                np.savetxt(f"{out_dir}/kl_divergence.txt", kl_vals, fmt="%.6f")
                np.savetxt(f"{out_dir}/cov_sigma_final.txt", P_sigma[0, -1], fmt="%.6f")
                np.savetxt(f"{out_dir}/cov_mc_final.txt", P_mc[0, 0, -1], fmt="%.6f")

                print(f"MC: {np.diag(P_mc[0,0,-1])}")
                print(f"SP: {np.diag(P_sigma[0,-1])}")

                delta_mu = mu_sigma - mu_mc
                mean_error_norm = np.linalg.norm(delta_mu)  # in DU or km
                print(f"[DEBUG] Mean difference norm: {mean_error_norm:.6f}")
                cov_norms = []
                for t in range(P_sigma.shape[1]):
                    P_sp = P_sigma[0, t]
                    P_mc_t = P_mc[0, 0, t]

                    cov_diff = P_sp - P_mc_t
                    frob_norm = np.linalg.norm(cov_diff, ord='fro')  # Frobenius norm
                    cov_norms.append(frob_norm)

                    print(f"[COV NORM @ t={t:03d}] ‖ΔP‖_F = {frob_norm:.3e}")
                for t in range(P_sigma.shape[1]):
                    mu_sp_t = mu_sigma[0, t]  # shape (7,)
                    mu_mc_t = mu_mc[0, 0, t]  # shape (7,)
                    delta_mu = mu_sp_t - mu_mc_t

                    P_sp = P_sigma[0, t]
                    P_mc_t = P_mc[0, 0, t]

                    D2_sp = delta_mu.T @ inv(P_sp) @ delta_mu
                    D2_mc = delta_mu.T @ inv(P_mc_t) @ delta_mu

                    print(f"[MHAL@t={t:03d}] SP = {D2_sp:.2e}, MC = {D2_mc:.2e}, Δ = {abs(D2_sp - D2_mc):.2e}")
                final_time = forwardTspan[idx + 1]
                mask = np.isclose(X_sigma[:, 0], final_time) & (X_sigma[:, -2] == bundle_idx) & (X_sigma[:, -1] == 0) & np.all(X_sigma[:, 8:15] == 0, axis=1)
                if not np.any(mask):
                    print(f"[WARN] No appended sigma₀ row found for {label}/{mode} at t = {final_time:.6f}")
                    continue
                row = X_sigma[mask][0]
                mee_final = row[1:8]
                r, v = mee2rv(*mee_final[:6], mu)
                sigma0_final = np.hstack((r, v, [mee_final[6]]))
                P_final = P_sigma[0, -1][:7, :7]
                trajs = [np.concatenate([seg[i] for seg in traj[0]], axis=0) for i in range(len(traj[0][0]))]
                mahal = [np.sqrt((x[-1, :7] - sigma0_final) @ inv(P_final + 1e-10*np.eye(7)) @ (x[-1, :7] - sigma0_final)) for x in trajs[1:]]
                np.savetxt(f"{out_dir}/mahalanobis_distances.txt", mahal, fmt="%.6f")

                fig = plt.figure(figsize=(6.5, 5.5))
                ax = fig.add_subplot(111, projection='3d')

                INFLATE_FACTOR = 1e9  # artificially inflate deviations
                # Reference trajectory: sigma₀
                traj_array = traj[0][0]  # shape = (15, 200, 7)
                r_sigma0 = traj_array[0, :, :3]  # sigma 0 position trajectory

                for i in range(1,traj_array.shape[0]):
                    r = traj_array[i, :, :3]  # shape (200, 3)
                    r_inflated = r_sigma0 + INFLATE_FACTOR * (r - r_sigma0)
                    ax.plot(r_inflated[:, 0], r_inflated[:, 1], r_inflated[:, 2],
                            color='black' if i == 0 else 'gray',
                            linestyle='-' if i == 0 else '--',
                            lw=2.2 if i == 0 else 0.8, alpha=1.0, zorder=5 if i == 0 else 4)

                    ax.scatter(r_inflated[0, 0], r_inflated[0, 1], r_inflated[0, 2],
                            color='black' if i == 0 else 'gray', marker='o', s=10, zorder=1, alpha=1.0)
                    ax.scatter(r_inflated[-1, 0], r_inflated[-1, 1], r_inflated[-1, 2],
                            color='black' if i == 0 else 'gray', marker='X', s=10, zorder=1, alpha=1.0)
                
                # === Monte Carlo Trajectories (inflated around σ₀) ===
                mc_traj_array = mc_traj[0][0]
                for j in range(0, mc_traj_array.shape[0], 10):
                    r_mc = mc_traj_array[j,:,:3]
                    r_mc_inflated = r_sigma0 + INFLATE_FACTOR * (r_mc - r_sigma0)
                    ax.plot(r_mc_inflated[:, 0], r_mc_inflated[:, 1], r_mc_inflated[:, 2],
                            linestyle=':', color='dimgray', lw=0.8, alpha=0.4, zorder=3)
                    ax.scatter(r_mc_inflated[0, 0], r_mc_inflated[0, 1], r_mc_inflated[0, 2],
                            color='0.4', s=8, marker='o', alpha=0.3, zorder=1)
                    ax.scatter(r_mc_inflated[-1, 0], r_mc_inflated[-1, 1], r_mc_inflated[-1, 2],
                            color='0.4', s=8, marker='X', alpha=0.3, zorder=1)
                # === 3σ Ellipsoids remain uninflated ===
                # plot_3sigma_ellipsoid(ax, mu_sigma[0, 0, :3], P_sigma[0, 0, :3, :3])
                # plot_3sigma_ellipsoid(ax, mu_sigma[0, -1, :3], P_sigma[0, -1, :3, :3])

                # === Auto-zoom ===
                set_axes_equal(ax)
                set_max_ticks(fig)

                ax.legend(handles=[
                    Line2D([0], [0], color='black', lw=2.2, label='Sub-nominal Mean State'),
                    Line2D([0], [0], color='gray', lw=0.8, linestyle='--', label='Sigma Points'),
                    Line2D([0], [0], color='0.4', lw=0.6, linestyle=':', label='Monte Carlo'),
                    Line2D([0], [0], marker='o', color='black', linestyle='', label='Start', markersize=4),
                    Line2D([0], [0], marker='X', color='black', linestyle='', label='End', markersize=5),
                    Patch(facecolor='0.5', edgecolor='0.5', alpha=0.2, label='3-σ Ellipsoid')
                ], loc='upper left', bbox_to_anchor=(0.03, 1.07), frameon=True, facecolor='white')

                plt.savefig(f"{out_dir}/sigma_mc_comparison.pdf", dpi=600, bbox_inches='tight', pad_inches=0.5)
                plt.close()

        runtime = time.time() - start_time
        with open(f"{out_root}/runtime.txt", "w") as f:
            f.write(f"{runtime:.2f}")

if __name__ == "__main__":
    main()
