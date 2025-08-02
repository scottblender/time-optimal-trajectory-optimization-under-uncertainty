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

def compute_kl_divergence(mu1, sigma1, mu2, sigma2):
    k = mu1.shape[0]
    sigma2_inv = np.linalg.inv(sigma2)
    trace_term = np.trace(sigma2_inv @ sigma1)
    diff = mu2 - mu1
    quadratic_term = diff.T @ sigma2_inv @ diff
    sign1, logdet1 = np.linalg.slogdet(sigma1)
    sign2, logdet2 = np.linalg.slogdet(sigma2)
    if sign1 <= 0 or sign2 <= 0:
        print(f"[WARN] Non-positive-definite covariance matrix detected")
        return 0.0
    log_det_term = logdet2 - logdet1
    kl_div = 0.5 * (trace_term + quadratic_term - k + log_det_term)
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

        for label, idx in [("max", max_t_idx), ("min", min_t_idx)]:
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

                DU = 696340e3 
                g0_s = 9.81
                TU = np.sqrt(DU / g0_s)  # time unit in seconds
                MU = 4000           # kg
                VU = DU / TU        # velocity unit in km/s

                # Dimensional covariances (in km^2, (km/s)^2, kg^2)
                P_pos_km2 = 1e-2*1e6
                P_vel_kms2 = 1e-4*1e6
                P_mass_kg2 = 1e-2

                # Non-dimensional covariances
                P_pos_nd = P_pos_km2 / (DU**2)
                P_vel_nd = P_vel_kms2 / (VU**2)
                P_mass_nd = P_mass_kg2 / (MU**2)

                sigmas_combined, _, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
                    nsd=7, alpha=3/np.sqrt(3), beta=2, kappa=3.-7,
                    P_pos=np.eye(3)*P_pos_nd, P_vel=np.eye(3)*P_vel_nd, P_mass=np.array([[P_mass_nd]]),
                    time_steps=time_steps, r_bundles=r0, v_bundles=v0, mass_bundles=m0s,
                    num_workers=os.cpu_count()
                )

                print("\n[DEBUG] --- Sigma Point vs Bundle Alignment ---")
                print("Bundle r0 (r0[0,:,0]):", r0[idx, :, 0])
                print("Bundle r0 (r0[0,:,0]):", r0[idx, :, 0])
                print("Sigma point r0 (sigmas_combined[0,0,:3,0]):", sigmas_combined[0, 0, :3, 0])
                print("Δr:", np.linalg.norm(sigmas_combined[0, 0, :3, 0] - r0[idx, :, 0]), "DU")
                print("-----------------------------------------------\n")


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
                mu_sigma = mu_sigma_list[0]

                delta = mu_sigma[0, -1] - mu_mc[0, 0, -1]
                quad_term = delta.T @ inv(P_mc[0,0,-1] + 1e-12*np.eye(7)) @ delta
                trace_term = np.trace(inv(P_mc[0,0,-1]) @ P_sigma[0, -1])
                _, logdet1 = np.linalg.slogdet(P_sigma[0, -1])
                _, logdet2 = np.linalg.slogdet(P_mc[0,0,-1])
                logdet_term = logdet2 - logdet1
                print("Quadratic term (mean mismatch):", quad_term)
                print("Trace term:", trace_term)
                print("Log-det term:", logdet_term)

                kl_vals = [compute_kl_divergence(mu_sigma[0, t], P_sigma[0, t], mu_mc[0, 0, t], P_mc[0, 0, t])
                           for t in range(P_sigma.shape[1])]
                np.savetxt(f"{out_dir}/kl_divergence.txt", kl_vals, fmt="%.6f")
                np.savetxt(f"{out_dir}/cov_sigma_final.txt", P_sigma[0, -1], fmt="%.6f")
                np.savetxt(f"{out_dir}/cov_mc_final.txt", P_mc[0, 0, -1], fmt="%.6f")

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

                P_sigma = P_sigma_list[0]
                P_mc = P_mc[0, 0]  # shape: [T, 7, 7]

                print("[DEBUG] Sigma Cov Diag (t0):", np.diag(P_sigma[0, 0, :3, :3]))
                print("[DEBUG] MC Cov Diag (t0):   ", np.diag(P_mc[0, :3, :3]))
                print(f"[DEBUG] Sigma Cov Diag at tf: {np.diag(P_sigma[0, -1, :3, :3])}")
                print(f"[DEBUG] MC Cov Diag at tf:    {np.diag(P_mc[-1, :3, :3])}")

                def compute_3sigma_volume(cov_du2, DU):
                    # Convert from DU² to km²
                    cov_m2 = cov_du2 * DU**2
                    cov_km2 = cov_m2 / 1e6 
                    cov_km2 = 0.5 * (cov_km2 + cov_km2.T) + np.eye(3) * 1e-12
                    eigvals = np.linalg.eigvalsh(cov_km2)
                    if np.any(eigvals <= 0): return 0.0
                    det_P = np.prod(eigvals)
                    return 36 * np.pi * np.sqrt(det_P)
                
                print("[CHECK] P_sigma at t0 in km²:", np.diag(P_sigma[0,0,:3,:3]) * DU**2/1e6)
                print("[CHECK] P_sigma at tf in km²:", np.diag(P_sigma[0,-1,:3,:3]) * DU**2/1e6)

                vol_t0 = compute_3sigma_volume(P_sigma[0, 0, :3, :3], DU)
                vol_tf = compute_3sigma_volume(P_sigma[0, -1, :3, :3], DU)
                print(f"[CHECK] Volume t0: {vol_t0:.6f} km³, Volume tf: {vol_tf:.6f} km³")
                
                volumes = {
                    "sigma_t0": compute_3sigma_volume(P_sigma[0, 0, :3, :3], DU),
                    "sigma_tf": compute_3sigma_volume(P_sigma[0, -1, :3, :3], DU),
                    "mc_t0": compute_3sigma_volume(P_mc[0, :3, :3], DU),
                    "mc_tf": compute_3sigma_volume(P_mc[-1, :3, :3], DU)
                }

                volume_outfile = os.path.join(out_dir, "ellipsoid_volumes.txt")
                with open(volume_outfile, "w") as f:
                    for key, val in volumes.items():
                        f.write(f"{key}: {val:.6e} km^3\n")
                print("[INFO] 3-sigma ellipsoid volumes saved:", volume_outfile)

                print("\n[SIGMA TRAJECTORY DEBUG]")
                for i in range(len(traj[0][0])):
                    full = np.concatenate([seg[i] for seg in traj[0]], axis=0)
                    r0, rf = full[0, :3], full[-1, :3]
                    print(f"  Sigma {i:02d}: r0 = {r0}, rf = {rf}, Δr = {np.linalg.norm(rf - r0):.3f} DU")

                print("\n[MC TRAJECTORY DEBUG]")
                for i in range(min(10, len(mc_traj[0][0]))):  # Print only first 10 for brevity
                    full = np.concatenate([seg[i] for seg in mc_traj[0]], axis=0)
                    r0, rf = full[0, :3], full[-1, :3]
                    print(f"  MC {i:02d}: r0 = {r0}, rf = {rf}, Δr = {np.linalg.norm(rf - r0):.3f} DU")

                r_start_bundle = r_b[idx, :, bundle_idx]
                r_end_bundle = r_b[idx + 1, :, bundle_idx]
                delta_r_bundle = np.linalg.norm(r_end_bundle - r_start_bundle)

                print(f"[BUNDLE SEGMENT DEBUG] Raw bundle r0 = {r_start_bundle}")
                print(f"[BUNDLE SEGMENT DEBUG] Raw bundle rf = {r_end_bundle}")
                print(f"[BUNDLE SEGMENT DEBUG] Raw Δr = {delta_r_bundle:.6f} DU ({delta_r_bundle * DU:.2f} km)")

                fig = plt.figure(figsize=(6.5, 5.5))
                ax = fig.add_subplot(111, projection='3d')
                for i in range(len(traj[0][0])):
                    full = np.concatenate([seg[i] for seg in traj[0]], axis=0)
                    r = full[:, :3]
                    ax.plot(r[:, 0], r[:, 1], r[:, 2],
                            color='black' if i == 0 else 'gray',
                            linestyle='-' if i == 0 else '--',
                            lw=2.2 if i == 0 else 0.8, alpha=1.0, zorder=5 if i==0 else 4)
                    ax.scatter(r[0, 0], r[0, 1], r[0, 2], color='black', marker='o', s=10, zorder=1)
                    ax.scatter(r[-1, 0], r[-1, 1], r[-1, 2], color='black', marker='X', s=10, zorder=1)

                plot_3sigma_ellipsoid(ax, mu_sigma[0, 0, :3], P_sigma[0, 0, :3, :3])
                plot_3sigma_ellipsoid(ax, mu_sigma[0, -1, :3], P_sigma[0, -1, :3, :3])

                for j in range(0, len(mc_traj[0][0]), 10):
                    full_mc = np.concatenate([seg[j] for seg in mc_traj[0]], axis=0)
                    ax.plot(full_mc[:, 0], full_mc[:, 1], full_mc[:, 2], linestyle=':', color='dimgray', lw=0.8, alpha=0.4, zorder=3)
                    ax.scatter(full_mc[0, 0], full_mc[0, 1], full_mc[0, 2], color='0.4', s=8, marker='o', alpha=0.3,zorder=1)
                    ax.scatter(full_mc[-1, 0], full_mc[-1, 1], full_mc[-1, 2], color='0.4', s=8, marker='X', alpha=0.3,zorder=1)

                ax.set_xlabel('X [DU]')
                ax.set_ylabel('Y [DU]')
                ax.set_zlabel('Z [DU]')
                ax.grid(False)
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
