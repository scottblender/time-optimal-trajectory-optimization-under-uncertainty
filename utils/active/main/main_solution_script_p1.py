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
from matplotlib.ticker import ScalarFormatter, FixedLocator
import joblib

plt.rcParams.update({'font.size': 10})

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
import generate_sigma_points
from solve_trajectories import solve_trajectories_with_covariance_parallel_with_progress
from generate_monte_carlo_trajectories import generate_monte_carlo_trajectories_parallel

# =========================
# Ellipsoid + plotting helpers
# =========================

def ellipsoid_aabb(mean, cov, scale=3.0, pad=0.05):
    """Axis-aligned bounds of a rotated ellipsoid defined by (mean, cov)."""
    cov = 0.5*(cov + cov.T) + 1e-12*np.eye(3)
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, 0, None)
    r = scale*np.sqrt(w)                 # 3σ principal radii
    A = V @ np.diag(r)                   # columns are axis vectors
    half = np.sum(np.abs(A), axis=1)     # AABB half-sizes
    lo = mean - half; hi = mean + half
    span = hi - lo
    return lo - pad*span, hi + pad*span

def plot_3sigma_ellipsoid_lines(ax, mean, cov, color='0.5', scale=3.0,
                                n_meridians=24, n_parallels=16, lw=0.9, alpha=0.9, zorder=5):
    """Artifact-free 3σ ellipsoid rendered as a line frame (no translucent surface)."""
    cov = 0.5*(cov + cov.T) + 1e-12*np.eye(3)
    w, V = np.linalg.eigh(cov)
    if np.any(w <= 0):
        return
    r = scale*np.sqrt(np.clip(w, 0, None))

    u = np.linspace(0, 2*np.pi, n_meridians, endpoint=False)
    v = np.linspace(-np.pi/2, np.pi/2, n_parallels)

    # Meridians
    for ui in u:
        cu, su = np.cos(ui), np.sin(ui)
        x = r[0]*np.cos(v)*cu
        y = r[1]*np.cos(v)*su
        z = r[2]*np.sin(v)
        E = (np.column_stack((x, y, z)) @ V.T) + mean
        ax.plot(E[:,0], E[:,1], E[:,2], color=color, lw=lw, alpha=alpha, zorder=zorder)

    # Parallels
    for vi in v[1:-1]:
        cv, sv = np.cos(vi), np.sin(vi)
        x = r[0]*cv*np.cos(u)
        y = r[1]*cv*np.sin(u)
        z = np.full_like(u, r[2]*sv)
        E = (np.column_stack((x, y, z)) @ V.T) + mean
        ax.plot(E[:,0], E[:,1], E[:,2], color=color, lw=lw, alpha=alpha, zorder=zorder)

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
            ax.zaxis.set_major_locator(MaxNLocator(nbins=n))

def add_ellipsoid_inset(ax_main, mu_sp, P_sp, mu_mc, P_mc,
                        r_sp=None, r_mc=None,           # NEW
                        rect=(0.08, 0.54, 0.34, 0.34)):
    """
    3D inset in TOP-LEFT with grayscale 3σ ellipsoids (+ optional scatter points).
    rect is (left, bottom, width, height) in figure coordinates.
    """
    fig = ax_main.figure
    ax_in = fig.add_axes(rect, projection='3d', facecolor='white')

    # Ellipsoids (light, line-frame)
    plot_3sigma_ellipsoid_lines(ax_in, mu_sp, P_sp, color='0.35', scale=3.0, lw=0.8, alpha=0.40,zorder=6)
    plot_3sigma_ellipsoid_lines(ax_in, mu_mc, P_mc, color='0.70', scale=3.0, lw=0.8, alpha=0.35,zorder=5)

    # Optional: overlay scatter points in grayscale
    if r_mc is not None:
        ax_in.scatter(r_mc[:,0], r_mc[:,1], r_mc[:,2], c='0.55', s=12, alpha=0.25,
                           label='Monte Carlo', zorder=3)
    if r_sp is not None:
        ax_in.scatter(r_sp[:,0], r_sp[:,1], r_sp[:,2], c='0.0',  s=26, alpha=1,
                           label='Sigma Points', zorder=4)

    # Fit inset to union of 3σ AABBs (so points sit inside)
    lo_sp, hi_sp = ellipsoid_aabb(mu_sp, P_sp, scale=3.0, pad=0.08)
    lo_mc, hi_mc = ellipsoid_aabb(mu_mc, P_mc, scale=3.0, pad=0.08)
    lo = np.minimum(lo_sp, lo_mc); hi = np.maximum(hi_sp, hi_mc)
    ax_in.set_xlim(lo[0], hi[0]); ax_in.set_ylim(lo[1], hi[1]); ax_in.set_zlim(lo[2], hi[2])
    set_axes_equal(ax_in)  # uses your existing helper. :contentReference[oaicite:0]{index=0}

    # Minimalist inset
    ax_in.set_xticks([]); ax_in.set_yticks([]); ax_in.set_zticks([])
    ax_in.set_xlabel(""); ax_in.set_ylabel(""); ax_in.set_zlabel("")
    ax_in.set_title("3σ Ellipsoids", fontsize=9, pad=2, color='0.25')
    return ax_in

def expand_xyz_labels_no_offset(ax, fmt=".6f"):
    """
    Freeze current tick LOCATIONS and relabel with raw absolute values
    (no offset, no 10**k scaling). Also nudges label rotation/alignment.
    """

    # Make sure ticks exist (formatter/locator have run)
    ax.figure.canvas.draw()

    def _relabel(ax, axis):
        from mpl_toolkits.mplot3d.axis3d import XAxis, YAxis, ZAxis
        # 1) turn OFF offset/scaling for LABELING (do NOT change locator)
        f = axis.get_major_formatter()
        if hasattr(f, "set_useOffset"):
            f.set_useOffset(False)
        if isinstance(f, ScalarFormatter):
            # prevent scientific scaling on labels
            f.set_powerlimits((0, 0))

        # 2) freeze current positions so nothing moves when we relabel
        ticks = axis.get_ticklocs()
        axis.set_major_locator(FixedLocator(ticks))

        # 3) build raw absolute labels (no offset, no scaling)
        labels = [format(t, fmt) for t in ticks]

        # ensure we have text artists, then update text + style
        axis.set_ticklabels(labels)
        if isinstance(axis, YAxis):
            ax.set_yticklabels(labels, rotation=-12,
                                verticalalignment='baseline',
                                horizontalalignment='left')
        # 4) hide the little "+…"/"×10^k" artist just in case
        axis.offsetText.set_visible(False)

    # Apply per axis (these alignments work well in 3D)
    _relabel(ax, ax.xaxis)
    _relabel(ax, ax.yaxis)
    _relabel(ax, ax.zaxis)

    # redraw to commit text changes
    ax.figure.canvas.draw()

def set_max_ticks_exact(ax, n=4, tick_inset=0.1):
    """
    Force exactly n ticks on each axis, inset from the endpoints
    by 'tick_inset' fraction of the axis span.
    """
    # X
    lo, hi = ax.get_xlim3d()
    span = hi - lo
    lo_i = lo + tick_inset * span
    hi_i = hi - tick_inset * span
    ax.xaxis.set_major_locator(FixedLocator(np.linspace(lo_i, hi_i, n)))

    # Y
    lo, hi = ax.get_ylim3d()
    span = hi - lo
    lo_i = lo + tick_inset * span
    hi_i = hi - tick_inset * span
    ax.yaxis.set_major_locator(FixedLocator(np.linspace(lo_i, hi_i, n)))

    # Z
    lo, hi = ax.get_zlim3d()
    span = hi - lo
    lo_i = lo + tick_inset * span
    hi_i = hi - tick_inset * span
    ax.zaxis.set_major_locator(FixedLocator(np.linspace(lo_i, hi_i, n)))

# =========================
# Metrics
# =========================

def compute_kl_divergence(mu1, sigma1, mu2, sigma2, epsilon=1e-10):
    """KL divergence D_KL(N1 || N2) for two Gaussians."""
    k = mu1.shape[0]
    sigma1_reg = sigma1 + np.eye(k) * epsilon
    sigma2_reg = sigma2 + np.eye(k) * epsilon
    try:
        sigma2_inv = np.linalg.pinv(sigma2_reg)
    except np.linalg.LinAlgError:
        print("[ERROR] σ₂ inverse failed even with pseudo-inverse fallback.")
        return np.inf

    trace_term = np.trace(sigma2_inv @ sigma1_reg)
    diff = mu2 - mu1
    quadratic_term = diff.T @ sigma2_inv @ diff
    s1, ld1 = np.linalg.slogdet(sigma1_reg)
    s2, ld2 = np.linalg.slogdet(sigma2_reg)
    if s1 <= 0 or s2 <= 0:
        print("[WARN] Non-PD covariance after regularization"); return np.inf
    kl_div = 0.5 * (trace_term + quadratic_term - k + (ld2 - ld1))

    ev_sp = np.linalg.eigvalsh(sigma1_reg); ev_mc = np.linalg.eigvalsh(sigma2_reg)
    print(f"[KL DEBUG] trace={trace_term:.4f} quad={quadratic_term:.4f} logdet={(ld2-ld1):.4f} total={kl_div:.4f}")
    print(f"  SP eig min/max: {ev_sp.min():.2e} / {ev_sp.max():.2e}")
    print(f"  MC eig min/max: {ev_mc.min():.2e} / {ev_mc.max():.2e}")
    return float(max(kl_div, 0.0))

def ellipsoid_3sigma_volume(P_pos, DU_km=696340.0):
    """
    3σ ellipsoid volume from a 3x3 position covariance (in DU^2).
    Returns (vol_DU3, vol_km3).
    """
    # be safe: symmetrize + clamp tiny negatives
    P = 0.5*(P_pos + P_pos.T)
    w = np.clip(np.linalg.eigvalsh(P), 0.0, None)   # DU^2
    radii_DU = 3.0 * np.sqrt(w)                     # 3σ semi-axes in DU
    vol_DU3 = (4.0/3.0) * np.pi * np.prod(radii_DU) # DU^3
    vol_km3 = vol_DU3 * (DU_km**3)                  # km^3 (1 DU = Sun radius)
    return vol_DU3, vol_km3

# =========================
# Main
# =========================

def main():
    stride_minutes_list = np.arange(1000, 6001, 500)

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

        # === Nominal vs bundle plots (grayscale) ===
        for name, include_bundles in [("bundle_vs_nominal_trajectories", True), ("nominal_only_trajectory", False)]:
            fig = plt.figure(figsize=(6.5, 5.5))
            ax = fig.add_subplot(111, projection='3d')
            if include_bundles:
                for i in range(num_bundles):
                    ax.plot(r_b[:, 0, i], r_b[:, 1, i], r_b[:, 2, i], color='0.6', alpha=0.25)
            ax.plot(r_tr[:, 0], r_tr[:, 1], r_tr[:, 2], color='0.0', linewidth=2, label='Nominal')
            ax.scatter(r_tr[0, 0], r_tr[0, 1], r_tr[0, 2], color='0.0', marker='o', s=20, label='Start')
            ax.scatter(r_tr[-1, 0], r_tr[-1, 1], r_tr[-1, 2], color='0.0', marker='X', s=25, label='End')
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

        for label, idx in [("max", max_t_idx),("min", min_t_idx)]:
            dists = np.linalg.norm(r_b[idx] - r_tr[idx][:, np.newaxis], axis=0)
            bundle_farthest = int(np.argmax(dists))
            bundle_closest = int(np.argmin(dists))

            for mode, bundle_idx in [("closest", bundle_closest),("farthest", bundle_farthest)]:
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
                g0_s = 9.81/1000
                TU = np.sqrt(DU_km / g0_s)
                VU_kms = DU_km / TU

                # Physical covariances (km / km/s / kg), then → non-dimensional
                P_pos_km2  = np.eye(3) * 0.01
                P_vel_kms2 = np.eye(3) * 1e-10
                P_mass_kg2 = np.array([[1e-3]])
                P_pos  = P_pos_km2  / (DU_km**2)
                P_vel  = P_vel_kms2 / (VU_kms**2)
                P_mass = P_mass_kg2 / (4000**2)

                sigmas_combined, _, _, _, Wm, Wc = generate_sigma_points.generate_sigma_points(
                    nsd=7, alpha=3/np.sqrt(3), beta=2, kappa=3.-7,
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

                # ---------- Scatter: Initial (t0) with 3σ inset ----------
                fig = plt.figure(figsize=(6.5, 5.5))
                ax  = fig.add_subplot(111, projection='3d')
                ax.set_proj_type('ortho')   # <<< keeps ticks glued to axes


                traj_array    = traj[0][0]        # (15, 200, 7) DU
                mc_traj_array = mc_traj[0][0]     # (1000, 200, 7) DU
                r_sigma0 = traj_array[:, 0, :3]
                r_mc0    = mc_traj_array[:, 0, :3]

                # SP mean/cov at t0 (position block only)
                mu_pos_t0 = mu_sigma_list[0][0, 0, :3]
                Ppos_t0   = P_sigma_list[0][0, 0, :3, :3]
                # MC empirical mean/cov at t0
                mu_mc_t0  = mu_mc[0,0,0,:3]
                P_mc_t0   = P_mc[0,0,0,:3,:3]

                # Main scatter (grayscale)
                ax.scatter(r_mc0[:,0], r_mc0[:,1], r_mc0[:,2], c='0.55', s=12, alpha=0.25,
                           label='Monte Carlo', zorder=3)
                ax.scatter(r_sigma0[:,0], r_sigma0[:,1], r_sigma0[:,2], c='0.0',  s=26, alpha=1.0,
                           label='Sigma Points', zorder=4)

                # MAIN AXIS LIMITS FROM POINTS (not ellipsoids)
                set_axes_equal(ax)
                set_max_ticks_exact(ax,4)
                expand_xyz_labels_no_offset(ax, fmt=".7f")
                
                # Move axis labels away from tick labels
                ax.xaxis.labelpad = 26   # distance in points
                ax.yaxis.labelpad = 33
                ax.zaxis.labelpad = 26
                ax.set_xlabel("X [DU]"); ax.set_ylabel("Y [DU]"); ax.set_zlabel("Z [DU]")
                legend_items = [
                    Line2D([0],[0], marker='o', color='none', markerfacecolor='0.55', markersize=6, label='Monte Carlo'),
                    Line2D([0],[0], marker='o', color='none', markerfacecolor='0.0',  markersize=7, label='Sigma Points'),
                    Patch(facecolor='none', edgecolor='0.35', label='SP 3σ'),
                    Patch(facecolor='none', edgecolor='0.7',  label='MC 3σ'),
                ]
                ax.legend(handles=legend_items, loc='upper left')

                # Inset (top-left, lighter alpha)
                add_ellipsoid_inset(ax, mu_pos_t0, Ppos_t0, mu_mc_t0, P_mc_t0,
                    r_sp=r_sigma0, r_mc=r_mc0,
                    rect=(0.62, 0.72, 0.34, 0.34))
                ax.tick_params(pad=13)
                plt.savefig(f"{out_dir}/scatter_initial_positions.pdf", dpi=600, bbox_inches='tight', pad_inches=0.5)
                plt.close()

                sp0_DU3, sp0_km3 = ellipsoid_3sigma_volume(Ppos_t0, DU_km)
                mc0_DU3, mc0_km3 = ellipsoid_3sigma_volume(P_mc_t0, DU_km)

                # ---------- Scatter: Final (tf) with 3σ inset ----------
                final_index = traj.shape[3] - 1
                r_sp_final  = traj[0, 0, :, final_index, :3]
                r_mc_final  = mc_traj[0, 0, :, final_index, :3]

                # SP mean/cov at tf
                mu_pos_tf = mu_sigma_list[0][0, -1, :3]
                Ppos_tf   = P_sigma_list[0][0, -1, :3, :3]
                # MC empirical mean/cov at tf
                mu_mc_tf = mu_mc[0,0,-1,:3]
                P_mc_tf  = P_mc[0,0,-1,:3,:3]

                fig = plt.figure(figsize=(6.5, 5.5))
                ax  = fig.add_subplot(111, projection='3d')
                ax.set_proj_type('ortho')   # <<< keeps ticks glued to axes


                ax.scatter(r_mc_final[:,0], r_mc_final[:,1], r_mc_final[:,2], s=12, c='0.55', alpha=0.25,
                           label='Monte Carlo', zorder=3)
                ax.scatter(r_sp_final[:,0], r_sp_final[:,1], r_sp_final[:,2], s=26, c='0.0',alpha=1,
                           label='Sigma Points', zorder=4)

                # MAIN AXIS LIMITS FROM POINTS
                set_axes_equal(ax)
                set_max_ticks_exact(ax,4)
                expand_xyz_labels_no_offset(ax, fmt=".6f")

                # Move axis labels away from tick labels
                ax.xaxis.labelpad = 26   # distance in points
                ax.yaxis.labelpad = 33
                ax.zaxis.labelpad = 26
                ax.set_xlabel("X [DU]"); ax.set_ylabel("Y [DU]"); ax.set_zlabel("Z [DU]")
                legend_items = [
                    Line2D([0],[0], marker='o', color='none', markerfacecolor='0.55', markersize=6, label='Monte Carlo'),
                    Line2D([0],[0], marker='o', color='none', markerfacecolor='0.0',  markersize=7, label='Sigma Points'),
                    Patch(facecolor='none', edgecolor='0.35', label='SP 3σ'),
                    Patch(facecolor='none', edgecolor='0.7',  label='MC 3σ'),
                ]
                ax.legend(handles=legend_items, loc='upper left')

                add_ellipsoid_inset(ax, mu_pos_tf, Ppos_tf, mu_mc_tf, P_mc_tf,
                    r_sp=r_sp_final, r_mc=r_mc_final,
                    rect=(0.62, 0.72, 0.34, 0.34))
                ax.tick_params(pad=13)
                plt.savefig(f"{out_dir}/scatter_final_positions.pdf", dpi=600, bbox_inches='tight', pad_inches=0.5)
                plt.close()

                spF_DU3, spF_km3 = ellipsoid_3sigma_volume(Ppos_tf, DU_km)
                mcF_DU3, mcF_km3 = ellipsoid_3sigma_volume(P_mc_tf, DU_km)

                # ---------- Diagnostics ----------
                P_sigma = P_sigma_list[0]
                P_init_diag = np.diag(P_pos)
                P_sigma_start_diag = np.diag(P_sigma[0, 0, :3, :3])
                print(f"[DEBUG] {label.upper()} / {mode} / Bundle {bundle_idx} — Segment Start Covariance Check:")
                print(f"    Empirical Sigma Cov Diag at t0: {P_sigma_start_diag}")
                print(f"    Expected P_init Diag:           {P_init_diag}")
                print(f"    Ratio:                          {P_sigma_start_diag / P_init_diag}")

                mu_sigma = mu_sigma_list[0]
                kl_vals = [compute_kl_divergence(mu_sigma[0, t], P_sigma[0, t], mu_mc[0, 0, t], P_mc[0, 0, t])
                           for t in range(P_sigma.shape[1])]

                # Mahalanobis distance squared per timestep
                for t in range(P_sigma.shape[1]):
                    diff = mu_sigma[0, t] - mu_mc[0, 0, t]
                    cov_inv = inv(P_mc[0, 0, t] + 1e-12 * np.eye(7))
                    d2 = diff.T @ cov_inv @ diff
                    print(f"[MAHAL@t={t:02d}] Mahalanobis² = {d2:.3f} — |μ_SP − μ_MC| = {np.linalg.norm(diff):.3e}")

                np.savetxt(f"{out_dir}/kl_divergence.txt", kl_vals, fmt="%.18f")
                np.savetxt(f"{out_dir}/cov_sigma_final.txt", P_sigma[0, -1], fmt="%.18f")
                np.savetxt(f"{out_dir}/cov_mc_final.txt", P_mc[0, 0, -1], fmt="%.18f")

                print(f"MC: {np.diag(P_mc[0,0,-1])}")
                print(f"SP: {np.diag(P_sigma[0,-1])}")

                delta_mu = mu_sigma - mu_mc
                mean_error_norm = np.linalg.norm(delta_mu)
                print(f"[DEBUG] Mean difference norm: {mean_error_norm:.6f}")

                cov_norms = []
                for t in range(P_sigma.shape[1]):
                    P_sp = P_sigma[0, t]
                    P_mc_t = P_mc[0, 0, t]
                    cov_diff = P_sp - P_mc_t
                    frob_norm = np.linalg.norm(cov_diff, ord='fro')
                    cov_norms.append(frob_norm)
                    print(f"[COV NORM @ t={t:03d}] ‖ΔP‖_F = {frob_norm:.3e}")

                for t in range(P_sigma.shape[1]):
                    mu_sp_t = mu_sigma[0, t]
                    mu_mc_t = mu_mc[0, 0, t]
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
                mahal = [np.sqrt((x[-1, :7] - sigma0_final) @ inv(P_final+ 1e-6*np.eye(7)) @ (x[-1, :7] - sigma0_final)) for x in trajs[1:]]
                np.savetxt(f"{out_dir}/mahalanobis_distances.txt", mahal, fmt="%.12f")
                with open(f"{out_dir}/ellipsoid_volumes.txt", "w") as f:
                    f.write("ellipsoid volume (position only)\n")
                    f.write(f"Initial SP: {sp0_DU3:.6f} DU^3  ({sp0_km3:.6f} km^3)\n")
                    f.write(f"Initial MC: {mc0_DU3:.6f} DU^3  ({mc0_km3:.6f} km^3)\n")
                    f.write(f"Final   SP: {spF_DU3:.6f} DU^3  ({spF_km3:.6f} km^3)\n")
                    f.write(f"Final   MC: {mcF_DU3:.6f} DU^3  ({mcF_km3:.6f} km^3)\n")


        runtime = time.time() - start_time
        with open(f"{out_root}/runtime.txt", "w") as f:
            f.write(f"{runtime:.2f}")

if __name__ == "__main__":
    main()
