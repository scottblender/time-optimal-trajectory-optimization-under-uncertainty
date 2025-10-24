import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import chi2, norm  # <-- Added chi2 and norm
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['LGBM_LOGLEVEL'] = '2'

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
try:
    from mee2rv import mee2rv
    from rv2mee import rv2mee
    from odefunc import odefunc
except ImportError:
    print("[ERROR] Failed to import helper functions (mee2rv, rv2mee, odefunc).")
    print("Please ensure they are in the 'helpers' directory adjacent to this script.")
    # Define dummy functions to allow script to load, but it will fail on run
    def mee2rv(*args): raise ImportError("Missing mee2rv implementation")
    def rv2mee(*args): raise ImportError("Missing rv2mee implementation")
    def odefunc(*args): raise ImportError("Missing odefunc implementation")

# --- MISSION PARAMETERS & CONSTANTS ---
DU_km = 696340.0
DEVIATION_THRESHOLD_KM = 25.0
REPLAN_COOLDOWN_STEPS = 50
NUM_MC_SAMPLES = 1000

# ==============================================================================
# === THRUST HELPER FUNCTION ===================================================
# ==============================================================================

def compute_thrust_direction(mu, mee, lam):
    """Computes the optimal thrust direction vector from costates."""
    p, f, g, h, k, L = mee
    # lam is (7,), but we only need the first 6 (p,f,g,h,k,L costates)
    lam_p, lam_f, lam_g, lam_h, lam_k, lam_L = lam[:6]
    
    lam_matrix = np.array([[lam_p, lam_f, lam_g, lam_h, lam_k, lam_L]]).T
    SinL, CosL = np.sin(L), np.cos(L)
    w = 1 + f * CosL + g * SinL
    
    if np.isclose(w, 0, atol=1e-10): return np.full(3, np.nan)
    
    s = 1 + h**2 + k**2
    C1 = np.sqrt(p / mu)
    C2 = 1 / w
    C3 = h * SinL - k * CosL
    
    A = np.array([
        [0, 2 * p * C2 * C1, 0],
        [C1 * SinL, C1 * C2 * ((w + 1) * CosL + f), -C1 * (g / w) * C3],
        [-C1 * CosL, C1 * C2 * ((w + 1) * SinL + g), C1 * (f / w) * C3],
        [0, 0, C1 * s * CosL * C2 / 2],
        [0, 0, C1 * s * SinL * C2 / 2],
        [0, 0, C1 * C2 * C3]
    ])
    
    mat = A.T @ lam_matrix
    norm = np.linalg.norm(mat)
    
    return mat.flatten() / norm if norm > 0 else np.full(3, np.nan)

# ==============================================================================
# === 3-SIGMA MC SAMPLING HELPER ===============================================
# ==============================================================================

def get_3sigma_mvn_samples(mean, cov, n_samples):
    """
    Generates n_samples from a multivariate normal distribution,
    ensuring all are within the 3-sigma (99.73%) ellipsoid
    using batched rejection sampling.
    """
    k = len(mean) # Number of dimensions
    if k == 0:
        raise ValueError("Mean vector is empty, cannot sample.")

    # 1. Find the 3-sigma distance threshold
    # This finds the squared Mahalanobis distance that contains 99.73% of the data
    prob_3sigma = norm.cdf(3) * 2 - 1 
    threshold_sq = chi2.ppf(prob_3sigma, df=k)
    
    # Pre-calculate for the distance check
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError as e:
        print(f"[ERROR] Covariance matrix is singular. Cannot compute 3-sigma samples. {e}")
        # Fallback: return n_samples of the mean
        return np.tile(mean, (n_samples, 1))

    accepted_samples = []
    num_accepted = 0
    max_loops = 2000 # Safety break to prevent infinite loops

    # 2. Keep sampling in batches until we have enough
    for _ in range(max_loops):
        if num_accepted >= n_samples:
            break
            
        n_needed = n_samples - num_accepted
        # Make a smart guess for batch size (10% buffer + 10 samples)
        n_guess = int(np.ceil(n_needed / prob_3sigma) * 1.1) + 10 
        
        try:
            candidates = np.random.multivariate_normal(mean, cov, size=n_guess)
        except Exception as e:
             print(f"[ERROR] Failed to generate multivariate_normal samples: {e}")
             continue # Try again

        # Handle 1D case from sampling
        if candidates.ndim == 1 and n_guess == 1:
            candidates = candidates.reshape(1, -1)
        
        # 3. Check which samples are "in the bubble"
        delta = candidates - mean
        # Calculate squared Mahalanobis distance
        dist_sq = np.sum((delta @ inv_cov) * delta, axis=1)
        
        # Keep only the samples inside the threshold
        new_accepted = candidates[dist_sq <= threshold_sq]
        
        if new_accepted.shape[0] > 0:
            accepted_samples.append(new_accepted)
            num_accepted += new_accepted.shape[0]
    
    if num_accepted < n_samples:
        print(f"[WARN] Hit max sampling loops. Returning {num_accepted} samples instead of {n_samples}.")

    # 4. Return the exact number of samples requested
    if not accepted_samples:
         print("[WARN] No samples were accepted. Returning mean values.")
         return np.tile(mean, (n_samples, 1))

    final_samples = np.vstack(accepted_samples)
    return final_samples[:n_samples, :]

# ==============================================================================
# === COORDINATE HELPER FUNCTIONS (RCN) ========================================
# ==============================================================================

def get_rcn_frame(r_hist, v_hist):
    """
    Computes the RCN (Radial, Cross-track, Normal) frame for a trajectory.
    r_hist: (N, 3) Cartesian position history
    v_hist: (N, 3) Cartesian velocity history
    Returns: (r_dir, c_dir, n_dir), each (N, 3)
    """
    # Radial direction (R)
    r_norm = np.linalg.norm(r_hist, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        r_dir = np.nan_to_num(r_hist / r_norm)
    
    # Angular momentum vector (h = r x v)
    h_vec = np.cross(r_hist, v_hist)
    h_norm = np.linalg.norm(h_vec, axis=1, keepdims=True)
    
    # Normal direction (N) (or W)
    with np.errstate(invalid='ignore', divide='ignore'):
        n_dir = np.nan_to_num(h_vec / h_norm)
    
    # Cross-track direction (C) (or S)
    # c_dir = n_dir x r_dir
    c_dir = np.cross(n_dir, r_dir)
    
    return r_dir, c_dir, n_dir

def cart_to_rcn_pos(cart_pos, r_dir, c_dir, n_dir):
    """Projects Cartesian position history (N, 3) into an RCN frame."""
    r_comp = np.einsum('ij,ij->i', cart_pos, r_dir)
    c_comp = np.einsum('ij,ij->i', cart_pos, c_dir)
    n_comp = np.einsum('ij,ij->i', cart_pos, n_dir)
    return np.stack([r_comp, c_comp, n_comp], axis=1)

def cart_to_rcn_vec(cart_vec, r_dir, c_dir, n_dir):
    """Projects a Cartesian vector history (e.g., thrust) (N, 3) into an RCN frame."""
    r_comp = np.einsum('ij,ij->i', cart_vec, r_dir)
    c_comp = np.einsum('ij,ij->i', cart_vec, c_dir)
    n_comp = np.einsum('ij,ij->i', cart_vec, n_dir)
    return np.stack([r_comp, c_comp, n_comp], axis=1)

# ==============================================================================
# === PLOTTING FUNCTION ========================================================
# ==============================================================================

def plot_final_deviations_rcn_and_thrust(t_eval, history_optimal, history_corrective, sol_nom, sol_nom_perturbed, mu, DU_km, window_type, covariance_multiplier, replan_times):
    """
    Plots the final RCN deviation and Thrust component comparison.
    This function is robust to integration failures by clipping plots to the
    minimum common trajectory length.
    """
    
    # --- Set larger font sizes for conference paper readability ---
    font_size = 18
    original_rc_params = plt.rcParams.copy() # Save original settings
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size + 2,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size - 2,
        'ytick.labelsize': font_size - 2,
        'legend.fontsize': font_size - 2,
        'figure.titlesize': font_size + 4,
    })

    # --- 1. Process Nominal Trajectory (The Reference) ---
    n_nom = sol_nom.y.shape[1]
    if n_nom < 2:
        print("[ERROR] Nominal trajectory failed to propagate. Skipping plot.")
        plt.rcParams.update(original_rc_params) # Reset params before exiting
        return
        
    t_nom = t_eval[:n_nom]
    r_nom_hist_cart, v_nom_hist_cart = mee2rv(
        sol_nom.y[0, :], sol_nom.y[1, :], sol_nom.y[2, :], 
        sol_nom.y[3, :], sol_nom.y[4, :], sol_nom.y[5, :], mu
    ) # (n_nom, 3)
    
    r_dir_nom, c_dir_nom, n_dir_nom = get_rcn_frame(r_nom_hist_cart, v_nom_hist_cart) # (n_nom, 3)
    rcn_nom_hist = cart_to_rcn_pos(r_nom_hist_cart, r_dir_nom, c_dir_nom, n_dir_nom) # (n_nom, 3)
    
    u_nom_cart_list = []
    for i in range(n_nom):
        mee = sol_nom.y[0:6, i] # (6,)
        lam = sol_nom.y[7:14, i] # (7,)
        u_vec = compute_thrust_direction(mu, mee, lam)
        u_nom_cart_list.append(u_vec)
    u_nom_cart = np.array(u_nom_cart_list) # (n_nom, 3)
    u_nom_rcn = cart_to_rcn_vec(u_nom_cart, r_dir_nom, c_dir_nom, n_dir_nom) # (n_nom, 3)

    # --- 2. Process Perturbed Trajectory ---
    n_pert = sol_nom_perturbed.y.shape[1]
    n_plot_pert = min(n_nom, n_pert)
    
    t_pert = t_eval[:n_plot_pert]
    r_pert_hist_cart, _ = mee2rv(
        sol_nom_perturbed.y[0, :n_plot_pert], sol_nom_perturbed.y[1, :n_plot_pert], sol_nom_perturbed.y[2, :n_plot_pert], 
        sol_nom_perturbed.y[3, :n_plot_pert], sol_nom_perturbed.y[4, :n_plot_pert], sol_nom_perturbed.y[5, :n_plot_pert], mu
    ) # (n_plot_pert, 3)
    
    rcn_perturbed_hist = cart_to_rcn_pos(
        r_pert_hist_cart, 
        r_dir_nom[:n_plot_pert], c_dir_nom[:n_plot_pert], n_dir_nom[:n_plot_pert]
    ) # (n_plot_pert, 3)
    delta_r_perturbed = (rcn_perturbed_hist - rcn_nom_hist[:n_plot_pert]) # (n_plot_pert, 3)
    
    u_perturbed_cart_list = []
    for i in range(n_plot_pert):
        mee = sol_nom_perturbed.y[0:6, i]
        lam = sol_nom_perturbed.y[7:14, i]
        u_vec = compute_thrust_direction(mu, mee, lam)
        u_perturbed_cart_list.append(u_vec)
    u_perturbed_cart = np.array(u_perturbed_cart_list) # (n_plot_pert, 3)
    
    u_perturbed_rcn = cart_to_rcn_vec(
        u_perturbed_cart,
        r_dir_nom[:n_plot_pert], c_dir_nom[:n_plot_pert], n_dir_nom[:n_plot_pert]
    ) # (n_plot_pert, 3)

    # --- 3. Process Optimal (Open-Loop) Trajectory ---
    n_opt = len(history_optimal)
    n_plot_opt = min(n_nom, n_opt)
    
    t_opt = t_eval[:n_plot_opt]
    mean_states_optimal = np.mean(history_optimal[:n_plot_opt], axis=1)   # (n_plot_opt, 14)
    r_opt_hist_cart, _ = mee2rv(
        mean_states_optimal[:, 0], mean_states_optimal[:, 1], mean_states_optimal[:, 2], 
        mean_states_optimal[:, 3], mean_states_optimal[:, 4], mean_states_optimal[:, 5], mu
    ) # (n_plot_opt, 3)

    rcn_optimal_hist = cart_to_rcn_pos(
        r_opt_hist_cart,
        r_dir_nom[:n_plot_opt], c_dir_nom[:n_plot_opt], n_dir_nom[:n_plot_opt]
    ) # (n_plot_opt, 3)
    delta_r_optimal = (rcn_optimal_hist - rcn_nom_hist[:n_plot_opt]) # (n_plot_opt, 3)
    
    u_optimal_cart_list = []
    for i in range(n_plot_opt):
        mee = mean_states_optimal[i, 0:6]
        lam = mean_states_optimal[i, 7:14]
        u_vec = compute_thrust_direction(mu, mee, lam)
        u_optimal_cart_list.append(u_vec)
    u_optimal_cart = np.array(u_optimal_cart_list) # (n_plot_opt, 3)
    
    u_optimal_rcn = cart_to_rcn_vec(
        u_optimal_cart,
        r_dir_nom[:n_plot_opt], c_dir_nom[:n_plot_opt], n_dir_nom[:n_plot_opt]
    ) # (n_plot_opt, 3)

    # --- 4. Process Corrective (Closed-Loop) Trajectory ---
    n_corr = len(history_corrective)
    n_plot_corr = min(n_nom, n_corr)
    
    t_corr = t_eval[:n_plot_corr]
    mean_states_corrective = np.mean(history_corrective[:n_plot_corr], axis=1) # (n_plot_corr, 14)
    r_corr_hist_cart, _ = mee2rv(
        mean_states_corrective[:, 0], mean_states_corrective[:, 1], mean_states_corrective[:, 2], 
        mean_states_corrective[:, 3], mean_states_corrective[:, 4], mean_states_corrective[:, 5], mu
    ) # (n_plot_corr, 3)
    
    rcn_corrective_hist = cart_to_rcn_pos(
        r_corr_hist_cart,
        r_dir_nom[:n_plot_corr], c_dir_nom[:n_plot_corr], n_dir_nom[:n_plot_corr]
    ) # (n_plot_corr, 3)
    delta_r_corrective = (rcn_corrective_hist - rcn_nom_hist[:n_plot_corr]) # (n_plot_corr, 3)
    
    u_corrective_cart_list = []
    for i in range(n_plot_corr):
        mee = mean_states_corrective[i, 0:6]
        lam = mean_states_corrective[i, 7:14]
        u_vec = compute_thrust_direction(mu, mee, lam)
        u_corrective_cart_list.append(u_vec)
    u_corrective_cart = np.array(u_corrective_cart_list) # (n_plot_corr, 3)
    
    u_corrective_rcn = cart_to_rcn_vec(
        u_corrective_cart,
        r_dir_nom[:n_plot_corr], c_dir_nom[:n_plot_corr], n_dir_nom[:n_plot_corr]
    ) # (n_plot_corr, 3)

    # --- 5. Apply Scaling for Plotting ---
    delta_r_perturbed *= DU_km
    delta_r_optimal *= DU_km
    delta_r_corrective *= DU_km

    # --- 6. Plotting ---
    fig, axes = plt.subplots(3, 2, figsize=(20, 18), sharex=True)
    # No suptitle as requested

    dev_labels = ['Radial (R) Deviation [km]', 'Cross-track (C) Deviation [km]', 'Normal (N) Deviation [km]']
    thrust_labels = ['Thrust Component ($u_r$)', 'Thrust Component ($u_c$)', 'Thrust Component ($u_n$)']
    
    # Grayscale colors and styles for conference paper
    c_pert = '0.7'      # Light gray
    ls_pert = '-'       # Solid
    lbl_pert = 'Perturbed Nominal (0.98F)'
    
    c_opt = '0.4'       # Medium gray
    ls_opt = '--'       # Dashed
    lbl_opt = 'Optimal (Open-Loop)'
    
    c_corr = '0.0'      # Black
    ls_corr = ':'       # Dotted
    lbl_corr = 'Corrective (Closed-Loop)'

    for i in range(3):
        # Left Column: Deviations
        ax_dev = axes[i, 0]
        ax_dev.plot(t_pert, delta_r_perturbed[:, i], color=c_pert, linestyle=ls_pert, lw=2, label=lbl_pert)
        ax_dev.plot(t_opt, delta_r_optimal[:, i], color=c_opt, linestyle=ls_opt, lw=2, label=lbl_opt)
        ax_dev.plot(t_corr, delta_r_corrective[:, i], color=c_corr, linestyle=ls_corr, lw=2.5, label=lbl_corr)
        ax_dev.set_ylabel(dev_labels[i])
        ax_dev.grid(True, linestyle=':')
        
        # Right Column: Thrust Components
        ax_thrust = axes[i, 1]
        ax_thrust.plot(t_pert, u_perturbed_rcn[:, i], color=c_pert, linestyle=ls_pert, lw=2, label=lbl_pert)
        ax_thrust.plot(t_opt, u_optimal_rcn[:, i], color=c_opt, linestyle=ls_opt, lw=2, label=lbl_opt)
        ax_thrust.plot(t_corr, u_corrective_rcn[:, i], color=c_corr, linestyle=ls_corr, lw=2.5, label=lbl_corr)
        ax_thrust.set_ylabel(thrust_labels[i])
        ax_thrust.grid(True, linestyle=':')
        
        # Add replan markers
        if replan_times:
            for t_replan in replan_times:
                # Check if the replan time is within the plotted range for the corrective trajectory
                if t_corr.size > 0 and t_replan <= t_corr[-1]:
                    # Find the index in the corrective time array (t_corr) closest to the replan time
                    idx = np.argmin(np.abs(t_corr - t_replan))
                    
                    # Get the y-value from the corrective trajectory at that index
                    y_dev = delta_r_corrective[idx, i]
                    y_thrust = u_corrective_rcn[idx, i]
                    
                    # Plot a large black 'x' marker ON the corrective trajectory
                    ax_dev.plot(t_replan, y_dev, 'kx', markersize=12, markeredgewidth=2.5, 
                                label='Replanning Event' if i == 0 and t_replan == replan_times[0] else None)
                    ax_thrust.plot(t_replan, y_thrust, 'kx', markersize=12, markeredgewidth=2.5)

        if i == 0:
            ax_dev.legend()
        
    axes[-1, 0].set_xlabel('Time [TU]')
    axes[-1, 1].set_xlabel('Time [TU]')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fname = f"final_rcn_and_thrust_{window_type}_cov{covariance_multiplier}.pdf"
    plt.savefig(os.path.join("uncertainty_aware_outputs", fname))
    plt.close()
    
    # Reset matplotlib parameters to default
    plt.rcParams.update(original_rc_params)
    
    print(f"\n[Saved Plot] {fname}")

# ==============================================================================
# === CORE PROPAGATION AND CONTROL LOGIC =======================================
# ==============================================================================

def propagate_and_control(models, strategy, t_eval, data, sol_nom, initial_mc_states, window_type, covariance_multiplier=1.0):
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    F_val = 0.98 * F_nom # Propagate with perturbed thrust
    mc_states_current = np.copy(initial_mc_states) # Shape (N_MC, 14)

    feature_cols_optimal = ['t', 'p', 'f', 'g', 'h', 'k', 'L', 'mass', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
    
    feature_cols_corrective = feature_cols_optimal + [
        'delta_rx', 'delta_ry', 'delta_rz', 'delta_vx', 'delta_vy', 'delta_vz',
        'delta_m', 't_go'
    ]

    # --- Initial open-loop "optimal" policy application (Control the Mean) ---
    tqdm.write(f"\n[INFO] Calculating initial control policy for {strategy.upper()} strategy...")
    initial_mee_states = mc_states_current[:, :7] # MEEs + mass
    mean_initial_state = np.mean(initial_mee_states, axis=0)
    initial_devs = initial_mee_states - mean_initial_state
    initial_cov = np.einsum('ji,jk->ik', initial_devs, initial_devs) / (len(initial_mee_states) - 1)
    
    # --- KEY CHANGE: Scale the *measured* covariance by the multiplier ---
    # This "tells" the model what it should *believe* the covariance is.
    initial_diag_vals = np.diag(initial_cov) * covariance_multiplier
    if covariance_multiplier != 1.0:
        tqdm.write(f"[INFO] Applying covariance multiplier (x{covariance_multiplier}) to model features.")
    # --- END OF CHANGE ---
    
    model_optimal = models['optimal']
    features_opt = np.hstack([t_eval[0], mean_initial_state, initial_diag_vals])
    predicted_lam_optimal = model_optimal.predict(pd.DataFrame([features_opt], columns=feature_cols_optimal))[0] # (7,)
    mc_states_current[:, 7:] = predicted_lam_optimal # Apply same control to all samples
    history_mc_states = [np.copy(mc_states_current)]
    steps_since_replan = REPLAN_COOLDOWN_STEPS
    replan_times = [] # List to store times of replanning
    
    tqdm.write(f"--- Propagating with {strategy.upper()} Strategy (Model sees Cov x{covariance_multiplier}) ---")
    tqdm.write(f"{'Step':<5} | {'Time (TU)':<12} | {'Deviation (km)':<18} | {'Ellipsoid Vol (km^3)':<22} | {'Action':<10}")
    tqdm.write("-" * 80)

    for i in tqdm(range(len(t_eval) - 1), desc=f"    -> Propagating ({strategy})", leave=False):
        t_start_step, t_end_step = t_eval[i], t_eval[i+1]
        
        next_states_list = []
        for s in mc_states_current:
            try:
                sol_step = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0), 
                                     [t_start_step, t_end_step], s, t_eval=[t_end_step], 
                                     rtol=1e-8, atol=1e-10)
                if sol_step.success and len(sol_step.y[0]) > 0:
                     next_states_list.append(sol_step.y[:, -1])
                else:
                    next_states_list.append(s)
            except Exception as e:
                print(f"\n[ERROR] Integrator failed at step {i}, t={t_start_step}. Error: {e}")
                print(f"State: {s}")
                next_states_list.append(s)
        
        if not next_states_list:
            tqdm.write(f"\n[ERROR] All integrators failed at step {i}. Stopping propagation.")
            break
            
        mc_states_current = np.array(next_states_list)
        
        if sol_nom.y.shape[1] <= i+1:
            tqdm.write(f"\n[WARN] Nominal solution ended prematurely. Stopping propagation.")
            break

        current_mc_positions, _ = mee2rv(
            mc_states_current[:, 0], mc_states_current[:, 1], mc_states_current[:, 2], 
            mc_states_current[:, 3], mc_states_current[:, 4], mc_states_current[:, 5], mu
        ) # (N, 3)
        
        r_nom_step, _ = mee2rv(*sol_nom.y[:6, i+1], mu)
             
        mean_mc_pos = np.mean(current_mc_positions, axis=0) # (3,)
        deviation_km = np.linalg.norm(mean_mc_pos - r_nom_step.flatten()) * DU_km
        
        if current_mc_positions.shape[0] > 1:
            cov_cartesian = np.cov(current_mc_positions.T)
            eigvals = np.maximum(np.linalg.eigvalsh(cov_cartesian), 0)
            volume_km3 = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals)) * (DU_km**3)
        else:
            volume_km3 = 0.0
            
        log_action = "---"

        if deviation_km > DEVIATION_THRESHOLD_KM and strategy == 'corrective' and steps_since_replan >= REPLAN_COOLDOWN_STEPS:
            log_action = "REPLAN"
            steps_since_replan = 0
            replan_times.append(t_end_step) # Store the time of replan
            
            # sol_nom.y is (14, N)
            mee_nom_full_curr = sol_nom.y[:7, i+1] # (7,) MEEs + mass
            lam_nom_curr = sol_nom.y[7:14, i+1]   # (7,) Costates
            
            r_nom_curr, v_nom_curr = mee2rv(*mee_nom_full_curr[:6], mu) # (3,)
            r_nom_curr = r_nom_curr.flatten()
            v_nom_curr = v_nom_curr.flatten()
            
            mee_states_current = mc_states_current[:, :7]
            mean_mee_state_curr = np.mean(mee_states_current, axis=0)
            
            if mee_states_current.shape[0] > 1:
                devs_current = mee_states_current - mean_mee_state_curr
                # --- SECOND KEY CHANGE: Scale the *current* covariance ---
                cov_current = np.einsum('ji,jk->ik', devs_current, devs_current) / (len(mc_states_current) - 1)
                diag_vals_current = np.diag(cov_current) * covariance_multiplier # <-- Modified line
                # --- END OF CHANGE ---
            else:
                diag_vals_current = np.zeros(7)

            r_mean_curr, v_mean_curr = mee2rv(*mean_mee_state_curr[:6], mu) # (3,)
            r_mean_curr = r_mean_curr.flatten()
            v_mean_curr = v_mean_curr.flatten()

            delta_r = r_nom_curr - r_mean_curr
            delta_v = v_nom_curr - v_mean_curr
            delta_m = mee_nom_full_curr[-1] - mean_mee_state_curr[-1]
            t_go = t_eval[-1] - t_end_step

            features_corr = np.hstack([
                t_end_step, mean_mee_state_curr, diag_vals_current, 
                delta_r.flatten(), delta_v.flatten(), delta_m, t_go
            ])
            
            model_corrective = models['corrective']
            lam_correction = model_corrective.predict(pd.DataFrame([features_corr], columns=feature_cols_corrective))[0] # (7,)
            
            # --- REVERTED CORRECTIVE LOGIC ---
            if window_type == "min":
                mc_states_current[:, 7:] = lam_nom_curr + lam_correction
            else: # max window
                mc_states_current[:, 7:] = lam_nom_curr - lam_correction
            # --- END OF REVERTED LOGIC ---

        history_mc_states.append(np.copy(mc_states_current))
        tqdm.write(f"{i+1:<5} | {t_end_step:<12.2f} | {deviation_km:<18.2f} | {volume_km3:<22.2e} | {log_action:<10}")
        steps_since_replan += 1

    return np.array(history_mc_states), replan_times

# ==============================================================================
# === MAIN SIMULATION RUNNER ===================================================
# ==============================================================================

def run_comparison_simulation(models, t_start_replan, t_end_replan, window_type, data, initial_mc_states, covariance_multiplier=1.0):
    print(f"\n{'='*25} Running Comparison for {window_type.upper()} WINDOW (Cov x{covariance_multiplier}) {'='*25}")
    
    mu, r_nom_hist, v_nom_hist = data["mu"], data["r_tr"], data["v_tr"]
    mass_nom_hist, lam_tr, t_vals = data["mass_tr"], data["lam_tr"], np.asarray(data["backTspan"][::-1])
    idx_start = np.argmin(np.abs(t_vals - t_start_replan))
    r0, v0, m0_val, initial_lam = r_nom_hist[idx_start], v_nom_hist[idx_start], mass_nom_hist[idx_start], lam_tr[idx_start]
    
    t_eval = np.linspace(t_start_replan, t_end_replan, 100)
    
    # sol_nom.y will be (14, N)
    nominal_state_start = np.hstack([rv2mee(r0.reshape(1,3), v0.reshape(1,3), mu).flatten(), m0_val, initial_lam])
    
    g0 = 9.81/1000 # km/s^2
    TU_s = np.sqrt(DU_km / g0) 
    print(f"[INFO] Using g0-based TU for conversions ({TU_s:.2f} s / TU), as in original script.")

    prop_time_tu = t_end_replan - t_start_replan
    prop_time_min = prop_time_tu * (TU_s / 60.0)
    v0_mag_canonical = np.linalg.norm(v0)
    v0_mag_kms = v0_mag_canonical * (DU_km / TU_s)
    
    print(f"\n[SIMULATION INFO]")
    print(f"  Start Time: {t_start_replan:.2f} TU  |  End Time: {t_end_replan:.2f} TU")
    print(f"  Propagation Time: {prop_time_tu:.2f} TU ({prop_time_min:.2f} minutes)")
    print(f"  Initial Velocity: {v0_mag_canonical:.4f} DU/TU ({v0_mag_kms:.4f} km/s)")

    print("\n[INFO] Propagating Nominal, Perturbed, and Controlled trajectories...")
    sol_nom = solve_ivp(lambda t, x: odefunc(t, x, data["mu"], data["F"], data["c"], data["m0"], data["g0"]), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval, rtol=1e-8, atol=1e-10)
    sol_nom_perturbed = solve_ivp(lambda t, x: odefunc(t, x, data["mu"], 0.98 * data["F"], data["c"], data["m0"], data["g0"]), [t_start_replan, t_end_replan], nominal_state_start, t_eval=t_eval, rtol=1e-8, atol=1e-10)

    # --- MC Sample Generation logic removed from here ---

    # --- Propagate using the provided initial_mc_states ---
    if initial_mc_states is None:
        print("[ERROR] No initial MC states provided. Bypassing MC propagation.")
        history_optimal = np.array([nominal_state_start])
        history_corrective = np.array([nominal_state_start])
        corrective_replan_times = []
    else:
        # Pass the covariance_multiplier to propagate_and_control
        history_corrective, corrective_replan_times = propagate_and_control(
            models, 'corrective', t_eval, data, sol_nom, initial_mc_states, window_type, covariance_multiplier
        )
        history_optimal, _ = propagate_and_control(
            models, 'optimal', t_eval, data, sol_nom, initial_mc_states, window_type, covariance_multiplier
        )


    plot_final_deviations_rcn_and_thrust(
        t_eval, history_optimal, history_corrective, sol_nom, sol_nom_perturbed, 
        data["mu"], DU_km, window_type, covariance_multiplier, corrective_replan_times
    )

    print(f"\n--- Performance Summary Table ({window_type.upper()} Window, Cov x{covariance_multiplier}) ---")
    
    if sol_nom.y.shape[1] < 2:
        print("[ERROR] Nominal solution failed to propagate. Cannot generate summary.")
        return
        
    r_final_nom, _ = mee2rv(*sol_nom.y[:6, -1], mu)
    
    def get_final_stats(history, name):
        if len(history) < 2:
            print(f"[WARN] {name} history is too short. Skipping stats.")
            return 0.0, 0.0
            
        final_states = history[-1]
        
        if final_states.ndim == 1:
             final_states = final_states.reshape(1, -1)
        
        if final_states.shape[0] < 1 or final_states.shape[1] < 14: # State is 14
             print(f"[WARN] {name} final states are invalid (shape {final_states.shape}). Skipping stats.")
             return 0.0, 0.0

        final_pos, _ = mee2rv(
            final_states[:, 0], final_states[:, 1], final_states[:, 2], 
            final_states[:, 3], final_states[:, 4], final_states[:, 5], mu
        ) # (N, 3)
        
        if final_pos.ndim == 1:
             final_pos = final_pos.reshape(1, 3) 

        if final_pos.shape[0] == 0:
            print(f"[WARN] {name} final position calculation failed. Skipping stats.")
            return 0.0, 0.0
            
        mean_final_pos = np.mean(final_pos, axis=0)
        
        if final_pos.shape[0] > 1:
            cov = np.cov(final_pos.T)
            eigvals = np.maximum(np.linalg.eigvalsh(cov), 0)
            vol = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals)) * (DU_km**3)
        else:
            vol = 0.0
        
        r_final_nom_flat = r_final_nom.flatten()
        final_dev = np.linalg.norm(mean_final_pos - r_final_nom_flat) * DU_km
        return final_dev, vol

    final_dev_opt, vol_opt = get_final_stats(history_optimal, "Optimal")
    final_dev_corr, vol_corr = get_final_stats(history_corrective, "Corrective")
    
    final_dev_pert = 0.0
    if sol_nom_perturbed.y.shape[1] > 1:
        r_final_pert, _ = mee2rv(*sol_nom_perturbed.y[:6, -1], mu)
        final_dev_pert = np.linalg.norm(r_final_pert.flatten() - r_final_nom.flatten()) * DU_km
    else:
        print("[WARN] Perturbed nominal solution failed. Deviation set to 0.")


    summary_data = {
        "Strategy": ["Optimal (Open-Loop)", "Corrective (Closed-Loop)", "Perturbed Nominal (0.98F)"],
        "Final Deviation (km)": [f"{final_dev_opt:.2f}", f"{final_dev_corr:.2f}", f"{final_dev_pert:.2f}"],
        "Final Ellipsoid Volume (km^3)": [f"{vol_opt:.2e}", f"{vol_corr:.2e}", "N/A"]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

# ==============================================================================
# === MAIN EXECUTION BLOCK =====================================================
# ==============================================================================
def main():
    try:
        models_max = { 'optimal': joblib.load("trained_model_max_optimal.pkl"), 'corrective': joblib.load("trained_model_max_corrective.pkl") }
        models_min = { 'optimal': joblib.load("trained_model_min_optimal.pkl"), 'corrective': joblib.load("trained_model_min_corrective.pkl") }
        data = joblib.load("stride_1440min/bundle_data_1440min.pkl")
        with open("stride_1440min/bundle_segment_widths.txt") as f:
            lines = f.readlines()[1:]
            times_arr = np.array([list(map(float, line.strip().split())) for line in lines])
    except FileNotFoundError as e:
        print(f"[ERROR] A required data file is missing: {e}. Please ensure all input files are present.")
        return
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred loading data: {e}")
        return

    os.makedirs("uncertainty_aware_outputs", exist_ok=True)
    time_vals = times_arr[:, 0]
    mu = data["mu"] # Get mu for sample generation
    
    # --- Define constants for sample generation ---
    g0 = 9.81/1000 # km/s^2
    TU_s = np.sqrt(DU_km / g0)
    P_cart_base = np.diag(np.concatenate([
        np.diag(np.eye(3) * 0.015 / (DU_km**2)), 
        np.diag(np.eye(3) * 1.5e-10 / ((DU_km / TU_s)**2)), 
        [1.5e-3 / (4000**2)]
    ]))
    P_cart_1x = 1.0 * P_cart_base # We always generate samples from the 1.0x covariance

    # --- MAX Window Setup & Sample Generation ---
    print("\n[INFO] Setting up MAX Window simulation...")
    max_idx = int(np.argmax(times_arr[:, 1]))
    t_start_max, t_end_replan_max = time_vals[max(0, max_idx - 1)], time_vals[max_idx]
    
    idx_start_max = np.argmin(np.abs(data["backTspan"][::-1] - t_start_max))
    r0_max, v0_max, m0_val_max, initial_lam_max = data["r_tr"][idx_start_max], data["v_tr"][idx_start_max], data["mass_tr"][idx_start_max], data["lam_tr"][idx_start_max]

    initial_mc_states_max = None
    try:
        cartesian_mean_max = np.hstack([r0_max, v0_max, m0_val_max])
        print(f"[INFO] Generating {NUM_MC_SAMPLES} MC samples for MAX window (1.0x Cov)...")
        cartesian_samples_max = get_3sigma_mvn_samples(cartesian_mean_max, P_cart_1x, n_samples=NUM_MC_SAMPLES)
        
        initial_mee_states_list = []
        for s in cartesian_samples_max:
            mee_state = rv2mee(s[:3].reshape(1,3), s[3:6].reshape(1,3), mu).flatten()
            initial_mee_states_list.append(np.hstack([mee_state, s[6]]))
        initial_mee_states_max = np.array(initial_mee_states_list)
        initial_mc_states_max = np.hstack([initial_mee_states_max, np.tile(initial_lam_max, (NUM_MC_SAMPLES, 1))])
        print(f"[INFO] ...sampling complete for MAX window.")
    except Exception as e:
        print(f"[ERROR] Failed during MC sampling for MAX window: {e}")
        return # Can't continue
    
    # Run Max Window simulations (pass the *same* samples to both)
    run_comparison_simulation(models_max, t_start_max, t_end_replan_max, 'max', data, initial_mc_states_max, covariance_multiplier=1.0)
    run_comparison_simulation(models_max, t_start_max, t_end_replan_max, 'max', data, initial_mc_states_max, covariance_multiplier=2.0)
    
    # --- MIN Window Setup & Sample Generation ---
    print("\n[INFO] Setting up MIN Window simulation...")
    search_end_idx = int(0.75 * len(times_arr))
    search_array = times_arr[:search_end_idx, 1]
    min_idx = np.argmin(search_array)
    t_start_min, t_end_replan_min = time_vals[max(0, min_idx - 1)], time_vals[min_idx]

    idx_start_min = np.argmin(np.abs(data["backTspan"][::-1] - t_start_min))
    r0_min, v0_min, m0_val_min, initial_lam_min = data["r_tr"][idx_start_min], data["v_tr"][idx_start_min], data["mass_tr"][idx_start_min], data["lam_tr"][idx_start_min]

    initial_mc_states_min = None
    try:
        cartesian_mean_min = np.hstack([r0_min, v0_min, m0_val_min])
        print(f"[INFO] Generating {NUM_MC_SAMPLES} MC samples for MIN window (1.0x Cov)...")
        cartesian_samples_min = get_3sigma_mvn_samples(cartesian_mean_min, P_cart_1x, n_samples=NUM_MC_SAMPLES)
        
        initial_mee_states_list = []
        for s in cartesian_samples_min:
            mee_state = rv2mee(s[:3].reshape(1,3), s[3:6].reshape(1,3), mu).flatten()
            initial_mee_states_list.append(np.hstack([mee_state, s[6]]))
        initial_mee_states_min = np.array(initial_mee_states_list)
        initial_mc_states_min = np.hstack([initial_mee_states_min, np.tile(initial_lam_min, (NUM_MC_SAMPLES, 1))])
        print(f"[INFO] ...sampling complete for MIN window.")
    except Exception as e:
        print(f"[ERROR] Failed during MC sampling for MIN window: {e}")
        return

    # Run Min Window simulations (pass the *same* samples to both)
    run_comparison_simulation(models_min, t_start_min, t_end_replan_min, 'min', data, initial_mc_states_min, covariance_multiplier=1.0)
    run_comparison_simulation(models_min, t_start_min, t_end_replan_min, 'min', data, initial_mc_states_min, covariance_multiplier=2.0)
    
    print("\n[SUCCESS] All comparison simulations complete.")

if __name__ == "__main__":
    main()