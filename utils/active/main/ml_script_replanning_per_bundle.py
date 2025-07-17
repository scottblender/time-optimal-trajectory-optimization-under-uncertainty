import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm
import pandas as pd

# === Helpers ===
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
from odefunc import odefunc

def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect([1.25, 1, 0.75])

def run_experiment(model, scaler, r_nom, v_nom, mass_nom, t_vals, mu, F_nom, c, m0, g0):
    tk_fracs = [0.75, 0.8, 0.85]
    cov_scales = [0.1, 1.0, 5.0]
    F_scales = [0.8, 1.0, 1.2]

    base_cov = np.block([
        [np.eye(3)*0.01, np.zeros((3, 3)), np.zeros((3, 1))],
        [np.zeros((3, 3)), np.eye(3)*0.0001, np.zeros((3, 1))],
        [np.zeros((1, 3)), np.zeros((1, 3)), np.array([[0.0001]])]
    ])

    feature_names = ['t', 'p', 'f', 'g', 'h', 'L', 'mass', 'dummy1',
                     'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']

    os.makedirs("uncertainty_aware_outputs", exist_ok=True)

    for tk_frac in tk_fracs:
        t_k = tk_frac * t_vals[-1]
        idx = np.argmin(np.abs(t_vals - t_k))
        r0, v0, m0_val = r_nom[idx], v_nom[idx], mass_nom[idx]
        state_nom = np.hstack([r0, v0, m0_val])

        fig = plt.figure(figsize=(6.5, 5.5))
        ax = fig.add_subplot(111, projection='3d')

        # Plot nominal trajectory
        ax.plot(r_nom[:, 0], r_nom[:, 1], r_nom[:, 2], color='black', linewidth=2.0, label="Nominal")

        for cov_scale in cov_scales:
            P_init = base_cov * cov_scale
            sample = np.random.multivariate_normal(state_nom, P_init)
            r_s, v_s, m_s = sample[:3], sample[3:6], sample[6]
            mee = np.hstack([rv2mee(r_s.reshape(1,3), v_s.reshape(1,3), mu).flatten(), m_s])

            # Predict control using model with proper feature names
            x_input = np.hstack([t_vals[idx], mee, np.zeros(7)])
            x_df = pd.DataFrame([x_input], columns=feature_names)
            x_scaled = scaler.transform(x_df)
            lam = model.predict(x_scaled)[0]

            for F_factor in F_scales:
                F_val = F_factor * F_nom
                S = np.hstack([mee, lam])
                sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0),
                                [t_vals[idx], t_vals[-1]], S,
                                t_eval=np.linspace(t_vals[idx], t_vals[-1], 100))
                r_traj, _ = mee2rv(*sol.y[:6], mu)

                label = f"Cov×{cov_scale:.1f}, F×{F_factor:.1f}"
                ax.plot(r_traj[:, 0], r_traj[:, 1], r_traj[:, 2], linestyle='--', linewidth=1.2, label=label)
                ax.scatter(r_traj[0, 0], r_traj[0, 1], r_traj[0, 2], color='black', marker='o', s=15)

        ax.set_title(f"$t_k$ = {tk_frac:.2f}")
        ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
        set_axes_equal(ax)
        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        out_path = f"uncertainty_aware_outputs/tk{int(tk_frac*100)}_variations.pdf"
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.5, dpi=600)
        plt.close()
        print(f"[Saved] {out_path}")

if __name__ == "__main__":
    model = joblib.load("trained_model.pkl")
    scaler = joblib.load("scaler.pkl")
    data = joblib.load("stride_4000min/bundle_data_4000min.pkl")

    r_nom = data["r_tr"]
    v_nom = data["v_tr"]
    mass_nom = data["mass_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]

    run_experiment(model, scaler, r_nom, v_nom, mass_nom, t_vals, mu, F_nom, c, m0, g0)
