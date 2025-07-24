import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.linalg import eigh
from scipy.integrate import solve_ivp
import torch
from torch import nn
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# === Setup ===
plt.rcParams.update({'font.size': 10})
warnings.filterwarnings("ignore", category=UserWarning)
device = torch.device("cpu")

# === Local imports ===
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from mee2rv import mee2rv
from rv2mee import rv2mee
from odefunc import odefunc

# === TCN Model Definition ===
class TCN_MANN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, kernel_size=3):
        super().__init__()
        layers = []
        dilation = 1
        for _ in range(num_layers):
            layers.append(nn.Conv1d(input_size, hidden_size, kernel_size, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU())
            input_size = hidden_size
            dilation *= 2
        self.tcn = nn.Sequential(*layers)
        self.controller = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, output_size, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.controller(x)
        return x.transpose(1, 2)

# === Utility Functions ===
def set_axes_equal(ax):
    x_limits, y_limits, z_limits = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    ranges = [abs(lim[1] - lim[0]) for lim in [x_limits, y_limits, z_limits]]
    centers = [np.mean(lim) for lim in [x_limits, y_limits, z_limits]]
    max_range = max(ranges) / 2
    ax.set_xlim3d([centers[0] - max_range, centers[0] + max_range])
    ax.set_ylim3d([centers[1] - max_range, centers[1] + max_range])
    ax.set_zlim3d([centers[2] - max_range, centers[2] + max_range])
    ax.set_box_aspect([1.25, 1, 0.75])
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

def plot_3sigma_ellipsoid(ax, mean, cov, color='gray', alpha=0.25, scale=3.0):
    cov = 0.5 * (cov + cov.T) + np.eye(3) * 1e-10
    vals, vecs = eigh(cov)
    if np.any(vals <= 0): return
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    radii = scale * np.sqrt(vals)
    u, v = np.linspace(0, 2*np.pi, 40), np.linspace(0, np.pi, 40)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    ellipsoid = np.stack((x, y, z), axis=-1) @ vecs.T + mean
    ax.plot_surface(ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2],
                    rstride=1, cstride=1, color=color, alpha=alpha, linewidth=0)
    ax.plot_wireframe(ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2],
                      rstride=5, cstride=5, color='k', alpha=0.2, linewidth=0.3)

def compute_thrust_direction(mu, F, mee, lam):
    p, f, g, h, k, L = mee[:-1]
    lam_p, lam_f, lam_g, lam_h, lam_k, lam_L = lam[:-1]
    lam_matrix = np.array([[lam_p, lam_f, lam_g, lam_h, lam_k, lam_L]]).T
    SinL, CosL = np.sin(L), np.cos(L)
    w = 1 + f * CosL + g * SinL
    if np.isclose(w, 0, atol=1e-10):
        return np.full(3, np.nan)
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
    return mat.flatten() / np.linalg.norm(mat)

# === Main Function ===
def main():
    model = TCN_MANN(input_size=15, hidden_size=128, output_size=7, num_layers=4).to(device)
    model.load_state_dict(torch.load("trained_model_tcn.pt", map_location=device))
    model.eval()

    scaler_X = joblib.load("scaler_tcn.pkl")
    scaler_y = joblib.load("scaler_tcn_y.pkl")
    data = joblib.load("stride_4000min/bundle_data_4000min.pkl")
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    r_nom, v_nom, mass_nom = data["r_tr"], data["v_tr"], data["mass_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    tf = t_vals[-1]
    os.makedirs("uncertainty_aware_outputs", exist_ok=True)

    t_fracs = [0.925, 0.95]
    diag_mins = np.array([9.098224e+01, 7.082445e-04, 6.788599e-04, 1.376023e-08, 2.346605e-08, 5.885859e-08, 1.000000e-04])
    diag_maxs = np.array([1.977489e+03, 6.013501e-03, 5.173225e-03, 4.284912e-04, 1.023625e-03, 6.818500e+00, 1.000000e-04])
    P_models = {'min': np.diag(diag_mins), 'max': np.diag(diag_maxs)}

    P_cart = np.block([
        [np.eye(3)*0.01,       np.zeros((3,3)), np.zeros((3,1))],
        [np.zeros((3,3)), np.eye(3)*0.0001,     np.zeros((3,1))],
        [np.zeros((1,3)), np.zeros((1,3)),      np.array([[0.0001]])]
    ])

    F_val = 0.9 * F_nom
    summary = []

    for t_frac in t_fracs:
        t_k = t_frac * tf
        print(f"[t_frac = {t_frac:.3f}] Replanning at time t_k = {t_k:.2f} TU")
        idx = np.argmin(np.abs(t_vals - t_k))
        r0, v0, m0_val = r_nom[idx], v_nom[idx], mass_nom[idx]
        state_k = np.hstack([r0, v0, m0_val])
        shared_samples = np.random.multivariate_normal(state_k, P_cart, size=1000)

        for level, P_model in P_models.items():
            print(f"  Level: {level} — propagating MC...")
            mc_endpoints = []
            fig = plt.figure(figsize=(6.5, 5.5))
            ax = fig.add_subplot(111, projection='3d')

            ax.plot(r_nom[:, 0], r_nom[:, 1], r_nom[:, 2], color='black', lw=2.0, label="Nominal")
            ax.scatter(*r0, color='black', s=20, label="Start", zorder=5)
            ax.scatter(*r_nom[-1], color='black', s=20, marker='X', label="End", zorder=5)

            for i, s in enumerate(shared_samples):
                try:
                    mee = np.hstack([rv2mee(s[:3].reshape(1, 3), s[3:6].reshape(1, 3), mu).flatten(), s[6]])
                    x_input = np.hstack([t_k, mee, np.diag(P_model)])
                    x_scaled = scaler_X.transform([x_input])
                    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).view(1, 1, -1).to(device)
                    with torch.no_grad():
                        lam = model(x_tensor)[:, -1, :].cpu().numpy()
                    lam = scaler_y.inverse_transform(lam)[0]
                    u_hat = compute_thrust_direction(mu, F_val, mee, lam)
                    if not np.isnan(u_hat).any():
                        ax.quiver(*s[:3], *u_hat, length=100, normalize=True,
                                color='0.5', linewidth=0.6, alpha=0.5)
                    S = np.hstack([mee, lam])
                    sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0),
                                    [t_k, tf], S, t_eval=np.linspace(t_k, tf, 100))
                    r, _ = mee2rv(*sol.y[:6], mu)
                    if i % 10 == 0:
                        ax.plot(r[:, 0], r[:, 1], r[:, 2], linestyle=':', color='0.6', lw=0.8, alpha=0.3)
                        ax.scatter(*r[0], color='0.6', s=5, alpha=0.25)
                        ax.scatter(*r[-1], color='0.6', s=5, marker='X', alpha=0.25)
                    mc_endpoints.append(r[-1])
                except:
                    continue

            mc_endpoints = np.array(mc_endpoints)
            mu_mc = np.mean(mc_endpoints, axis=0)
            cov_mc = np.cov(mc_endpoints.T)
            eigvals = np.maximum(np.linalg.eigvalsh(cov_mc), 0)
            volume = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals))
            plot_3sigma_ellipsoid(ax, mu_mc, cov_mc)

            # === Inset: zoom near final MC region
            inset_ax = fig.add_axes([0.5,0.63, 0.34, 0.34], projection='3d')
            for i, s in enumerate(shared_samples[::10]):
                try:
                    mee = np.hstack([rv2mee(s[:3].reshape(1, 3), s[3:6].reshape(1, 3), mu).flatten(), s[6]])
                    x_input = np.hstack([t_k, mee, np.diag(P_model)])
                    x_scaled = scaler_X.transform([x_input])
                    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).view(1, 1, -1).to(device)
                    with torch.no_grad():
                        lam = model(x_tensor)[:, -1, :].cpu().numpy()
                    lam = scaler_y.inverse_transform(lam)[0]
                    S = np.hstack([mee, lam])
                    sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0),
                                    [t_k, tf], S, t_eval=np.linspace(t_k, tf, 100))
                    r, _ = mee2rv(*sol.y[:6], mu)
                    inset_ax.scatter(*r[-1], color='0.6', s=5, marker='X', alpha=0.5)
                except:
                    continue

            plot_3sigma_ellipsoid(inset_ax, mu_mc, cov_mc, color='gray', alpha=0.3)
            center = mu_mc
            radius = np.max(np.linalg.norm(mc_endpoints - center, axis=1)) * 1.2
            inset_ax.set_xlim(center[0] - radius, center[0] + radius)
            inset_ax.set_ylim(center[1] - radius, center[1] + radius)
            inset_ax.set_zlim(center[2] - radius, center[2] + radius)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_zticks([])
            inset_ax.set_box_aspect([1, 1, 1])

            ax.set_xlabel("X [km]")
            ax.set_ylabel("Y [km]")
            ax.set_zlabel("Z [km]")
            set_axes_equal(ax)
            ax.legend(handles=[
                    Line2D([0], [0], color='black', lw=2.0, label='Nominal'),
                    Line2D([0], [0], linestyle=':', color='0.6', lw=1.0, label='Monte Carlo'),
                    Line2D([0], [0], marker='o', color='black', linestyle='', label='Start', markersize=5),
                    Line2D([0], [0], marker='X', color='black', linestyle='', label='End', markersize=6),
                    Line2D([0], [0], color='0.5', lw=1.5, label='Control (Start)'),
                    Patch(color='gray', alpha=0.25, label='3σ Ellipsoid (MC)')
                ], loc='upper left', bbox_to_anchor=(0.03, 0.95), frameon=True)
            fname = f"tk{int(t_frac*100)}_unc_{level}_tcn.pdf"
            plt.savefig(os.path.join("uncertainty_aware_outputs", fname), bbox_inches='tight', dpi=600, pad_inches=0.5)
            plt.close()
            print(f"[Saved] {fname}")

            summary.append({
                "t_frac": t_frac,
                "uncertainty": level,
                "volume": volume,
                "n_samples": len(mc_endpoints)
            })

    df = pd.DataFrame(summary)
    df.to_csv("uncertainty_aware_outputs/mc_ellipsoid_volumes_tcn.csv", index=False)
    print("[Saved] mc_ellipsoid_volumes_tcn.csv")
    print(df)

if __name__ == "__main__":
    main()
