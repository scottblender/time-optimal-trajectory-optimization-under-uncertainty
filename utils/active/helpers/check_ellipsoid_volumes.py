import os
import warnings
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.integrate import solve_ivp
from numpy.linalg import eigh

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'helpers')))
from rv2mee import rv2mee
from mee2rv import mee2rv
from odefunc import odefunc

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['LGBM_LOGLEVEL'] = '2'

def numerical_jacobian_rv2mee(rv_nom, mu, eps=1e-6):
    """
    Compute the numerical Jacobian of MEE w.r.t. Cartesian state [r, v]
    Args:
        rv_nom: (7,) array of [r_x, r_y, r_z, v_x, v_y, v_z, mass]
        mu: gravitational parameter
        eps: perturbation step size
    Returns:
        J: (7, 7) Jacobian matrix (MEE+mass) w.r.t. [r, v, m]
    """
    # Unpack the nominal state
    r_nom, v_nom, m_nom = rv_nom[:3], rv_nom[3:6], rv_nom[6]
    mee_nom = rv2mee(r_nom.reshape(1, 3), v_nom.reshape(1, 3), mu).flatten()

    J = np.zeros((7, 7))

    for i in range(6):  # Perturb only r and v
        perturbed = rv_nom.copy()
        perturbed[i] += eps
        r_pert = perturbed[:3].reshape(1, 3)
        v_pert = perturbed[3:6].reshape(1, 3)

        mee_pert = rv2mee(r_pert, v_pert, mu).flatten()
        J[:6, i] = (mee_pert - mee_nom) / eps

    # Mass → mass is identity; rest are zero
    J[6, 6] = 1.0
    return J


def compute_thrust_direction(mu, F, mee, lam):
    p, f, g, h, k, L = mee[:-1]
    lam_p, lam_f, lam_g, lam_h, lam_k, lam_L = lam[:-1]
    lam_matrix = np.array([[lam_p, lam_f, lam_g, lam_h, lam_k, lam_L]]).T
    SinL, CosL = np.sin(L), np.cos(L)
    w = 1 + f * CosL + g * SinL
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

def main():
    model = joblib.load("trained_model.pkl")
    data = joblib.load("stride_4000min/bundle_data_4000min.pkl")
    mu, F_nom, c, m0, g0 = data["mu"], data["F"], data["c"], data["m0"], data["g0"]
    r_nom, v_nom, mass_nom = data["r_tr"], data["v_tr"], data["mass_tr"]
    t_vals = np.asarray(data["backTspan"][::-1])
    tf = t_vals[-1]
    F_val = 0.9 * F_nom

    # Time of replanning
    t_frac = 0.95
    t_k = t_frac * tf
    idx = np.argmin(np.abs(t_vals - t_k))
    r0, v0, m0_val = r_nom[idx], v_nom[idx], mass_nom[idx]
    state_k = np.hstack([r0, v0, m0_val])

    # Initial RV+mass covariance
    P_cart = np.block([
        [np.eye(3)*0.01,       np.zeros((3,3)), np.zeros((3,1))],
        [np.zeros((3,3)), np.eye(3)*0.0001,     np.zeros((3,1))],
        [np.zeros((1,3)), np.zeros((1,3)),      np.array([[0.0001]])]
    ])

    # Monte Carlo sampling
    N = 1000
    samples_rv = np.random.multivariate_normal(state_k, P_cart, size=N)
    mc_endpoints = []

    for s in tqdm(samples_rv, desc="Propagating MC"):
        try:
            J =  numerical_jacobian_rv2mee(s, mu)
            P_mee = J @ P_cart @ J.T
            diag = np.diag(P_mee)

            mee = np.hstack([rv2mee(s[:3].reshape(1, 3), s[3:6].reshape(1, 3), mu).flatten(), s[6]])
            x_input = np.hstack([t_k, mee, diag])
            x_df = pd.DataFrame([x_input], columns=[
                't', 'p', 'f', 'g', 'h', 'L', 'mass',
                'dummy1', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7'
            ])
            lam = model.predict(x_df)[0]
            S = np.hstack([mee, lam])
            sol = solve_ivp(lambda t, x: odefunc(t, x, mu, F_val, c, m0, g0),
                            [t_k, tf], S, t_eval=np.linspace(t_k, tf, 100))
            r, _ = mee2rv(*sol.y[:6], mu)
            mc_endpoints.append(r[-1])
        except:
            continue

    mc_endpoints = np.array(mc_endpoints)
    mu_mc = np.mean(mc_endpoints, axis=0)
    cov_mc = np.einsum("ni,nj->ij", mc_endpoints - mu_mc, mc_endpoints - mu_mc) / mc_endpoints.shape[0]
    eigvals = np.maximum(np.linalg.eigvalsh(cov_mc), 0)
    volume = (4/3) * np.pi * np.prod(3.0 * np.sqrt(eigvals))

    print(f"[t_k = {t_k:.2f} TU] Final 3σ ellipsoid volume = {volume:.3f} DU³")
    print(f"Eigenvalues: {eigvals}")

if __name__ == "__main__":
    main()
