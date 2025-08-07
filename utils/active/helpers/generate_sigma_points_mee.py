import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from rv2mee import rv2mee
from mee2rv import mee2rv

def _generate_sigma_points_for_bundle_mee(i, nsd, time_steps, P_combined,
                                          r_bundles, v_bundles, mass_bundles, mu,
                                          alpha, beta, kappa):
    weights = MerweScaledSigmaPoints(nsd, alpha=alpha, beta=beta, kappa=kappa)
    num_points = len(time_steps)
    sigmas = np.zeros((2 * nsd + 1, nsd, num_points))

    for j in range(num_points):
        r = r_bundles[time_steps[j], :, i]
        v = v_bundles[time_steps[j], :, i]
        m = mass_bundles[time_steps[j], i]

        mee = rv2mee(np.array([r]), np.array([v]), mu).flatten()
        state_mee = np.concatenate([mee, [m]])

        sigmas[:, :, j] = weights.sigma_points(state_mee, P_combined)

        if j == 0:
            mean_sp = np.sum(weights.Wm[:, None] * sigmas[:, :, j], axis=0)
            diffs = sigmas[:, :, j] - mean_sp
            cov_sp = np.einsum("i,ij,ik->jk", weights.Wc, diffs, diffs)
            print(f"\n[SP MEE CHECK] Bundle {i}, t_idx=0")
            print("→ Mean diff norm:", np.linalg.norm(mean_sp - state_mee))
            print("→ Cov diff max rel error:", np.max(np.abs(P_combined - cov_sp) / np.maximum(P_combined, 1e-15)))

    return sigmas

def generate_sigma_points_mee(nsd=None, alpha=None, beta=None, kappa=None,
                               P_mee=None, P_mass=None, time_steps=None,
                               r_bundles=None, v_bundles=None, mass_bundles=None,
                               mu=None, num_workers=4):

    if kappa is None:
        kappa = float(3 - nsd)

    P_combined = np.block([
        [P_mee, np.zeros((6, 1))],
        [np.zeros((1, 6)), P_mass]
    ])

    if time_steps is None:
        raise ValueError("You must provide a time_steps array.")

    num_bundles = r_bundles.shape[2]
    args_list = [
        (i, nsd, time_steps, P_combined, r_bundles, v_bundles, mass_bundles, mu, alpha, beta, kappa)
        for i in range(num_bundles)
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(_generate_sigma_points_for_bundle_mee, *zip(*args_list)),
            total=num_bundles,
            desc="MEE sigma point generation"
        ))

    sigmas_combined_mee = np.stack(results, axis=0)

    # Convert MEE → RV for downstream propagation
    sigmas_combined_rv = np.zeros_like(sigmas_combined_mee)  # shape: (bundles, 2n+1, 7, num_times)

    for b in range(sigmas_combined_mee.shape[0]):
        for j in range(sigmas_combined_mee.shape[3]):
            for i in range(sigmas_combined_mee.shape[1]):
                mee = sigmas_combined_mee[b, i, :6, j]
                m = sigmas_combined_mee[b, i, 6, j]
                r, v = mee2rv(mee[0], mee[1], mee[2], mee[3], mee[4], mee[5], mu)
                sigmas_combined_rv[b, i, :, j] = np.concatenate([r, v, [m]])

    weights = MerweScaledSigmaPoints(nsd, alpha=alpha, beta=beta, kappa=kappa)
    return sigmas_combined_rv, P_combined, time_steps, len(time_steps), weights.Wm, weights.Wc
