import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# Moved to top-level for multiprocessing
def _generate_sigma_points_for_bundle(i, nsd, time_steps, P_combined, r_bundles, v_bundles, mass_bundles, alpha, beta, kappa):
    weights = MerweScaledSigmaPoints(nsd, alpha=alpha, beta=beta, kappa=kappa)
    num_points = len(time_steps)
    sigmas = np.zeros((2 * nsd + 1, nsd, num_points))

    for j in range(num_points):
        nominal_combined = np.concatenate([
            r_bundles[time_steps[j], :, i],
            v_bundles[time_steps[j], :, i],
            [mass_bundles[time_steps[j], i]]
        ])
        sigmas[:, :, j] = weights.sigma_points(nominal_combined, P_combined)

    return sigmas

def generate_sigma_points(nsd=None, alpha=None, beta=None, kappa=None,
                          P_pos=None, P_vel=None, P_mass=None,
                          time_steps=None,  # <-- Updated: pass this in explicitly
                          r_bundles=None, v_bundles=None, mass_bundles=None,
                          num_workers=4):
    if kappa is None:
        kappa = float(3 - nsd)

    P_combined = np.block([
        [P_pos, np.zeros((3, 3)), np.zeros((3, 1))],
        [np.zeros((3, 3)), P_vel, np.zeros((3, 1))],
        [np.zeros((1, 3)), np.zeros((1, 3)), P_mass]
    ])

    if time_steps is None:
        raise ValueError("You must provide a time_steps array.")

    num_bundles = r_bundles.shape[2]

    args_list = [
        (i, nsd, time_steps, P_combined, r_bundles, v_bundles, mass_bundles, alpha, beta, kappa)
        for i in range(num_bundles)
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(_generate_sigma_points_for_bundle, *zip(*args_list)),
            total=num_bundles,
            desc="Parallel sigma point generation"
        ))

    sigmas_combined = np.stack(results, axis=0)
    weights = MerweScaledSigmaPoints(nsd, alpha=alpha, beta=beta, kappa=kappa)
    return sigmas_combined, P_combined, time_steps, len(time_steps), weights.Wm, weights.Wc
