import numpy as np

def rv2mee(r_eci, v_eci, mu):
    """
    Converts position and velocity vectors (ECI frame) to Modified Equinoctial Elements (MEE).

    Parameters:
    - r_eci: ndarray of shape (3,) or (N, 3)
    - v_eci: ndarray of shape (3,) or (N, 3)
    - mu: gravitational parameter

    Returns:
    - mee: ndarray of shape (6,) or (N, 6)
    """
    # Promote to 2D if single input
    single_input = False
    if r_eci.ndim == 1:
        r_eci = r_eci[np.newaxis, :]
        v_eci = v_eci[np.newaxis, :]
        single_input = True

    r = np.linalg.norm(r_eci, axis=1)
    h_vec = np.cross(r_eci, v_eci)
    hmag = np.linalg.norm(h_vec, axis=1)

    p = hmag**2 / mu

    h_hat = h_vec / hmag[:, None]
    denom = 1 + h_hat[:, 2]
    h_elem = -h_hat[:, 1] / denom
    k_elem = h_hat[:, 0] / denom

    # Equinoctial frame unit vectors
    f_hat = np.zeros_like(r_eci)
    g_hat = np.zeros_like(r_eci)

    h2 = h_elem**2
    k2 = k_elem**2

    f_hat[:, 0] = 1 - k2 + h2
    f_hat[:, 1] = 2 * k_elem * h_elem
    f_hat[:, 2] = -2 * k_elem

    g_hat[:, 0] = f_hat[:, 1]
    g_hat[:, 1] = 1 + k2 - h2
    g_hat[:, 2] = 2 * h_elem

    ssqrd = 1 + k2 + h2
    f_hat /= ssqrd[:, None]
    g_hat /= ssqrd[:, None]

    rdotv = np.sum(r_eci * v_eci, axis=1)
    cross_vh = np.cross(v_eci, h_vec)
    e_vec = -r_eci / r[:, None] + cross_vh / mu

    g_elem = np.sum(e_vec * g_hat, axis=1)
    f_elem = np.sum(e_vec * f_hat, axis=1)

    uhat = r_eci / r[:, None]
    vhat = (v_eci * r[:, None] - (rdotv / r)[:, None] * r_eci) / hmag[:, None]

    cosl = uhat[:, 0] + vhat[:, 1]
    sinl = uhat[:, 1] - vhat[:, 0]
    L = np.arctan2(sinl, cosl)

    mee = np.column_stack([p, f_elem, g_elem, h_elem, k_elem, L])

    return mee.squeeze() if single_input else mee
