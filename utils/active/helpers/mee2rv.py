import numpy as np

def mee2rv(p, f, g, h, k, L, mu):
    """
    Converts Modified Equinoctial Elements (p, f, g, h, k, L) to ECI position and velocity.

    Supports both single-sample (1D) and batched (N,) or (N,1) input formats.
    """
    # Promote to array and ensure 1D arrays are treated correctly
    p = np.atleast_1d(p)
    f = np.atleast_1d(f)
    g = np.atleast_1d(g)
    h = np.atleast_1d(h)
    k = np.atleast_1d(k)
    L = np.atleast_1d(L)

    single_input = p.shape == () or p.shape[0] == 1

    # Compute radius
    cosL = np.cos(L)
    sinL = np.sin(L)
    radius = p / (1 + f * cosL + g * sinL)

    alpha2 = h**2 - k**2
    tani2s = h**2 + k**2
    s2 = 1 + tani2s
    sqrt_mu_p = np.sqrt(mu / p)

    # Position
    r_eci = np.zeros((len(L), 3))
    r_eci[:, 0] = radius * ((cosL + alpha2 * cosL + 2 * h * k * sinL) / s2)
    r_eci[:, 1] = radius * ((sinL - alpha2 * sinL + 2 * h * k * cosL) / s2)
    r_eci[:, 2] = 2 * radius * ((h * sinL - k * cosL) / s2)

    # Velocity
    v_eci = np.zeros_like(r_eci)
    v_eci[:, 0] = -sqrt_mu_p * ((sinL + alpha2 * sinL - 2 * h * k * cosL + g - 2 * f * h * k + alpha2 * g) / s2)
    v_eci[:, 1] = -sqrt_mu_p * ((-cosL + alpha2 * cosL + 2 * h * k * sinL - f + 2 * g * h * k + alpha2 * f) / s2)
    v_eci[:, 2] = 2 * sqrt_mu_p * ((h * cosL + k * sinL + f * h + g * k) / s2)

    return r_eci.squeeze() if single_input else r_eci, v_eci.squeeze() if single_input else v_eci
