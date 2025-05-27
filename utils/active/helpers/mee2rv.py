import numpy as np

def mee2rv(p, f, g, h, k, L, mu):
    """
    Converts the Equinoctial Orbital Elements (p, f, g, h, k, L) to position and velocity vectors
    in the Earth-Centered Inertial (ECI) frame.

    Parameters:
    - p: Semi-latus rectum (km)
    - f: First equinoctial element
    - g: Second equinoctial element
    - h: Third equinoctial element
    - k: Fourth equinoctial element
    - L: True longitude (radians)
    - mu: Standard gravitational parameter (km^3/s^2)

    Returns:
    - r_eci: Position vectors in ECI coordinates (Nx3 numpy array)
    - v_eci: Velocity vectors in ECI coordinates (Nx3 numpy array)
    """

    # Step 1: Compute the radius (distance from the central body) using the semi-latus rectum (p)
    # and the equinoctial elements f, g, and the true longitude L
    radius = np.divide(p, (1 + f * np.cos(L) + g * np.sin(L)))  # radius = p / (1 + f*cos(L) + g*sin(L))

    # Step 2: Precompute common terms that will be used in position and velocity equations
    alpha2 = h**2 - k**2  # alpha2 = h^2 - k^2 (used in position and velocity equations)
    tani2s = h**2 + k**2  # tani2s = h^2 + k^2 (used for normalization)
    s2 = 1 + tani2s  # s2 = 1 + (h^2 + k^2) (used for scaling position and velocity components)

    # Step 3: Initialize position and velocity arrays in the ECI frame
    # These arrays will store the position and velocity vectors for each time step (for each L)
    r_eci = np.zeros((len(L), 3))  # Position vectors (Nx3 array)
    v_eci = np.zeros((len(L), 3))  # Velocity vectors (Nx3 array)

    # Step 4: Compute the position vector in the ECI frame using the equinoctial elements
    # Each component of r_eci is computed based on the true longitude (L) and the equinoctial elements
    r_eci[:, 0] = radius * np.divide((np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L)), s2)
    r_eci[:, 1] = radius * np.divide((np.sin(L) - alpha2 * np.sin(L) + 2 * h * k * np.cos(L)), s2)
    r_eci[:, 2] = 2 * radius * np.divide((h * np.sin(L) - k * np.cos(L)), s2)

    # Step 5: Compute the velocity vector in the ECI frame using the equinoctial elements
    # Each component of v_eci is calculated based on the semi-latus rectum (p), gravitational parameter (mu),
    # and the equinoctial elements (f, g, h, k)
    v_eci[:, 0] = -np.sqrt(mu / p) * np.divide((np.sin(L) + alpha2 * np.sin(L) - 2 * h * k * np.cos(L) + g - 2 * f * h * k + alpha2 * g), s2)
    v_eci[:, 1] = -np.sqrt(mu / p) * np.divide((-np.cos(L) + alpha2 * np.cos(L) + 2 * h * k * np.sin(L) - f + 2 * g * h * k + alpha2 * f), s2)
    v_eci[:, 2] = 2 * np.sqrt(mu / p) * np.divide((h * np.cos(L) + k * np.sin(L) + f * h + g * k), s2)

    # Step 6: Return the computed position (r_eci) and velocity (v_eci) vectors in the ECI frame
    return r_eci, v_eci
