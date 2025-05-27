import numpy as np
def rv2mee(r_eci, v_eci, mu):
    """
    Converts position and velocity vectors (ECI frame) to Equinoctial Orbital Elements.

    Parameters:
    - r_eci: Position vector in ECI (Earth-Centered Inertial) coordinates (3D numpy array in km).
    - v_eci: Velocity vector in ECI coordinates (3D numpy array in km/s).
    - mu: Standard gravitational parameter (km^3/s^2).

    Returns:
    - mee: Equinoctial orbital elements (6-element numpy array), which include:
        [p, f, g, h, k, l] corresponding to:
        - p: Semi-latus rectum
        - f: First equinoctial element
        - g: Second equinoctial element
        - h: Third equinoctial element
        - k: Fourth equinoctial element
        - l: True longitude (in radians)
    """

    # Step 1: Compute the magnitude of the position vector
    radius = np.linalg.norm(r_eci)  # r = ||r_eci||

    # Step 2: Calculate the angular momentum vector (cross product of r and v)
    h_vec = np.cross(r_eci, v_eci)  # h = r x v
    hmag = np.linalg.norm(h_vec)  # hmag = ||h_vec|| (angular momentum magnitude)

    # Step 3: Calculate the semi-latus rectum (p) from angular momentum and gravitational parameter
    p = np.divide(np.square(hmag), mu)  # p = h^2 / mu

    # Step 4: Compute the unit vector of angular momentum (h_hat)
    h_hat = np.divide(h_vec, hmag)  # h_hat = h / ||h||

    # Step 5: Calculate the Equinoctial Elements K and H
    denom = 1 + h_hat[0, 2]  # denominator used to compute k and h
    h = -h_hat[0, 1] / denom  # h element
    k = h_hat[0, 0] / denom   # k element

    # Step 6: Construct the unit vectors (f_hat, g_hat) in the Equinoctial frame
    f_hat = np.zeros(3)
    g_hat = np.zeros(3)

    f_hat[0] = 1 - np.square(k) + np.square(h)  # f_hat[0] = 1 - k^2 + h^2
    f_hat[1] = 2 * np.multiply(k, h)  # f_hat[1] = 2kh
    f_hat[2] = -2 * k  # f_hat[2] = -2k

    g_hat[0] = f_hat[1]  # g_hat[0] = f_hat[1]
    g_hat[1] = 1 + np.square(k) - np.square(h)  # g_hat[1] = 1 + k^2 - h^2
    g_hat[2] = 2 * h  # g_hat[2] = 2h

    # Step 7: Compute some additional parameters for further calculations
    rdotv = np.dot(r_eci, v_eci.T)  # dot product of r_eci and v_eci (r.v)
    cross_vh = np.cross(v_eci, h_vec)  # cross product of v_eci and h_vec (v x h)
    ssqrd = 1 + np.square(k) + np.square(h)  # denominator for normalization

    # Step 8: Normalize the unit vectors (f_hat and g_hat)
    f_hat = np.divide(f_hat, ssqrd)  # normalize f_hat
    g_hat = np.divide(g_hat, ssqrd)  # normalize g_hat

    # Step 9: Compute the eccentricity vector (e_vec)
    e_vec = np.divide(-r_eci, radius) + np.divide(cross_vh, mu)  # eccentricity vector (eqn A-7 from Cefola & Broucke)

    # Step 10: Compute the unit vectors uhat and vhat
    uhat = np.divide(r_eci, radius)  # uhat = r / radius
    vhat = np.divide((np.multiply(v_eci, radius) - np.multiply(np.divide(rdotv, radius), r_eci)), hmag)  # vhat calculation

    # Step 11: Compute the f and g equinoctial elements (dot products with e_vec)
    g = np.dot(e_vec, g_hat)  # g = e_vec . g_hat
    f = np.dot(e_vec, f_hat)  # f = e_vec . f_hat

    # Step 12: Compute the true longitude (l_nonmod) using geometry
    cosl = uhat[0, 0] + vhat[0, 1]  # cosine of longitude
    sinl = uhat[0, 1] - vhat[0, 0]  # sine of longitude
    l_nonmod = np.arctan2(sinl, cosl)  # true longitude (in radians)

    # Step 13: Assemble the final array of Equinoctial Elements
    mee = np.zeros(6)
    mee[0] = p  # Semi-latus rectum
    mee[1] = f  # First Equinoctial element
    mee[2] = g  # Second Equinoctial element
    mee[3] = h  # Third Equinoctial element
    mee[4] = k  # Fourth Equinoctial element
    mee[5] = l_nonmod  # True longitude (in radians)

    # Return the Equinoctial Elements as a numpy array
    return np.array(mee)
