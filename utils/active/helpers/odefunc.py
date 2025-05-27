import numpy as np
import l1_dot_2B_propul
import lm_dot_2B_propul

def odefunc(t, x, mu, F, c, m0, g0):
    """
    Computes the derivatives of the orbital and control states for the given dynamical system.
    This function is used in an ordinary differential equation (ODE) solver to integrate the
    motion of the spacecraft based on its orbital elements and control inputs.

    Parameters:
    - t: Current time (s)
    - x: State vector containing:
        [p, f, g, h, k, L, m, lam_p, lam_f, lam_g, lam_h, lam_k, lam_L, lam_m]
        Where:
        p = semi-latus rectum (km)
        f, g, h, k = equinoctial elements
        L = true longitude (radians)
        m = mass (kg)
        lam_* = Lagrange multipliers for the optimization problem
    - mu: Standard gravitational parameter (km^3/s^2)
    - F: Thrust force (N)
    - c: Specific impulse (s)
    - m0: Initial mass (kg)
    - g0: Gravitational constant (m/s^2)

    Returns:
    - odereturn: Derivative of the state vector [p_dot, f_dot, g_dot, h_dot, k_dot, L_dot, m_dot, lam_dot]
    """
    # Extract the orbital elements and mass from the state vector 'x'
    p = x[0]  # Semi-latus rectum (km)
    f = x[1]  # First equinoctial element
    g = x[2]  # Second equinoctial element
    h = x[3]  # Third equinoctial element
    k = x[4]  # Fourth equinoctial element
    L = x[5]  # True longitude (radians)
    m = x[6]  # Current mass (kg)

    # Extract the Lagrange multipliers (used in optimization)
    lam_p = x[7]  # Lagrange multiplier for p
    lam_f = x[8]  # Lagrange multiplier for f
    lam_g = x[9]  # Lagrange multiplier for g
    lam_h = x[10] # Lagrange multiplier for h
    lam_k = x[11] # Lagrange multiplier for k
    lam_L = x[12] # Lagrange multiplier for L
    lam_m = x[13] # Lagrange multiplier for m

    # Create Lagrange multiplier matrix for use in the calculation of the control law
    lam_matrix = np.array([[lam_p, lam_f, lam_g, lam_h, lam_k, lam_L]]).T

    # Precompute trigonometric values and other useful terms
    SinL = np.sin(L)  # sine of true longitude
    CosL = np.cos(L)  # cosine of true longitude
    w = 1 + f * CosL + g * SinL  # denominator term that normalizes the orbit
    if np.isclose(w, 0, atol=1e-10):  # Prevent divide-by-zero
        print(f"Skipping computation due to zero/near-zero w at time {t}.")
        return np.zeros_like(x)  # Return zeroed derivatives or handle as needed
    s = 1 + h**2 + k**2  # scaling factor for the velocity equation
    alpha = h**2 - k**2  # additional term for orbital element relations

    # Precompute constant terms for position and velocity derivatives
    C1 = np.sqrt(p / mu)  # scale factor for position and velocity
    C2 = 1 / w  # another normalization factor
    C3 = h * SinL - k * CosL  # velocity-related term

    # Define the 'b' vector and the matrix 'A' for calculating the state derivatives
    b = np.array([[0, 0, 0, 0, 0, np.sqrt(mu) * (w**2 / p**1.5)]]).T  # constant terms for the differential equations
    A = np.array([
        [0, 2 * p * C2 * C1, 0],
        [C1 * SinL, C1 * C2 * ((w + 1) * CosL + F), -C1 * (g / w) * C3],
        [-C1 * CosL, C1 * C2 * ((w + 1) * SinL + g), C1 * (f / w) * C3],
        [0, 0, C1 * s * CosL * C2 / 2],
        [0, 0, C1 * s * SinL * C2 / 2],
        [0, 0, C1 * C2 * C3]
    ])  # system matrix for position/velocity derivatives

    # Define thrust force (F_t) and calculate the direction of Lagrange multipliers
    F_t = F  # Thrust force
    mat = np.dot(A.T, lam_matrix)  # matrix multiplication for Lagrange multipliers
    normmat = np.linalg.norm(mat)  # normalize the matrix
    del_t = np.divide(mat, normmat)  # normalized vector

    # Calculate the state derivatives (position, velocity, and mass)
    x_dot = b + np.dot(A, del_t) * (F_t / (m0 * m * g0))  # calculate the state derivatives
    mdot = -F_t / (m0 * c)  # rate of change of mass due to thrust

    # Calculate the Lagrange multiplier rates for the optimization problem
    lam_dot_2b = l1_dot_2B_propul.l1_dot_2B_propul(F, g, h, k, L, p, F_t, g0, lam_f, lam_g, lam_h, lam_k, lam_L, lam_p, m, m0, mu)
    lam_m_dot_2b = lm_dot_2B_propul.lm_dot_2B_propul(F, g, h, k, L, p, F_t, g0, lam_f, lam_g, lam_h, lam_k, lam_L, lam_p, m, m0, mu)

    # Concatenate the Lagrange multiplier derivatives into a single vector
    ldot = np.hstack((lam_dot_2b, lam_m_dot_2b))  # concatenate the Lagrange multiplier derivatives

    # Return the state derivatives along with the mass and Lagrange multipliers' rates
    odereturn = np.append(x_dot, np.array([[mdot]]))  # append mass derivative to the state derivatives
    odereturn = np.append(odereturn, ldot)  # append Lagrange multiplier derivatives to the state

    return odereturn  # return the complete state derivative vector
