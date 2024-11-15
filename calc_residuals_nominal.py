import numpy as np
import scipy.integrate

def calc_residuals_nominal(y, s0, s_t, mu, F, c, m0, g0):
    """
    Calculates the residuals for the nominal trajectory optimization problem.
    This function computes the difference between the state at the final time and the target
    state, and also computes the Lagrange multipliers' residuals to assess the optimization's
    convergence. It integrates the system of equations defined by the orbital dynamics.

    Parameters:
    - y: The vector of optimization variables, including the Lagrange multipliers (`lam_0`)
      and the optimization parameter `th`.
    - s0: Initial state vector (orbital elements and initial conditions).
    - s_t: Target state vector (orbital elements at the target time).
    - mu: Gravitational parameter (km^3/s^2).
    - F: Thrust force (N).
    - c: Specific impulse (s).
    - m0: Initial mass (kg).
    - g0: Gravitational constant (m/s^2).

    Returns:
    - res: A numpy array containing the residuals of the optimization:
      [p_residual, f_residual, g_residual, h_residual, k_residual, lam_l_residual, lam_m_residual, H_residual]
    """

    # Step 1: Extract optimization variables from input vector `y`
    lam_0 = y[0:7]  # Lagrange multipliers for the optimization variables (p, f, g, h, k, L)
    th = y[7]       # Optimization parameter (angle) used for time scaling

    # Step 2: Calculate the final time `tf` using the given angle `th` and target time bounds
    tf = 0.5 * ((tf_g_UB + tf_g_LB) + (tf_g_UB - tf_g_LB) * np.sin(wrapTo2Pi(th)))
    tspan = [0, tf]  # Time span for the integration, from 0 to tf

    # Step 3: Define the system of differential equations using the odefunc function
    func = lambda t, x: odefunc(t, x, mu, F, c, m0, g0)

    # Step 4: Initialize the state vector by appending the initial state `s0` and Lagrange multipliers `lam_0`
    initial_conditions = np.append(s0, lam_0)

    # Step 5: Integrate the system using `scipy.integrate.solve_ivp` to solve the ODEs
    Sf = scipy.integrate.solve_ivp(func, tspan, initial_conditions, method='RK45', rtol=1e-3, atol=1e-6)

    # Step 6: Extract the final values of the state variables and Lagrange multipliers from the solution
    m = Sf.y[6, -1]  # Final mass
    lam_values = Sf.y[7:13, -1]  # Final values of Lagrange multipliers
    lam_m = Sf.y[13, -1]  # Final value of the Lagrange multiplier for mass
    p = Sf.y[0, -1]  # Final value of semi-latus rectum
    f = Sf.y[1, -1]  # Final value of equinoctial element f
    g = Sf.y[2, -1]  # Final value of equinoctial element g
    h = Sf.y[3, -1]  # Final value of equinoctial element h
    k = Sf.y[4, -1]  # Final value of equinoctial element k
    L = Sf.y[5, -1]  # Final value of true longitude

    # Step 7: Extract the individual Lagrange multipliers
    lam_p = lam_values[0]
    lam_f = lam_values[1]
    lam_g = lam_values[2]
    lam_h = lam_values[3]
    lam_k = lam_values[4]
    lam_l = lam_values[5]

    # Step 8: Create a matrix of the Lagrange multipliers for further calculations
    lam_matrix = np.vstack((lam_p, lam_f, lam_g, lam_h, lam_k, lam_l))

    # Step 9: Calculate the residuals for the Lagrange multipliers
    # H is the norm of the Lagrange multiplier matrix minus 1 (a measure of the error in the multipliers)
    H = np.linalg.norm(lam_matrix) - 1

    # Step 10: Compute the residuals for each of the orbital elements and Lagrange multipliers
    # These residuals represent the differences between the calculated state and the target state
    res = [
        p - s_t[0],  # Residual for semi-latus rectum
        f - s_t[1],  # Residual for f element
        g - s_t[2],  # Residual for g element
        h - s_t[3],  # Residual for h element
        k - s_t[4],  # Residual for k element
        lam_l,        # Residual for Lagrange multiplier associated with longitude
        lam_m,        # Residual for Lagrange multiplier associated with mass
        H             # Residual for the Lagrange multiplier norm (should be close to 0)
    ]

    # Step 11: Return the residuals as a numpy array
    return np.array(res)
