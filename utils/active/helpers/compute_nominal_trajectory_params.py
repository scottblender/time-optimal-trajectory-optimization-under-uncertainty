import numpy as np
import scipy.optimize
import time
import wrapTo2Pi
import rv2mee
import calc_residuals_nominal

def compute_nominal_trajectory_params():
    """
    This function calculates the optimal parameters for a nominal trajectory based on initial conditions
    and system parameters. It uses numerical optimization to solve for the residuals in the trajectory
    equations, and it returns the solution, the calculated final time of flight, and several important
    parameters.

    Returns:
    - p_sol: The solution to the optimization problem (set of trajectory parameters).
    - tfound: The calculated final time of flight for the trajectory.
    - s0: Initial state in non-dimensional units.
    - mu: Gravitational parameter for the system.
    - F: Non-dimensional thrust force.
    - c: Specific impulse in non-dimensional units.
    - m0: Initial spacecraft mass.
    - g0: Gravitational constant.
    - R_V_0: Target spacecraft initial position.
    - V_V_0: Target spacecraft initial velocity.
    - DU: Distance unit (Sun's radius).
    """

    # Define Parameters for Nominal Trajectory (Initial Conditions)
    tf_g_LB = 10000 # lower bound of final time
    tf_g_UB = 25000 # upper bound of final time

    # Earth Initial Conditions (position and velocity in m and m/s)
    R_E_0 = 1e3 * 1e8 * np.array([-0.036378710816509, 1.470997987841786, -0.000022614410420])  # m
    V_E_0 = 1e3 * np.array([-30.265097988218205, -0.848685467901138, 0.000050530360628])  # m/s
    
    # Dionysus Initial Conditions (position and velocity in km and km/s)
    R_V_0 = 1e3 * 1e8 * np.array([-3.024520148842469, 3.160971796320275,  0.828722900755180])  # m
    V_V_0 = 1e3 * np.array([-4.533473799840294, -13.110309800847542, 0.656163826017450]) # m/s
    
    # Constants for the system
    mu_s = 132712 * 10**6 * 1e9  # m^3/s^2 (Gravitational Parameter for the Sun)
    g0_s = 9.81  # m/s^2 (Standard Gravity)
    
    # Spacecraft Parameters (Mass and ISP)
    m0 = 4000  # kg (Initial Mass of the spacecraft)
    Isp = 3800  # s (Specific Impulse)
    F0 = 0.33  # N (Thrust Force)
    
    # Non-dimensional Units for Sun
    DU = 696340e3  # m, Sun's Radius
    TU = np.sqrt(DU / g0_s)  # Time Unit (for non-dimensionalization)
    MU = m0  # Mass Unit
    
    # Unit Conversions for the system
    mu = (mu_s * TU**2) / (DU**3)  # Gravitational Parameter (non-dimensionalized)
    VU = DU / TU  # Velocity Unit (non-dimensionalized)
    AU = DU / (TU**2)  # Acceleration Unit
    FU = MU * AU  # Force Unit
    
    # Non-dimensional thrust and gravity
    F = F0
    g0 = g0_s
    IsDU = np.sqrt(DU / (g0**3))  # Scale for thrust to gravitational units
    c = (Isp / IsDU)  # Specific Impulse in non-dimensional units

    # Converting initial conditions to non-dimensional units
    s_e = np.array([[R_E_0, V_E_0]])
    s_d = np.array([[R_V_0, V_V_0]])
    
    # Convert initial position and velocity to Equinoctial Elements (MEE)
    s0 = rv2mee.rv2mee(R_E_0, V_E_0, mu_s)
    s0 = np.hstack([s0, 1.0])  # Add a dummy value for further processing
    s0[0] = np.divide(s0[0], DU)  # Non-dimensionalize position
    
    s_t = rv2mee.rv2mee(R_V_0, V_V_0, mu_s)
    s_t[0] = np.divide(s_t[0], DU)  # Non-dimensionalize position
    
    # Modify target state to be a weighted mix of initial and target states
    eps = 1
    s_t_t = np.multiply(s_t, eps) + np.multiply(s0[0:6], (1 - eps))
    s_t = s_t_t

    # Norm of the residual (used to check convergence)
    normfval = 1
    
    # Start timing the optimization process
    start = time.time()

    # Begin optimization loop: continue until residuals are sufficiently small (converged)
    while normfval >= 1e-4:
        # Generate new random initial guesses for the trajectory parameters
        lam_0 = np.multiply(np.random.random(7), -2) + np.random.random(7)  # Random guesses for parameters
        th = (-np.pi / 2) + np.multiply(np.random.random(1), np.pi / 2)  # Random angle for the optimization
        y = np.append(lam_0, th, axis=0)  # Combine guesses into a single array
        
        # Set the initial guess (for debugging purposes, can use specific values here)
        y = [0.00777453, 0.85823374, -0.28062779, 0.01970831, 0.86882985, -0.11250828, -0.88233416, -3.33541801]

        # Prepare the input data for optimization
        data = (s0, s_t, mu, F, c, m0, g0)
        
        # Use `fsolve` to solve the residuals and find the trajectory parameters (lam_0)
        p_sol = scipy.optimize.fsolve(calc_residuals_nominal.calc_residuals_nominal, y, args=data, xtol=1e-12, maxfev=10000)
        
        # Compute the residuals for the current solution
        fval = calc_residuals_nominal.calc_residuals_nominal(p_sol, s0, s_t, mu, F, c, m0, g0)
        
        # Calculate the norm (magnitude) of the residuals to check convergence
        normfval = np.linalg.norm(fval)
    
    # End the timing of the optimization process
    end = time.time()

    # Output the time taken for the optimization process (in minutes)
    print(round((end - start) / 60, 2))
    
    # Calculate the final time of flight (tfound) using the optimized solution
    tfound = 0.5 * ((tf_g_UB + tf_g_LB) + (tf_g_UB - tf_g_LB) * np.sin(wrapTo2Pi.wrapTo2Pi(p_sol[-1])))

    # Return the optimized solution (p_sol), final time of flight (tfound), and the requested parameters
    return p_sol, tfound, s0, mu, F, c, m0, g0, R_V_0, V_V_0, DU,TU
