import numpy as np
import scipy.optimize
import time
import scipy.integrate
from threading import Thread
import twobody
import mee2rv
import odefunc
import calc_residuals_bundle
import run_solve_ivp

def compute_bundle_trajectory_params(p_sol, s0, tfound, mu, F, c, m0, g0, R_V_0, V_V_0, DU, num_bundles):
    """
    Computes the spacecraft trajectory and generates perturbed trajectory bundles for Monte Carlo sampling.

    Parameters:
    - p_sol: Solution from the optimization process (e.g., initial trajectory solution), numpy array of shape (7,)
    - s0: Initial state of the spacecraft, numpy array of shape (7,)
    - tfound: Time of flight from the previous calculations, float (s)
    - mu: Gravitational parameter (non-dimensionalized), float
    - F: Thrust force, float (N)
    - c: Specific impulse in non-dimensional units, float (s)
    - m0: Initial mass of the spacecraft, float (kg)
    - g0: Gravitational constant in non-dimensional units, float (m/s^2)
    - R_V_0: Initial position of the target body, numpy array of shape (3,)
    - V_V_0: Initial velocity of the target body, numpy array of shape (3,)
    - DU: Unit of distance for non-dimensionalization, float
    - num_bundles: Number of perturbed trajectories to generate, integer

    Returns:
    - r_tr: The trajectory positions, numpy array of shape (num_steps, 3)
    - v_tr: The trajectory velocities, numpy array of shape (num_steps, 3)
    - S_bundles: The bundle of state vectors for all perturbed trajectories, numpy array of shape (num_steps, 14, num_bundles)
    - r_bundles: The bundle of position vectors for all perturbed trajectories, numpy array of shape (num_steps, 3, num_bundles)
    - v_bundles: The bundle of velocity vectors for all perturbed trajectories, numpy array of shape (num_steps, 3, num_bundles)
    - new_lam_bundles: 2D array for storing perturbed `new_lam` values, numpy array of shape (7, num_bundles)
    - backTspan: Time span for backward integration, numpy array of shape (num_steps,)
    """

    # Create time span for trajectory integration (1000 steps)
    tspan = np.linspace(0, tfound, 1000)

    # Extract the initial trajectory solution parameters (lam_sol)
    lam_sol = p_sol[0:7]
    
    # Combine initial state and lam_sol parameters to define the full initial state for the spacecraft
    S = np.append(s0, lam_sol)

    # Define the ODE function to model spacecraft dynamics
    func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)

    # Integrate spacecraft trajectory using Runge-Kutta 45
    Sf = scipy.integrate.solve_ivp(func, [0, tfound], S, method='RK45', rtol=1e-3, atol=1e-6, t_eval=tspan)
    
    # Extract position and velocity from the integrated state vector (state)
    state = Sf.y[0:7, :]
    r_tr, v_tr = mee2rv.mee2rv(state[0, :], state[1, :], state[2, :], state[3, :], state[4, :], state[5, :], mu)

    # Initialize the target body's initial position and velocity
    y0V = np.append(R_V_0, V_V_0)
    mu_sun = 1.32712440018e20  # Gravitational parameter of the Sun (m^3/s^2)
    r = 149597870700  # 1 AU in meters (distance from Earth to Sun)

    # Define time span for the target body's motion (3.5 years in seconds)
    tspan_target = [0, 3.5 * 365.25 * 86400]
    func_target = lambda t, y: twobody.twobody(t, y, mu_sun)

    # Integrate the target body's motion over the defined time span
    yV = scipy.integrate.solve_ivp(func_target, tspan_target, y0V, method='RK45', rtol=1e-3, atol=1e-6)
    yV = yV.y / DU  # Non-dimensionalize the result

    # Extract final position and trajectory parameters from the integrated spacecraft state
    final_pos = Sf.y[0:7, -1]
    final_lam = Sf.y[7:14, -1]

    # Initialize the trajectory bundle arrays for Monte Carlo sampling
    num_bundles = num_bundles  # Number of perturbed trajectories
    num_steps = 1000  # Number of time steps in each trajectory
    S_bundles = np.zeros((num_steps, 14, num_bundles))  # 3D array for state vectors of perturbed trajectories
    r_bundles = np.zeros((num_steps, 3, num_bundles))  # 3D array for position bundles
    v_bundles = np.zeros((num_steps, 3, num_bundles))  # 3D array for velocity bundles
    new_lam_bundles = np.zeros((7, num_bundles))  # 2D array for new lam bundles (7, num_bundles)

    # Time span for backward integration
    backTspan = np.linspace(tfound, 0, num_steps)

    # Perturbation parameters for Monte Carlo sampling
    alpha = 1  # Magnitude of perturbation scaling
    max_angle = 1e-5  # Maximum angle for perturbations

    # Generate perturbed trajectories via Monte Carlo sampling
    for index in range(S_bundles.shape[2]):
        normfval2 = 1
        
        # Perturb the trajectory parameters until the residual is sufficiently small
        while normfval2 >= 1e-4:
            # Adjust perturbations for mass and trajectory parameters based on orders of magnitude
            order_mag = np.floor(np.log10(np.abs(final_lam[:5])))
            order_mag = 10 ** order_mag
            delta_values = alpha * order_mag * np.random.rand()
            delta_mass = 0.4 * np.random.rand() - 0.2

            # Optimize residuals for the perturbed trajectory parameters
            data = (final_lam, max_angle)
            p_sol2 = scipy.optimize.fsolve(calc_residuals_bundle.calc_residuals_bundle, delta_values, args=data, xtol=1e-10, maxfev=10000)
            normfval2 = np.linalg.norm(calc_residuals_bundle.calc_residuals_bundle(p_sol2, final_lam, max_angle))

        # Apply the perturbations to the lam parameters and mass
        p_sol2 = np.array(p_sol2)
        new_lam = np.zeros(7)
        new_lam[0:5] = p_sol2[0:5] + final_lam[0:5]
        new_lam[5:] = final_lam[5:]
        new_pos = final_pos.copy()
        new_pos[-1] = (1 + delta_mass) * new_pos[-1]
        y_pert = np.append(new_pos, new_lam)

        # Define the ODE function for spacecraft with perturbations
        func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)

        # Run the integration for the perturbed trajectory
        result_container = {}
        thread = Thread(target=run_solve_ivp.run_solve_ivp, args=(func, [tfound, 0], y_pert, 'RK45', backTspan, 1e-3, 1e-8, result_container))
        start_time = time.time()
        thread.start()

        # Wait for the thread to finish or timeout after 10 seconds
        thread.join(timeout=10)
        if thread.is_alive():
            print("Timeout exceeded for bundle " + str(index) + ". Redetermining p_sol2.")
            thread.join()  # Clean up the thread
            continue  # Return to the while loop to redetermine p_sol2

        # If the thread finishes in time, extract the result
        Sf = result_container['value']
        bundle_Sf = Sf.y

        # Convert the state vectors to position and velocity
        r_bundle, v_bundle = mee2rv.mee2rv(bundle_Sf[0, :], bundle_Sf[1, :], bundle_Sf[2, :], bundle_Sf[3, :], bundle_Sf[4, :], bundle_Sf[5, :], mu)

        # Store the results in the bundle arrays
        x_bundle = r_bundle[:, 0]
        y_bundle = r_bundle[:, 1]
        z_bundle = r_bundle[:, 2]

        # Store the bundled state vectors, position, and velocity
        bundle_array_size = bundle_Sf.shape
        S_bundles[0:bundle_array_size[1], 0:bundle_array_size[0], index] += bundle_Sf.T
        bundle_array_size = r_bundle.shape
        r_bundles[0:bundle_array_size[0], 0:bundle_array_size[1], index] += r_bundle
        v_bundles[0:bundle_array_size[0], 0:bundle_array_size[1], index] += v_bundle

        # Store the new lambda parameters for the perturbed trajectory
        new_lam_bundles[:, index] += new_lam

    # Return the trajectory results and bundles for analysis
    return r_tr, v_tr, S_bundles, r_bundles, v_bundles, new_lam_bundles, backTspan
