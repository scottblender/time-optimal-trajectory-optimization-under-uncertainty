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
    This function computes the spacecraft trajectory, bundles the resulting state vectors, and 
    performs the necessary optimization to create perturbed trajectories for Monte Carlo sampling.
    
    Parameters:
    - p_sol: The solution from the optimization process (e.g., initial trajectory solution).
    - s0: The initial state of the spacecraft.
    - tfound: The time of flight from the previous calculations.
    - mu: Gravitational parameter (non-dimensionalized).
    - F: Thrust force.
    - c: Specific impulse in non-dimensional units.
    - m0: Initial mass of the spacecraft.
    - g0: Gravitational constant in non-dimensional units.
    - R_V_0: Initial position of the target body.
    - V_V_0: Initial velocity of the target body.
    - DU: Unit of distance (non-dimensional).

    Returns:
    - r_tr: The trajectory positions (3D array).
    - v_tr: The trajectory velocities (3D array).
    - s_bundle: The bundle of state vectors for all perturbed trajectories.
    - r_bundle: The bundle of position vectors for all perturbed trajectories.
    - v_bundle: The bundle of velocity vectors for all perturbed trajectories.
    """

    # Set time span for trajectory integration
    tspan = np.linspace(0, tfound, 1000)

    # Get the solution of the nominal trajectory parameters (lam_sol)
    lam_sol = p_sol[0:7]
    
    # Combine initial state and trajectory parameters for the spacecraft
    S = np.append(s0, lam_sol)
    
    # Define the ODE function for integration (spacecraft dynamics)
    func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)

    # Integrate spacecraft trajectory using Runge-Kutta 45
    Sf = scipy.integrate.solve_ivp(func, [0, tfound], S, method='RK45', rtol=1e-3, atol=1e-6, t_eval=tspan)
    
    # Extract state (position and velocity)
    state = Sf.y[0:7, :]
    r_tr, v_tr = mee2rv.mee2rv(state[0, :], state[1, :], state[2, :], state[3, :], state[4, :], state[5, :], mu)

    # Set up the initial conditions for the target body (e.g., Earth or another body)
    y0V = np.append(R_V_0, V_V_0)
    mu_sun = 1.32712440018e20  # m^3/s^2, gravitational parameter of the Sun
    r = 149597870700  # m, distance of 1 AU

    # Define the time span for the target body's motion (e.g., 3.5 years)
    tspan_target = [0, 3.5 * 365.25 * 86400]  # 3.5 years in seconds
    func_target = lambda t, y: twobody.twobody(t, y, mu_sun)

    # Integrate the target body's motion
    yV = scipy.integrate.solve_ivp(func_target, tspan_target, y0V, method='RK45', rtol=1e-3, atol=1e-6)
    yV = yV.y / DU  # Non-dimensionalize the result

    # Get the final position and lam parameters
    final_pos = Sf.y[0:7, -1]
    final_lam = Sf.y[7:14, -1]

    # Set up bundles for Monte Carlo sampling of trajectories
    num_bundles = num_bundles  # Number of perturbed trajectories
    num_steps = 1000  # Number of time steps in each trajectory
    S_bundles = np.zeros((num_steps, 14, num_bundles))  # 3D array to hold all trajectories
    r_bundles = np.zeros((num_steps, 3, num_bundles))  # 3D array for position bundles
    v_bundles = np.zeros((num_steps, 3, num_bundles))  # 3D array for velocity bundles

    backTspan = np.linspace(tfound, 0, num_steps)  # Time span for backward integration

    # Perturbation parameters for Monte Carlo sampling
    alpha = 1
    max_angle = 1e-3

    # Generate perturbed trajectories
    for index in range(S_bundles.shape[2]):
        normfval2 = 1
        while normfval2 >= 1e-4:
            # Perturb the trajectory by adjusting the lam parameters
            order_mag = np.floor(np.log10(np.abs(final_lam[:5])))
            order_mag = 10 ** order_mag
            delta_values = alpha * order_mag * np.random.rand()
            delta_mass = 0.4 * np.random.rand() - 0.2

            # Optimize the residuals
            data = (final_lam, max_angle)
            p_sol2 = scipy.optimize.fsolve(calc_residuals_bundle.calc_residuals_bundle, delta_values, args=data, xtol=1e-10, maxfev=10000)
            normfval2 = np.linalg.norm(calc_residuals_bundle.calc_residuals_bundle(p_sol2, final_lam, max_angle))

        # Apply the perturbations to the lam parameters and initial state
        p_sol2 = np.array(p_sol2)
        new_lam = np.zeros(7)
        new_lam[0:5] = p_sol2[0:5] + final_lam[0:5]
        new_lam[5:] = final_lam[5:]

        # Update the position with a perturbed mass
        new_pos = final_pos.copy()
        new_pos[-1] = (1 + delta_mass) * new_pos[-1]

        # Create new initial state with perturbed parameters
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
            continue  # Go back to the while loop to redetermine p_sol2

        # If the thread finished in time, extract the result
        Sf = result_container['value']
        bundle_Sf = Sf.y

        # Convert the state vectors to position and velocity
        r_bundle, v_bundle = mee2rv.mee2rv(bundle_Sf[0, :], bundle_Sf[1, :], bundle_Sf[2, :], bundle_Sf[3, :], bundle_Sf[4, :], bundle_Sf[5, :], mu)

        # Store the results in the bundles
        x_bundle = r_bundle[:, 0]
        y_bundle = r_bundle[:, 1]
        z_bundle = r_bundle[:, 2]

        # Store the results in the corresponding arrays
        bundle_array_size = bundle_Sf.shape
        S_bundles[0:bundle_array_size[1], 0:bundle_array_size[0], index] += bundle_Sf.T
        bundle_array_size = r_bundle.shape
        r_bundles[0:bundle_array_size[0], 0:bundle_array_size[1], index] += r_bundle
        v_bundles[0:bundle_array_size[0], 0:bundle_array_size[1], index] += v_bundle

    # Return the computed trajectory and bundle data
    return r_tr, v_tr, S_bundles, r_bundles, v_bundles
