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

def compute_bundle_trajectory_params(p_sol, s0, tfound, mu, F, c, m0, g0, R_V_0, V_V_0, DU, TU, num_bundles, time_resolution_minutes=60):
    """
    Computes the spacecraft trajectory and generates perturbed trajectory bundles for Monte Carlo sampling.
    Now also returns mass history using mass directly from integration.
    """
    step_seconds = time_resolution_minutes * 60  # 180 minutes → 10800 seconds
    resolution_nd = step_seconds / TU            # Convert to non-dimensional units

    # build tspan array
    num_steps = int(round(tfound / resolution_nd)) + 1
    tspan = np.linspace(0, tfound, num_steps)

    # Extract the initial trajectory solution parameters (lam_sol)
    lam_sol = p_sol[0:7]
    
    # Combine initial state and lam_sol parameters to define the full initial state for the spacecraft
    S = np.append(s0, lam_sol)

    # Define the ODE function to model spacecraft dynamics
    func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)

    # Integrate spacecraft trajectory using Runge-Kutta 45
    Sf = scipy.integrate.solve_ivp(func, [0, tfound], S, method='RK45', rtol=1e-3, atol=1e-6, t_eval=tspan)

    # Extract position and velocity from the integrated state vector
    state = Sf.y[0:7, :]
    r_tr, v_tr = mee2rv.mee2rv(state[0, :], state[1, :], state[2, :], state[3, :], state[4, :], state[5, :], mu)

    # Directly use the mass history from integration results
    mass_tr = Sf.y[6, :]  # No need to integrate m_dot manually

    # Preserve all existing code
    y0V = np.append(R_V_0, V_V_0)
    mu_sun = 1.32712440018e20
    r = 149597870700

    tspan_target = [0, 3.5 * 365.25 * 86400]
    func_target = lambda t, y: twobody.twobody(t, y, mu_sun)
    yV = scipy.integrate.solve_ivp(func_target, tspan_target, y0V, method='RK45', rtol=1e-3, atol=1e-6)
    yV = yV.y / DU

    final_pos = Sf.y[0:7, -1]
    final_lam = Sf.y[7:14, -1]

    S_bundles = np.zeros((num_steps, 14, num_bundles))
    r_bundles = np.zeros((num_steps, 3, num_bundles))
    v_bundles = np.zeros((num_steps, 3, num_bundles))
    new_lam_bundles = np.zeros((num_steps, 7, num_bundles))
    mass_bundles = np.zeros((num_steps, num_bundles))  # Store mass history

    backTspan = np.linspace(tfound, 0, num_steps)

    alpha = 1
    max_angle = 1e-5

    for index in range(S_bundles.shape[2]):
        normfval2 = 1
        while normfval2 >= 1e-4:
            order_mag = np.floor(np.log10(np.abs(final_lam[:5])))
            order_mag = 10 ** order_mag
            delta_values = alpha * order_mag * np.random.rand()
            delta_mass = 0.4 * np.random.rand() - 0.2

            data = (final_lam, max_angle)
            p_sol2 = scipy.optimize.fsolve(calc_residuals_bundle.calc_residuals_bundle, delta_values, args=data, xtol=1e-10, maxfev=10000)
            normfval2 = np.linalg.norm(calc_residuals_bundle.calc_residuals_bundle(p_sol2, final_lam, max_angle))

        p_sol2 = np.array(p_sol2)
        new_lam = np.zeros(7)
        new_lam[0:5] = p_sol2[0:5] + final_lam[0:5]
        new_lam[5:] = final_lam[5:]
        new_pos = final_pos.copy()
        new_pos[-1] = (1 + delta_mass) * new_pos[-1]
        y_pert = np.append(new_pos, new_lam)

        func = lambda t, x: odefunc.odefunc(t, x, mu, F, c, m0, g0)

        result_container = {}
        thread = Thread(target=run_solve_ivp.run_solve_ivp, args=(func, [tfound, 0], y_pert, 'RK45', backTspan, 1e-3, 1e-6, result_container))
        start_time = time.time()
        thread.start()
        thread.join(timeout=10)

        if thread.is_alive():
            print(f"Timeout exceeded for bundle {index}, skipping.")
            thread.join()
            continue

        Sf = result_container['value']
        bundle_Sf = Sf.y

        r_bundle, v_bundle = mee2rv.mee2rv(bundle_Sf[0, :], bundle_Sf[1, :], bundle_Sf[2, :], 
                                           bundle_Sf[3, :], bundle_Sf[4, :], bundle_Sf[5, :], mu)

        # Directly store the mass history
        mass_pert = bundle_Sf[6, :]

        S_bundles[:, :, index] = bundle_Sf.T
        r_bundles[:, :, index] = r_bundle
        v_bundles[:, :, index] = v_bundle
        new_lam_bundles[:, :, index] = bundle_Sf[7:, :].T
        mass_bundles[:, index] = mass_pert  # Store mass history

    return r_tr, v_tr, mass_tr, S_bundles, r_bundles, v_bundles, new_lam_bundles, mass_bundles, backTspan
