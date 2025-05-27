import scipy.integrate

def run_solve_ivp(func, t_span, y0, method, t_eval, rtol, atol, result_container):
    """
    Solves an initial value problem (IVP) using the `scipy.integrate.solve_ivp` function and stores
    the result in a dictionary provided by the user.

    Parameters:
    - func: The function that defines the system of ODEs (dy/dt = func(t, y)).
    - t_span: A tuple specifying the time interval for the solution, typically (t_start, t_end).
    - y0: The initial state (vector) at the start time t_start.
    - method: The numerical integration method to use, e.g., 'RK45', 'DOP853', etc.
    - t_eval: The times at which the solution should be evaluated (an array of time points).
    - rtol: The relative tolerance for the solver's error estimation (controls accuracy).
    - atol: The absolute tolerance for the solver's error estimation (controls accuracy).
    - result_container: A dictionary where the solution will be stored. The solution will be placed
      under the key 'value'.

    Returns:
    - None: The function modifies the `result_container` dictionary in-place, storing the solution.
    """
    
    # Step 1: Use the solve_ivp function from scipy.integrate to solve the ODE system.
    # The result of the ODE integration is stored in the 'value' key of the result_container dictionary.
    result_container['value'] = scipy.integrate.solve_ivp(
        func,             # The function defining the ODE system
        t_span,           # Time interval over which to solve the ODE (start and end times)
        y0,               # Initial state (initial values of the ODE)
        method=method,    # Integration method (e.g., 'RK45', 'LSODA', etc.)
        t_eval=t_eval,    # Times at which the solution is computed
        rtol=rtol,        # Relative tolerance for the integration
        atol=atol         # Absolute tolerance for the integration
    )
