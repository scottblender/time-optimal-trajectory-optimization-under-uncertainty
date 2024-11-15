import numpy as np

def twobody(t, y, mu):
    """
    Computes the differential equations for a two-body problem, describing the motion of an object
    under the influence of gravitational attraction from a central body (e.g., a planet or star).
    The system is described by Newton's law of gravitation and the equations of motion for position
    and velocity in three dimensions.

    Parameters:
    - t: The current time (not used in this specific case but required for the ODE solver).
    - y: A vector containing the current state of the system. This vector has the following elements:
      - y[0], y[1], y[2]: Position components (rx, ry, rz) in 3D space (in km).
      - y[3], y[4], y[5]: Velocity components (vx, vy, vz) in 3D space (in km/s).
    - mu: The gravitational parameter (GM) of the central body (in km^3/s^2).

    Returns:
    - f: A vector of the time derivatives of the system state (i.e., the right-hand side of the ODEs):
      - f[0], f[1], f[2]: Velocity components (vx, vy, vz) in 3D space.
      - f[3], f[4], f[5]: Acceleration components (ax, ay, az) due to gravitational force.
    """

    # Step 1: Extract the position and velocity components from the state vector `y`
    rx = y[0]  # Position in the x-direction (km)
    ry = y[1]  # Position in the y-direction (km)
    rz = y[2]  # Position in the z-direction (km)
    vx = y[3]  # Velocity in the x-direction (km/s)
    vy = y[4]  # Velocity in the y-direction (km/s)
    vz = y[5]  # Velocity in the z-direction (km/s)

    # Step 2: Calculate the distance (magnitude of the position vector) from the central body
    r = np.sqrt(rx**2 + ry**2 + rz**2)  # Distance (in km)

    # Step 3: Initialize the output array `f`, which will store the derivatives of position and velocity
    f = np.zeros(6)  # Array to store the derivatives of the state (6 elements)

    # Step 4: Define the equations of motion for the system (two-body problem)
    # The velocity components are simply the derivatives of the position components:
    f[0] = vx  # dx/dt = vx
    f[1] = vy  # dy/dt = vy
    f[2] = vz  # dz/dt = vz

    # The acceleration components are calculated using Newton's law of gravitation:
    # a = - (GM / r^3) * r_hat, where r_hat is the unit vector in the direction of the position vector
    f[3] = -mu * rx / r**3  # Acceleration in the x-direction (km/s^2)
    f[4] = -mu * ry / r**3  # Acceleration in the y-direction (km/s^2)
    f[5] = -mu * rz / r**3  # Acceleration in the z-direction (km/s^2)

    # Step 5: Return the time derivatives of the system's state vector (position and velocity)
    return f
