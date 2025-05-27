import numpy as np

def wrapTo2Pi(angle):
    """
    Wraps the input angle to the range [0, 2π). This is useful for ensuring that angles
    representing circular motion stay within the standard interval of a full circle.

    Parameters:
    - angle: The input angle (radians). This can be any real number, including angles
      greater than 2π or less than 0, and it will be normalized to the range [0, 2π).

    Returns:
    - Wrapped angle: The input angle wrapped to the interval [0, 2π), ensuring it stays
      within one full rotation of a circle (0 ≤ angle < 2π).
    """

    # Step 1: Compute the remainder of angle divided by 2π to normalize it
    # The modulo operation will map the angle to the interval [0, 2π)
    return angle % (2 * np.pi)
