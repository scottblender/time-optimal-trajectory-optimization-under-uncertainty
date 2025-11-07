import scipy
import numpy as np
nsd = 6  # Dimensionality of the state (3D position and 3D velocity combined)
beta = 2.  # UKF parameter
kappa = float(3 - nsd)  # UKF parameter
alpha = 0.5  # UKF parameter
lambda_ = alpha**2 * (nsd + kappa) - nsd  # UKF scaling parameter
sigmas = np.zeros((2*nsd+1, nsd))
print(sigmas.shape)
P_combined = np.block([
    [np.eye(nsd//2) * 0.01, np.zeros((nsd//2, nsd//2))],  # Position covariance with zero velocity covariance
    [np.zeros((nsd//2, nsd//2)), np.eye(nsd//2) * 0.001]  # Velocity covariance
])
U = scipy.linalg.cholesky((nsd+lambda_)*P_combined) # cholesky decomposition
X = np.array([0,1.5,2, -0.5, -0.3, 0]) # initial state
sigmas[0] = X
for k in range (nsd):
   sigmas[k+1] =  X + U[k]
   sigmas[nsd+k+1] = X- U[k]
print(sigmas)