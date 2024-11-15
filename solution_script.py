import numpy as np
import matplotlib.pyplot as plt
import compute_nominal_trajectory_params
import compute_bundle_trajectory_params
import evaluate_bundle_widths

p_sol, tfound, s0, mu, F, c, m0, g0, R_V_0, V_V_0, DU = compute_nominal_trajectory_params.compute_nominal_trajectory_params()
r_tr, v_tr, S_bundles, r_bundles, v_bundles = compute_bundle_trajectory_params.compute_bundle_trajectory_params(p_sol, s0, tfound, mu, F, c, m0, g0, R_V_0, V_V_0, DU)
r_nom = r_tr[::-1]
bundle_counts = np.array([r_bundles.shape[2]])
max_widths = evaluate_bundle_widths.evaluate_bundle_widths(bundle_counts, r_nom, r_bundles, tfound)
x = r_nom[:,0]
y = r_nom[:,1]
z = r_nom[:,2]
x_bundle = r_bundles[:,0, 1]
y_bundle = r_bundles[:,1, 1]
z_bundle = r_bundles[:, 2,1]
fig = plt.figure(figsize=(12, 12))
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
# Plot the data
ax.plot(x, y, z, label='Nominal Trajectory', color='r')
ax.plot(x_bundle, y_bundle, z_bundle, label='Perturbed Trajectory', color='b')
# Set labels
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title('3D Trajectory Plot')

# Optional: Add a legend
ax.legend()

# Show the plot
plt.show()