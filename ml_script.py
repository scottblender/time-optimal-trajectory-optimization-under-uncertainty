import numpy as np
from scipy.integrate import solve_ivp
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import odefunc
import mee2rv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting

# Load the saved data from the .pkl file
data = joblib.load('data.pkl')

# Extract the relevant data
X = data['X']
y = data['y']
time_match_indices = data['time_match_indices']

# Train-test split
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, y, np.arange(len(X)), test_size=0.2, random_state=42
)

# Initialize and train the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# Predict the control states for both train and test sets
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Evaluate the model performance
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print results
print(f"Train Mean Squared Error: {train_mse:.6f}")
print(f"Test Mean Squared Error: {test_mse:.6f}")
print(f"Train R^2 Score: {train_r2:.6f}")
print(f"Test R^2 Score: {test_r2:.6f}")

# Save the trained model (optional)
joblib.dump(rf, 'random_forest_model.pkl')

# Feature Importance Plot
feature_importance = rf.feature_importances_
feature_names = [f"Feature {i+1}" for i in range(X.shape[1])]

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance, color='blue', alpha=0.7)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance from Random Forest")
plt.gca().invert_yaxis()  # Flip order for better visualization
plt.show()

# Define a function for propagation
def propagate_control(tstart, tend, state, control, mu, F, c, m0, g0):
    """
    Propagate the state using the given initial state and control.
    """
    print(f"Integrating from t = {tstart} to t = {tend} with control: {control}")  # Print the control being integrated
    
    # Combine state and control into one array for propagation
    S = np.concatenate([state, control])

    func = lambda t, S: odefunc.odefunc(t, S, mu, F, c, m0, g0)
    tspan = np.linspace(tstart, tend, 20)  # Time steps for propagation
    
    # Propagate using the given control
    Sf = solve_ivp(func, [tstart, tend], S, method='RK45', rtol=1e-6, atol=1e-8, t_eval=tspan)
    
    if Sf.success:
        return Sf.t, Sf.y.T  # Return time and state trajectory
    else:
        print("Propagation failed!")
        return None, None

# Randomly sample 6 pairs of indices from time_match_indices
np.random.seed(42)
num_pairs = 6
selected_pairs = []

# Define constants
mu = 27.899633640439433
F = 0.33
c = 4.4246246663455135
m0 = 4000
g0 = 9.81

# Ensure that we select pairs of consecutive indices within test set range
while len(selected_pairs) < num_pairs:
    rand_idx = np.random.randint(0, len(time_match_indices) - 1)
    start_idx, end_idx = time_match_indices[rand_idx], time_match_indices[rand_idx + 1]

    # Ensure indices belong to the test set
    if start_idx in test_indices and end_idx in test_indices:
        selected_pairs.append((start_idx, end_idx))

# Create subplots for multiple trajectories
fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw={'projection': '3d'})
plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Increase spacing between subplots

# Now propagate the actual and predicted controls for each selected pair
for i, (start_idx, end_idx) in enumerate(selected_pairs):
    
    # Get the corresponding indices in the test set
    test_start_idx = np.where(test_indices == start_idx)[0][0]
    test_end_idx = np.where(test_indices == end_idx)[0][0]

    # Retrieve times
    tstart = X[start_idx, 0]
    tend = X[end_idx, 0]

    # Retrieve the corresponding actual control and predicted control states
    actual_control = y[start_idx, :]
    predicted_control = y_test_pred[test_start_idx, :]  # Corrected index

    # Assume initial state comes from X
    initial_state = X[start_idx, 1:8]  # Extract state variables

    # Propagate the actual control
    print(f"Starting propagation with actual control at t = {tstart}")
    actual_time, actual_trajectory = propagate_control(tstart, tend, initial_state, actual_control, mu, F, c, m0, g0)
    
    # Propagate the predicted control
    print(f"Starting propagation with predicted control at t = {tstart}")
    predicted_time, predicted_trajectory = propagate_control(tstart, tend, initial_state, predicted_control, mu, F, c, m0, g0)
    
    # Convert MEE to ECI for positions
    if actual_trajectory is not None and predicted_trajectory is not None:
        # Convert both actual and predicted trajectories from MEE to ECI
        actual_r, actual_v = mee2rv.mee2rv(actual_trajectory[:, 0], actual_trajectory[:, 1], actual_trajectory[:, 2], 
                                           actual_trajectory[:, 3], actual_trajectory[:, 4], actual_trajectory[:, 5], mu)
        predicted_r, predicted_v = mee2rv.mee2rv(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], 
                                                 predicted_trajectory[:, 3], predicted_trajectory[:, 4], predicted_trajectory[:, 5], mu)
        
        # Select subplot
        ax = axes[i // 3, i % 3]

        # Plot actual trajectory
        ax.plot(actual_r[:, 0], actual_r[:, 1], actual_r[:, 2], label=f"Actual Control {i+1}", color='b')
        
        # Plot predicted trajectory
        ax.plot(predicted_r[:, 0], predicted_r[:, 1], predicted_r[:, 2], label=f"Predicted Control {i+1}", linestyle='--', color='r')

        # Labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title(f"Trajectory {i+1}")
        ax.legend()

plt.show()
