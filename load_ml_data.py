import joblib
import pandas as pd

# Load the saved data
data = joblib.load('data.pkl')

# Extract X, y, and time history (assuming time is included in X)
X = data['X']
y = data['y']
time_match_indices = data['time_match_indices']

# Define column names for modified equinoctial elements (MEE)
X_columns = [
    'time',  # Time information
    'p', 'f', 'g', 'h', 'k', 'L', 'm',  # MEE elements + mass
    'cov_p', 'cov_f', 'cov_g', 'cov_h', 'cov_k', 'cov_L', 'cov_m',  # Covariance diagonal elements
    'mean_p', 'mean_f', 'mean_g', 'mean_h', 'mean_k', 'mean_L', 'mean_m'  # Mean values
]

y_columns = ['lambda_p', 'lambda_f', 'lambda_g', 'lambda_h', 'lambda_k', 'lambda_L', 'lambda_m']  # Costates in MEE

# Add the time information as the first column in X
df_X = pd.DataFrame(X, columns=X_columns)

# Create DataFrame for y (control states)
df_y = pd.DataFrame(y, columns=y_columns)

# Display first few rows
print("First few rows of X:")
print(df_X.head())

print("\nFirst few rows of y:")
print(df_y.head())
