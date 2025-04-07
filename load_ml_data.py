import joblib
import pandas as pd

# Load the saved data
data = joblib.load('data.pkl')

# Extract X, y, and time match indices
X = data['X']
y = data['y']

# Define column names
X_columns = [
    'time',  # Time information
    'p', 'f', 'g', 'h', 'k', 'L', 'm',  # MEE elements + mass
    'cov_p', 'cov_f', 'cov_g', 'cov_h', 'cov_k', 'cov_L', 'cov_m',  # Covariance diagonals
    'bundle_idx', 'sigma_point_idx'  # New metadata columns
]

y_columns = ['lambda_p', 'lambda_f', 'lambda_g', 'lambda_h', 'lambda_k', 'lambda_L', 'lambda_m']

# Convert to DataFrames
df_X = pd.DataFrame(X, columns=X_columns)
df_y = pd.DataFrame(y, columns=y_columns)

# Display sample output
print("First few rows of X:")
print(df_X.head())

print("\nFirst few rows of y:")
print(df_y.head())

