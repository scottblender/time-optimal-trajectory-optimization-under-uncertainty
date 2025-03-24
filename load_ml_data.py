import joblib
import pandas as pd

# Load the saved data
data = joblib.load('data.pkl')

# Extract X and y
X = data['X']
y = data['y']

# Define column names
X_columns = [
    'x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 
    'cov_x', 'cov_y', 'cov_z', 'cov_vx', 'cov_vy', 'cov_vz', 'cov_m',
    'mean_x', 'mean_y', 'mean_z', 'mean_vx', 'mean_vy', 'mean_vz', 'mean_m'
]

y_columns = ['lambda_x', 'lambda_y', 'lambda_z', 'lambda_vx', 'lambda_vy', 'lambda_vz', 'lambda_m']

# Convert to DataFrame
df_X = pd.DataFrame(X, columns=X_columns)
df_y = pd.DataFrame(y, columns=y_columns)

# Display first few rows
print("First few rows of X:")
print(df_X.head())

print("\nFirst few rows of y:")
print(df_y.head())
