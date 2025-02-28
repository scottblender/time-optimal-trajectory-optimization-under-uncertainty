import joblib

# Load the data
data = joblib.load('data.pkl')
trajectories = data['trajectories']
P_combined_history = data['P_combined_history']
