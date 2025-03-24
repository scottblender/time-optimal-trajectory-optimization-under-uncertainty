import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the saved data from the .pkl file
data = joblib.load('data.pkl')

# Extract the relevant data
X = data['X']
y = data['y']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
