import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor

# --- Configuration ---
# Specify the path to your trained model file
MODEL_FILENAME = "trained_model_min_corrective.pkl" 
OUTPUT_IMAGE_FILENAME = "feature_importance_plot.png"

# --- Main Script ---
print(f"Loading model from: {MODEL_FILENAME}")
try:
    # Load the entire model object from the file
    model = joblib.load(MODEL_FILENAME)
except FileNotFoundError:
    print(f"[ERROR] Model file not found at '{MODEL_FILENAME}'. Please ensure the path is correct.")
    exit()

# The model is a MultiOutputRegressor, which wraps individual LightGBM models.
# We need to access one of the underlying estimators to plot its importance.
# Let's use the first one (for the first costate variable).
if isinstance(model, MultiOutputRegressor) and len(model.estimators_) > 0:
    first_estimator = model.estimators_[0]
    
    print("Generating feature importance plot...")
    
    # Create the plot using LightGBM's built-in function
    ax = lgb.plot_importance(
        first_estimator, 
        figsize=(12, 10), 
        importance_type='gain', # 'gain' is usually the most informative
        title=f"Feature Importance for {MODEL_FILENAME}"
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE_FILENAME)
    
    print(f"Successfully saved plot to: {OUTPUT_IMAGE_FILENAME}")

else:
    print("[ERROR] The loaded object is not a trained MultiOutputRegressor or has no estimators.")