import joblib
import pandas as pd

# --- Configuration ---
PROCESSED_DATA_FILENAME = "processed_training_data.pkl"
EXPECTED_TIME_POINTS = 101  # Based on n=50 (50 before + 50 after + 1 center point)

def check_time_points(data_key, data_dict):
    """
    Checks a single dataset (e.g., 'max_data') for the correct number of time points.

    Args:
        data_key (str): The name of the dataset (e.g., 'max_data').
        data_dict (dict): The dictionary containing the dataframe.

    Returns:
        bool: True if the check passes, False otherwise.
    """
    print(f"\n--- Checking {data_key.upper()} ---")
    try:
        df = data_dict['df']
        if not isinstance(df, pd.DataFrame):
            print(f"[ERROR] '{data_key}' does not contain a valid pandas DataFrame.")
            return False

        # Get the number of unique values in the 't' (time) column
        unique_times_count = df['t'].nunique()

        print(f"Found {unique_times_count} unique time points.")
        print(f"Expected {EXPECTED_TIME_POINTS} unique time points.")

        if unique_times_count == EXPECTED_TIME_POINTS:
            print("[STATUS] PASS: The number of time points is correct.")
            return True
        else:
            print(f"[STATUS] FAIL: Incorrect number of time points found.")
            return False

    except KeyError:
        print(f"[ERROR] Could not find the 'df' key in '{data_key}'.")
        return False
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        return False

def main():
    """
    Main function to load the data file and run the checks.
    """
    print(f"[INFO] Loading data from '{PROCESSED_DATA_FILENAME}'...")
    try:
        processed_data = joblib.load(PROCESSED_DATA_FILENAME)
    except FileNotFoundError:
        print(f"\n[FATAL ERROR] The file '{PROCESSED_DATA_FILENAME}' was not found.")
        print("Please make sure you have run '1_generate_datasets.py' successfully.")
        return
    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to load the data file: {e}")
        return

    # Check both the 'max_data' and 'min_data' entries in the file
    max_check_passed = check_time_points('max_data', processed_data.get('max_data', {}))
    min_check_passed = check_time_points('min_data', processed_data.get('min_data', {}))

    print("\n--- FINAL RESULT ---")
    if max_check_passed and min_check_passed:
        print("✅ SUCCESS: Both datasets have the correct number of unique time points.")
    else:
        print("❌ FAILURE: One or both datasets have an incorrect number of time points.")
        print("You may need to re-run '1_generate_datasets.py' with the corrected script.")

if __name__ == "__main__":
    main()